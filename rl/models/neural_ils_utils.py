"""Shared utilities for neural-guided Iterated Local Search.

This module isolates the logic that is common between
``scripts/neural_guided_ils_v2.py`` and
``scripts/run_neural_ils_v2_benchmark.py`` so that the model, search,
and move-conversion code stay in one place.
"""
from __future__ import annotations

import random
from typing import Any

import torch

from fjsp.graph.solution_graph import SolutionGraph, build_solution_graph
from fjsp.parser.fjs_parser import FJSPInstance
from fjsp.scheduler.local_search import (
    _assignments,
    _makespan,
    _neighbors_reassign,
    _neighbors_swap_machines,
    _neighbors_swap_order_across_machines,
    _neighbors_swap_same_machine,
    _recompute,
    critical_path_perturb,
    iterated_local_search,
)
from fjsp.scheduler.validator import earliest_finish_schedule
from rl.models.move_rank_net import MoveRankNet
from rl.models.neighborhood_move import NeighborhoodMove


NEIGHBORHOOD_FNS = {
    "reassign": _neighbors_reassign,
    "swap_machine": _neighbors_swap_machines,
    "swap_same_machine": _neighbors_swap_same_machine,
    "swap_order_across": _neighbors_swap_order_across_machines,
}


def graph_to_device(graph: SolutionGraph, device: torch.device) -> SolutionGraph:
    """Move a SolutionGraph's tensors to ``device``."""
    return SolutionGraph(
        operation_refs=graph.operation_refs,
        op_features=graph.op_features.to(device),
        machine_features=graph.machine_features.to(device),
        precedence_edges=graph.precedence_edges.to(device),
        machine_seq_edges=graph.machine_seq_edges.to(device),
        eligibility_edges=graph.eligibility_edges.to(device),
        eligibility_durations=graph.eligibility_durations.to(device),
        critical_op_indices=graph.critical_op_indices,
        global_features=graph.global_features.to(device),
    )


def load_net(checkpoint: str, device: torch.device) -> MoveRankNet:
    """Load a trained MoveRankNet from ``checkpoint``."""
    net = MoveRankNet(
        hidden_dim=128,
        num_mp_rounds=2,
        num_transformer_blocks=2,
        num_heads=8,
        ffn_dim=128,
    ).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.eval()
    return net


def neighbor_to_move(
    current: list[tuple[int, int, int]],
    neighbor: list[tuple[int, int, int]],
    neighborhood: str,
) -> NeighborhoodMove:
    """Convert a neighbour assignment list to a ``NeighborhoodMove``."""
    if neighborhood == "reassign":
        for i, (cur, nxt) in enumerate(zip(current, neighbor)):
            if cur[2] != nxt[2]:
                return NeighborhoodMove(
                    move_type="reassign",
                    op_indices=(i,),
                    target_machines=(nxt[2],),
                )
        raise ValueError("reassign neighbor has no machine change")

    diffs = [i for i, (cur, nxt) in enumerate(zip(current, neighbor)) if cur != nxt]
    if len(diffs) != 2:
        diffs = [i for i in range(len(current)) if current[i] != neighbor[i]]
    if len(diffs) != 2:
        raise ValueError(f"expected 2 differing positions, got {len(diffs)}")
    i, j = diffs
    return NeighborhoodMove(
        move_type=neighborhood,
        op_indices=(i, j),
        target_machines=(neighbor[i][2], neighbor[j][2]),
    )


def sample_candidates(
    instance: FJSPInstance,
    assignments: list[tuple[int, int, int]],
    max_per_type: int | None,
    rng: random.Random,
) -> tuple[list[NeighborhoodMove], list[list[tuple[int, int, int]]]]:
    """Generate a capped candidate set and return moves + neighbor assignments."""
    moves: list[NeighborhoodMove] = []
    neighbors: list[list[tuple[int, int, int]]] = []
    for name, fn in NEIGHBORHOOD_FNS.items():
        try:
            cands = fn(instance, assignments)
        except Exception:
            continue
        if max_per_type is not None and len(cands) > max_per_type:
            cands = rng.sample(cands, max_per_type)
        for cand in cands:
            try:
                move = neighbor_to_move(assignments, cand, name)
                moves.append(move)
                neighbors.append(cand)
            except Exception:
                continue
    return moves, neighbors


def neural_guided_ils_single_run(
    instance: FJSPInstance,
    net: MoveRankNet,
    device: torch.device,
    *,
    seed: int = 0,
    n_iterations: int = 30,
    top_k: int = 20,
    max_candidates_per_type: int | None = 200,
    patience: int = 5,
    perturb_n_swaps: int = 5,
    fallback_ils: bool = True,
    accept_worse_prob: float = 0.0,
) -> tuple[list[Any], int, list[int]]:
    """Run one neural-guided ILS trajectory.

    Parameters
    ----------
    accept_worse_prob
        Probability of accepting a non-improving neural short-list move
        (only if it is the best ranked move).  Acts as a cheap SA-style
        diversification without evaluating the full neighborhood.

    Returns
    -------
    best_schedule, best_makespan, history
    """
    rng = random.Random(seed)

    schedule = earliest_finish_schedule(instance)
    assignments = _assignments(schedule)
    current_ms = _makespan(instance, assignments)
    best_assignments = list(assignments)
    best_ms = current_ms
    history = [best_ms]

    no_improve_count = 0
    iteration = 0
    while iteration < n_iterations:
        moves, neighbors = sample_candidates(
            instance, assignments, max_candidates_per_type, rng
        )
        if not moves:
            break

        graph = build_solution_graph(instance, _recompute(instance, assignments))
        graph = graph_to_device(graph, device)
        with torch.no_grad():
            scores = net(graph, moves)
        ranked = torch.argsort(scores, descending=True).tolist()

        # Best-improvement within the neural short-list.
        best_neighbor = None
        best_neighbor_ms = current_ms
        best_ranked_idx = None
        for idx in ranked[:top_k]:
            neighbor = neighbors[idx]
            try:
                new_ms = _makespan(instance, neighbor)
            except Exception:
                continue
            if new_ms < best_neighbor_ms:
                best_neighbor_ms = new_ms
                best_neighbor = neighbor
                best_ranked_idx = idx

        accepted = best_neighbor is not None and best_neighbor_ms < current_ms
        # SA-style acceptance of the top-1 move when it is non-improving.
        if not accepted and best_neighbor is not None and rng.random() < accept_worse_prob:
            accepted = True

        if accepted:
            assignments = best_neighbor
            current_ms = best_neighbor_ms

        history.append(current_ms)

        if current_ms < best_ms:
            best_ms = current_ms
            best_assignments = list(assignments)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if not accepted or no_improve_count >= patience:
            perturbed = critical_path_perturb(
                instance,
                _recompute(instance, best_assignments),
                n_swaps=perturb_n_swaps,
                seed=rng.randint(0, 1_000_000),
            )
            assignments = _assignments(perturbed)
            current_ms = _makespan(instance, assignments)
            no_improve_count = 0
            iteration += 1

    if fallback_ils:
        # ``local_search`` knows ``swap_order`` (same-machine swap) rather
        # than ``swap_same_machine``.
        fallback_neighborhoods = (
            "reassign",
            "swap_machine",
            "swap_order",
            "swap_order_across",
        )
        final_schedule, final_ms = iterated_local_search(
            instance,
            _recompute(instance, best_assignments),
            n_iterations=20,
            perturb_n_swaps=5,
            max_ls_iterations=100,
            neighborhoods=fallback_neighborhoods,
            perturb_mode="critical",
            strategy="best",
        )
        if final_ms < best_ms:
            best_ms = final_ms
            best_assignments = _assignments(final_schedule)
            history.append(best_ms)

    best_schedule = _recompute(instance, best_assignments)
    return best_schedule, best_ms, history
