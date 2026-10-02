"""BARI Improvement Environment for FJSP.

Bottleneck-Aware Reinforced Improvement (BARI) environment.

This environment operates on *complete* FJSP solutions. Instead of
constructing a schedule from scratch, the agent starts from an initial
solution and makes improvement moves to reduce the makespan.

Core innovations
----------------
1. **Coupled neighborhood**: ``coupled_reassign`` moves combine machine
   reassignment with intelligent insertion. When moving an operation to
   a new machine, the environment tries all insertion positions and
   selects the one yielding the lowest makespan. This mirrors how human
   schedulers think: "move this operation to that machine AND put it in
   the best slot." Prior work (L2S, ICLR 2024) only does sequencing
   swaps for JSSP; traditional heuristics do isolated reassign without
   considering insertion position.

2. **Bottleneck contribution score**: Instead of a binary critical-path
   mask, each operation receives a continuous score reflecting how much
   it contributes to the bottleneck. This guides the agent to focus on
   high-impact operations.

3. **Improvement potential filtering**: Before generating candidate
   moves, the environment estimates an upper bound on the makespan
   reduction for each operation. Only operations with non-trivial
   potential are considered, reducing the action space and improving
   learning efficiency.

Reward
------
Dense per-step reward: ``-(new_makespan - old_makespan)``.
A negative makespan change (improvement) yields a positive reward.

Episode termination
-------------------
The episode ends after ``max_steps`` improvement steps, or when no
improving move is found for ``patience`` consecutive steps.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Literal, TypedDict

from fjsp.graph.solution_graph import SolutionGraph, build_solution_graph, find_critical_path
from fjsp.parser.fjs_parser import FJSPInstance, parse_fjs
from fjsp.scheduler.validator import ScheduledOperation, earliest_finish_schedule, validate_schedule
from fjsp.scheduler.local_search import _assignments, _recompute


@dataclass(frozen=True)
class ImprovementMove:
    """A single improvement move in the FJSP solution space.

    Move types:
    - ``coupled_reassign``: Move op to a different machine AND find the
      best insertion position on the target machine. This is the key
      BARI innovation — coupling reassignment with insertion.
    - ``swap_prev``: Swap op with its predecessor on the same machine.
    - ``swap_next``: Swap op with its successor on the same machine.
    """

    move_type: Literal["coupled_reassign", "swap_prev", "swap_next"]
    op_index: int  # index into the assignments list
    target_machine: int | None = None  # for coupled_reassign moves

    @property
    def is_reassign(self) -> bool:
        return self.move_type == "coupled_reassign"


class ImprovementObservation(TypedDict):
    graph: SolutionGraph
    valid_moves: list[ImprovementMove]
    makespan: int
    step: int
    bottleneck_scores: list[float]  # per-operation bottleneck contribution


class FJSPImprovementEnv:
    """Environment for Bottleneck-Aware Reinforced Improvement of FJSP.

    Parameters
    ----------
    instance
        The FJSP instance to solve.
    initial_schedule
        Starting solution. If ``None``, earliest-finish schedule is used.
    max_steps
        Maximum number of improvement steps per episode.
    patience
        Stop early if no improvement for this many consecutive steps.
    no_improve_penalty
        Reward penalty for a step that does not improve the makespan.
    potential_threshold
        Minimum improvement potential for an operation to be considered.
        Operations with potential below this threshold are filtered out.
    """

    def __init__(
        self,
        instance: FJSPInstance,
        *,
        initial_schedule: list[ScheduledOperation] | None = None,
        max_steps: int = 100,
        patience: int = 20,
        no_improve_penalty: float = 0.0,
        potential_threshold: float = 0.0,
        reward_shaping: str = "dense",
    ) -> None:
        """Initialize the environment.

        Parameters
        ----------
        reward_shaping : {"sparse", "dense", "potential"}
            - sparse: reward = makespan_delta (current code).
            - dense: reward = -0.01 * (new_makespan / initial_makespan) + makespan_delta.
            - potential: reward based on distance to best known lower bound.
        """
        self.instance = instance
        self._initial_schedule = initial_schedule
        self.max_steps = max_steps
        self.patience = patience
        self.no_improve_penalty = no_improve_penalty
        self.potential_threshold = potential_threshold
        self.reward_shaping = reward_shaping
        self._initial_makespan = 1
        self.reset()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        initial_schedule: list[ScheduledOperation] | None = None,
        max_steps: int = 100,
        patience: int = 20,
        no_improve_penalty: float = 0.0,
        potential_threshold: float = 0.0,
        reward_shaping: str = "dense",
    ) -> "FJSPImprovementEnv":
        instance = parse_fjs(path)
        if initial_schedule is None:
            initial_schedule = earliest_finish_schedule(instance)
        return cls(
            instance,
            initial_schedule=initial_schedule,
            max_steps=max_steps,
            patience=patience,
            no_improve_penalty=no_improve_penalty,
            potential_threshold=potential_threshold,
            reward_shaping=reward_shaping,
        )

    @property
    def done(self) -> bool:
        return self._done

    @property
    def makespan(self) -> int:
        return self._makespan

    @property
    def best_makespan(self) -> int:
        return self._best_makespan

    @property
    def schedule(self) -> list[ScheduledOperation]:
        return list(self._schedule)

    @property
    def step_count(self) -> int:
        return self._step

    def reset(self, initial_schedule: list[ScheduledOperation] | None = None) -> ImprovementObservation:
        """Reset the environment."""
        if initial_schedule is not None:
            self._schedule = list(initial_schedule)
        elif self._initial_schedule is not None:
            self._schedule = list(self._initial_schedule)
        else:
            self._schedule = earliest_finish_schedule(self.instance)

        self._assignments = _assignments(self._schedule)
        self._makespan = max((o.end for o in self._schedule), default=0)
        self._initial_makespan = self._makespan
        self._best_makespan = self._makespan
        self._step = 0
        self._no_improve_count = 0
        self._done = False

        self._build_machine_sequences()
        self._bottleneck_scores = self._compute_bottleneck_scores()
        self._valid_moves = self._compute_valid_moves()

        return self.observe()

    def observe(self) -> ImprovementObservation:
        """Build the current observation."""
        graph = build_solution_graph(self.instance, self._schedule)
        return {
            "graph": graph,
            "valid_moves": list(self._valid_moves),
            "makespan": self._makespan,
            "step": self._step,
            "bottleneck_scores": list(self._bottleneck_scores),
        }

    def step(self, action: int | ImprovementMove) -> tuple[ImprovementObservation, float, bool, dict]:
        """Apply an improvement move."""
        if self._done:
            return self.observe(), 0.0, True, {"error": "episode already finished"}

        if isinstance(action, int):
            if action < 0 or action >= len(self._valid_moves):
                return self.observe(), -1.0, False, {"invalid_action": True}
            move = self._valid_moves[action]
        else:
            move = action

        old_makespan = self._makespan

        new_assignments = self._apply_move(move)

        try:
            new_schedule = _recompute(self.instance, new_assignments)
            validation = validate_schedule(self.instance, new_schedule)
            if not validation.is_valid:
                return self.observe(), -0.5, False, {"invalid_schedule": True, "errors": validation.errors}
        except ValueError:
            return self.observe(), -0.5, False, {"decode_error": True}

        self._assignments = new_assignments
        self._schedule = new_schedule
        self._makespan = validation.makespan
        self._step += 1

        if self._makespan < self._best_makespan:
            self._best_makespan = self._makespan

        delta = old_makespan - self._makespan
        reward = float(delta)
        if self.reward_shaping == "dense":
            # Dense survival reward: small negative pressure proportional to
            # current makespan, plus any improvement delta. This keeps the
            # signal alive even when no improvement happens.
            reward += -0.01 * (self._makespan / max(self._initial_makespan, 1))
        elif self.reward_shaping == "potential":
            # Potential-based reward: distance to a rough lower bound.
            lb = self._lower_bound_makespan()
            old_potential = max(0.0, old_makespan - lb) / max(old_makespan, 1)
            new_potential = max(0.0, self._makespan - lb) / max(self._makespan, 1)
            reward = old_potential - new_potential

        if delta <= 0:
            self._no_improve_count += 1
            reward -= self.no_improve_penalty
        else:
            self._no_improve_count = 0

        self._build_machine_sequences()
        self._bottleneck_scores = self._compute_bottleneck_scores()
        self._valid_moves = self._compute_valid_moves()

        if self._step >= self.max_steps:
            self._done = True
        elif self._no_improve_count >= self.patience:
            self._done = True
        elif not self._valid_moves:
            self._done = True

        info: dict = {
            "makespan": self._makespan,
            "best_makespan": self._best_makespan,
            "delta": delta,
            "step": self._step,
        }
        return self.observe(), reward, self._done, info

    def _build_machine_sequences(self) -> None:
        """Build machine sequence mapping from current schedule."""
        self._by_machine: dict[int, list[ScheduledOperation]] = defaultdict(list)
        for sop in self._schedule:
            self._by_machine[sop.machine].append(sop)
        for m in self._by_machine:
            self._by_machine[m].sort(key=lambda o: o.start)

        self._assignment_index: dict[tuple[int, int], int] = {}
        for i, (j, o, _) in enumerate(self._assignments):
            self._assignment_index[(j, o)] = i

    def _lower_bound_makespan(self) -> int:
        """Compute a simple lower bound for potential-based reward shaping.

        LB = max(max machine min-load, max job total-min-duration).
        """
        # Machine lower bound: for each machine, sum the shortest eligible
        # processing time of every operation.
        machine_lb = 0
        for m in range(self.instance.machine_count):
            total = 0
            for job in self.instance.jobs:
                for op in job.operations:
                    eligible = [opt.duration for opt in op.options if opt.machine == m]
                    if eligible:
                        total += min(eligible)
            machine_lb = max(machine_lb, total)

        # Job lower bound: sum of min durations per operation
        job_lb = 0
        for job in self.instance.jobs:
            total = sum(min(opt.duration for opt in op.options) for op in job.operations)
            job_lb = max(job_lb, total)

        return max(machine_lb, job_lb)

    def _compute_bottleneck_scores(self) -> list[float]:
        """Compute continuous bottleneck contribution score for each operation.

        The score combines:
        - Critical path membership (binary, but weighted by depth)
        - Machine load ratio (how busy the assigned machine is)
        - Slack ratio (how much idle time around the operation)

        Operations with high scores are bottlenecks: they are on the
        critical path, on busy machines, with little slack.
        """
        _, critical_indices = find_critical_path(self._schedule, self.instance)
        critical_set = set()
        for idx in critical_indices:
            if idx < len(self._schedule):
                sop = self._schedule[idx]
                critical_set.add((sop.job, sop.op))

        # Compute machine loads
        machine_loads: dict[int, int] = {}
        for m, ops in self._by_machine.items():
            machine_loads[m] = sum(o.end - o.start for o in ops)
        max_load = max(machine_loads.values()) if machine_loads else 1

        # Compute makespan for normalization
        makespan = self._makespan if self._makespan > 0 else 1

        scores = []
        for sop in self._schedule:
            is_cp = 1.0 if (sop.job, sop.op) in critical_set else 0.0
            load_ratio = machine_loads.get(sop.machine, 0) / max_load if max_load > 0 else 0.0

            # Slack: idle time before and after on the machine
            m_ops = self._by_machine.get(sop.machine, [])
            idle_before = 0
            idle_after = 0
            for i, mo in enumerate(m_ops):
                if mo.job == sop.job and mo.op == sop.op:
                    if i > 0:
                        idle_before = sop.start - m_ops[i - 1].end
                    if i < len(m_ops) - 1:
                        idle_after = m_ops[i + 1].start - sop.end
                    break
            slack = max(0, idle_before) + max(0, idle_after)
            slack_ratio = 1.0 - min(1.0, slack / makespan) if makespan > 0 else 0.0

            # Bottleneck score: high when on CP, on busy machine, low slack
            score = is_cp * (0.4 + 0.3 * load_ratio + 0.3 * slack_ratio)
            scores.append(score)

        return scores

    def _compute_improvement_potential(self, sop: ScheduledOperation) -> float:
        """Estimate upper bound on makespan reduction if this operation
        were removed from the critical path.

        This is the difference between:
        - Current critical path length (makespan)
        - Longest path not passing through this operation

        A simplified estimate: if the operation is on the critical path,
        the potential is proportional to its duration plus surrounding
        idle time. If not on the critical path, potential is 0.
        """
        _, critical_indices = find_critical_path(self._schedule, self.instance)
        critical_set = set()
        for idx in critical_indices:
            if idx < len(self._schedule):
                cp_sop = self._schedule[idx]
                critical_set.add((cp_sop.job, cp_sop.op))

        if (sop.job, sop.op) not in critical_set:
            return 0.0

        # Duration of the operation
        duration = sop.end - sop.start

        # Idle time around the operation on its machine
        m_ops = self._by_machine.get(sop.machine, [])
        idle_before = 0
        idle_after = 0
        for i, mo in enumerate(m_ops):
            if mo.job == sop.job and mo.op == sop.op:
                if i > 0:
                    idle_before = max(0, sop.start - m_ops[i - 1].end)
                if i < len(m_ops) - 1:
                    idle_after = max(0, m_ops[i + 1].start - sop.end)
                break

        # Potential: if we move this op elsewhere, the machine frees up
        # (duration + idle_before + idle_after) time. But the op still
        # needs to run somewhere, so the net gain is bounded by the
        # idle time that gets absorbed.
        potential = float(idle_before + idle_after)
        return potential

    def _compute_valid_moves(self) -> list[ImprovementMove]:
        """Enumerate valid improvement moves using BARI's approach.

        1. Compute bottleneck scores for all operations.
        2. Filter to operations with non-trivial improvement potential.
        3. For each candidate, generate coupled_reassign + swap moves.
        """
        _, critical_indices = find_critical_path(self._schedule, self.instance)
        critical_set = set()
        for idx in critical_indices:
            if idx < len(self._schedule):
                sop = self._schedule[idx]
                critical_set.add((sop.job, sop.op))

        # Expand candidate set: critical-path ops + their machine neighbors
        candidate_set = set(critical_set)
        for sop in self._schedule:
            if (sop.job, sop.op) not in critical_set:
                continue
            m_ops = self._by_machine.get(sop.machine, [])
            for i, mo in enumerate(m_ops):
                if mo.job == sop.job and mo.op == sop.op:
                    if i > 0:
                        prev = m_ops[i - 1]
                        candidate_set.add((prev.job, prev.op))
                    if i < len(m_ops) - 1:
                        nxt = m_ops[i + 1]
                        candidate_set.add((nxt.job, nxt.op))
                    break

        # Filter by improvement potential (coarse-to-fine)
        if self.potential_threshold > 0:
            filtered_set = set()
            for sop in self._schedule:
                if (sop.job, sop.op) not in candidate_set:
                    continue
                potential = self._compute_improvement_potential(sop)
                if potential >= self.potential_threshold or (sop.job, sop.op) in critical_set:
                    filtered_set.add((sop.job, sop.op))
            candidate_set = filtered_set

        moves: list[ImprovementMove] = []

        for sop in self._schedule:
            if (sop.job, sop.op) not in candidate_set:
                continue

            key = (sop.job, sop.op)
            if key not in self._assignment_index:
                continue
            op_idx = self._assignment_index[key]

            # Coupled reassign: move to each alternative machine with best insertion
            operation = self.instance.jobs[sop.job].operations[sop.op]
            for option in operation.options:
                if option.machine != sop.machine:
                    moves.append(ImprovementMove(
                        move_type="coupled_reassign",
                        op_index=op_idx,
                        target_machine=option.machine,
                    ))

            # Swap adjacent on same machine
            m_ops = self._by_machine.get(sop.machine, [])
            for i, mo in enumerate(m_ops):
                if mo.job == sop.job and mo.op == sop.op:
                    if i > 0:
                        prev = m_ops[i - 1]
                        if not self._would_violate_precedence(sop, prev):
                            moves.append(ImprovementMove(
                                move_type="swap_prev",
                                op_index=op_idx,
                            ))
                    if i < len(m_ops) - 1:
                        nxt = m_ops[i + 1]
                        if not self._would_violate_precedence(sop, nxt):
                            moves.append(ImprovementMove(
                                move_type="swap_next",
                                op_index=op_idx,
                            ))
                    break

        return moves

    def _would_violate_precedence(
        self, op_a: ScheduledOperation, op_b: ScheduledOperation
    ) -> bool:
        """Check if swapping op_a and op_b would violate job precedence."""
        if op_a.job == op_b.job:
            return abs(op_a.op - op_b.op) != 1
        return False

    def _apply_move(self, move: ImprovementMove) -> list[tuple[int, int, int]]:
        """Apply an improvement move and return new assignments.

        For ``coupled_reassign``: changes the machine assignment AND tries
        every insertion position on the target machine, returning the
        assignment that yields the lowest makespan. This is the true
        "coupled" neighborhood: reassignment + insertion optimization.

        Note: ``move.op_index`` is an index into ``self._assignments``.
        """
        new_assignments = list(self._assignments)

        if move.move_type == "coupled_reassign":
            job, op_idx, _ = new_assignments[move.op_index]

            # Try every insertion position on the target machine and pick
            # the one with the lowest makespan. The operation is moved to
            # the target machine; we permute the order of operations on that
            # machine while keeping other machines' orders fixed.
            target_machine = move.target_machine
            best_assignments = new_assignments
            best_ms = self._makespan

            # Gather operations currently on the target machine
            target_ops = [a for a in new_assignments if a[2] == target_machine]
            other_ops = [a for a in new_assignments if a[2] != target_machine]

            # Try inserting the moving op at every position in target_ops
            for pos in range(len(target_ops) + 1):
                reordered = list(target_ops)
                # Remove the moving op if it is already in target_ops (it
                # should not be, since target_machine != current machine)
                reordered = [a for a in reordered if not (a[0] == job and a[1] == op_idx)]
                reordered.insert(pos, (job, op_idx, target_machine))
                candidate = other_ops + reordered
                try:
                    cand_schedule = _recompute(self.instance, candidate)
                    cand_valid = validate_schedule(self.instance, cand_schedule)
                    if cand_valid.is_valid and cand_valid.makespan < best_ms:
                        best_ms = cand_valid.makespan
                        best_assignments = candidate
                except (ValueError, Exception):
                    continue

            return best_assignments

        elif move.move_type == "swap_prev":
            job, op_idx, machine = new_assignments[move.op_index]
            m_ops = self._by_machine.get(machine, [])
            for i, mo in enumerate(m_ops):
                if mo.job == job and mo.op == op_idx:
                    if i > 0:
                        prev = m_ops[i - 1]
                        prev_key = (prev.job, prev.op)
                        if prev_key in self._assignment_index:
                            prev_idx = self._assignment_index[prev_key]
                            new_assignments[move.op_index], new_assignments[prev_idx] = \
                                new_assignments[prev_idx], new_assignments[move.op_index]
                    break

        elif move.move_type == "swap_next":
            job, op_idx, machine = new_assignments[move.op_index]
            m_ops = self._by_machine.get(machine, [])
            for i, mo in enumerate(m_ops):
                if mo.job == job and mo.op == op_idx:
                    if i < len(m_ops) - 1:
                        nxt = m_ops[i + 1]
                        nxt_key = (nxt.job, nxt.op)
                        if nxt_key in self._assignment_index:
                            nxt_idx = self._assignment_index[nxt_key]
                            new_assignments[move.op_index], new_assignments[nxt_idx] = \
                                new_assignments[nxt_idx], new_assignments[move.op_index]
                    break

        return new_assignments

    def sample_valid_action(self, rng: Random | None = None) -> int:
        """Sample a random valid action index."""
        if not self._valid_moves:
            raise ValueError("No valid actions available.")
        random_source = rng if rng is not None else Random()
        return random_source.randint(0, len(self._valid_moves) - 1)

    def validate(self) -> tuple[bool, int]:
        """Validate the current schedule. Returns (is_valid, makespan)."""
        result = validate_schedule(self.instance, self._schedule)
        return result.is_valid, result.makespan
