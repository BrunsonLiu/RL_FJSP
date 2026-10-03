"""Read-only research artifact audit; reports are written outside data/results."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import pickle
from pathlib import Path
import re
import runpy
from statistics import mean, median, pstdev
import subprocess

import _bootstrap  # noqa: F401
import torch

from fjsp.env import FJSPDispatchEnv
from fjsp.env.improvement_env import FJSPImprovementEnv, ImprovementMove
from fjsp.parser.fjs_parser import FJSPInstance, Job, Operation, OperationOption, parse_fjs
from fjsp.scheduler.validator import ScheduledOperation, schedule_from_dict, validate_schedule
from rl.models.action_scorer import ActionScorer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/history_audit"
RESULTS = ROOT / "data/results"


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def tracked_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True,
    )
    return set(result.stdout.decode("utf-8").split("\0"))


def driver_route(path: str) -> str:
    if "/analysis/" in path:
        return "reporting"
    if "/sota/" in path:
        return "classical_pipeline"
    name = Path(path).stem
    if "hgt" in name:
        return "hgt_dispatch"
    if "imitation" in name or name in {"bc_ac_repro", "bc_ac_quick"}:
        return "dispatch_imitation"
    if "neural" in name or "move_rank" in name or "neighborhood_data" in name:
        return "neural_search"
    if "bari" in name or "fjsp_l2s" in name:
        return "bari_improvement"
    if "graph" in name:
        return "graph_dispatch"
    if "l2i" in name:
        return "operator_selection"
    if "reinforce" in name or "a2c" in name or "actor_critic" in name:
        return "flat_dispatch"
    if "bc" in name or "trajectories" in name:
        return "expert_data"
    return "other_tool_or_probe"


def driver_inventory(files: list[dict]) -> list[dict]:
    rows = []
    for item in files:
        path = item["path"]
        if not path.endswith(".py") or not path.startswith(("experiments/", "scripts/dev/", "rl/train", "rl/evaluate")):
            continue
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
        rows.append({
            "path": path, "route_hint": driver_route(path),
            "source_summary": (ast.get_docstring(tree) or "").split("\n")[0],
            "entry_points": "; ".join(item.get("entry_points", [])),
            "artifact_references": "; ".join(item.get("artifact_strings", [])),
            "in_published_snapshot": item["in_published_snapshot"],
        })
    return rows


def call_signature_findings(files: list[dict]) -> list[dict]:
    from rl.agents.fjsp_l2s import evaluate_agent, train_fjsp_l2s
    from fjsp.scheduler.local_search import iterated_local_search

    signatures = {function.__name__: set(inspect.signature(function).parameters) for function in (evaluate_agent, train_fjsp_l2s, iterated_local_search)}
    findings = []
    for item in files:
        path = item["path"]
        if not path.endswith(".py"):
            continue
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module in {"rl.agents.fjsp_l2s", "fjsp.scheduler.local_search"} for alias in node.names}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in imported and node.func.id in signatures:
                unsupported = [keyword.arg for keyword in node.keywords if keyword.arg and keyword.arg not in signatures[node.func.id]]
                if unsupported:
                    findings.append({"path": path, "line": node.lineno, "function": node.func.id, "unsupported_keywords": unsupported})
    return findings


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def file_inventory() -> list[dict]:
    tracked = tracked_paths()
    publication = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "codex/engineering-foundation"],
        cwd=ROOT, capture_output=True,
    )
    published = set(publication.stdout.decode("utf-8").splitlines())
    files = []
    for base in ("fjsp", "rl", "experiments", "scripts", "tests", "docs", "paper", "data/results"):
        for path in sorted((ROOT / base).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            name = relative(path)
            if name.startswith("docs/history_audit/") or path.name == "audit_research_history.py":
                continue
            row = {"path": name, "bytes": path.stat().st_size, "tracked_in_local_index": name in tracked, "in_published_snapshot": name in published}
            if path.suffix in {".py", ".md", ".json", ".csv", ".log", ".tex"}:
                content = path.read_bytes()
                row["sha256"] = hashlib.sha256(content).hexdigest()
                if path.suffix == ".py":
                    try:
                        tree = ast.parse(content.decode("utf-8-sig"))
                        row["entry_points"] = [
                            node.name for node in tree.body
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                        ]
                        row["artifact_strings"] = sorted({
                            node.value for node in ast.walk(tree)
                            if isinstance(node, ast.Constant) and isinstance(node.value, str)
                            and len(node.value) < 180
                            and ("data/results" in node.value or node.value.endswith((".json", ".pt", ".pkl")))
                        })
                    except (SyntaxError, UnicodeError) as exc:
                        row["parse_error"] = str(exc)
                elif path.suffix == ".json":
                    try:
                        value = json.loads(content)
                        if isinstance(value, dict):
                            row["json_keys"] = list(value)
                            records = value.get("operations", value.get("results", []))
                            row["rows"] = len(records) if isinstance(records, list) else None
                        elif isinstance(value, list):
                            row["rows"] = len(value)
                            row["json_keys"] = list(value[0]) if value and isinstance(value[0], dict) else []
                    except (ValueError, UnicodeError) as exc:
                        row["parse_error"] = str(exc)
            files.append(row)
    return files


def schedule_inventory() -> list[dict]:
    instances = {path.stem: parse_fjs(path) for path in sorted((ROOT / "data/instances/brandimarte").glob("mk*.txt"))}
    instances["tiny_2x2"] = parse_fjs(ROOT / "data/instances/tiny_2x2.fjs")
    rows = []
    for path in sorted(RESULTS.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("operations"), list):
            continue
        row = {"path": relative(path)}
        try:
            schedule = schedule_from_dict(payload)
            match = re.search(r"mk\d{2}", path.name)
            declared = match.group() if match else ("tiny_2x2" if "tiny" in path.name or "engineering_smoke" in path.name or "engineering_eval" in path.name else None)
            row.update({"operations": len(schedule), "makespan": max((op.end for op in schedule), default=0), "filename_instance": declared})
            valid_matches = []
            for name, instance in instances.items():
                if len(schedule) == instance.operation_count and validate_schedule(instance, schedule).is_valid:
                    valid_matches.append(name)
            row["valid_instance_matches"] = valid_matches
            if declared:
                result = validate_schedule(instances[declared], schedule)
                row["valid_for_filename_instance"] = result.is_valid
                row["errors"] = list(result.errors[:3])
            else:
                row["valid_for_filename_instance"] = None
        except (ValueError, TypeError, KeyError) as exc:
            row["parse_error"] = str(exc)
        rows.append(row)
    return rows


def seed_inventory() -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob("reinforce_*_5seed.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            values = [run["best"] for run in item["seed_results"]]
            trimmed = sorted(values)[:-1]
            rows.append({
                "instance": item["instance"], "path": relative(path),
                "seeds": [run["seed"] for run in item["seed_results"]],
                "values": values, "best": min(values), "mean": mean(values),
                "median": median(values), "population_std": pstdev(values),
                "sample_std": __import__("statistics").stdev(values),
                "mean_without_worst": mean(trimmed),
                "stored_mean": item["reinforce_mean"], "stored_std": item["reinforce_std"],
            })
    return rows


def hgt_probe() -> list[dict]:
    from rl.agents.hgt_fjsp import HGTActorCriticAgent

    results = []
    for name in ("hgt_bc_mk01_ss.pt", "hgt_a2c_mk01_bc.pt", "hgt_ppo_mk01_bc_v5.pt"):
        path = RESULTS / name
        row = {"checkpoint": relative(path)}
        if not path.exists():
            row["status"] = "missing_local_checkpoint"
            results.append(row)
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            config = dict(payload.get("config", {}))
            state = payload["model_state"]
            for key, value in state.items():
                if key.endswith("encoder.op_blocks.0.ffn.0.weight"):
                    config.setdefault("ffn_dim", int(value.shape[0]))
            allowed = set(inspect.signature(HGTActorCriticAgent.create).parameters)
            config = {key: value for key, value in config.items() if key in allowed}
            agent = HGTActorCriticAgent.create(**config)
            agent.net.load_state_dict(state)
            env = FJSPDispatchEnv.from_file(ROOT / "data/instances/brandimarte/mk01.txt")
            with torch.no_grad():
                rollouts = [agent.rollout(env, greedy=True) for _ in range(5)]
            row.update({"status": "evaluated", "makespans": [item.makespan for item in rollouts], "valid": all(item.is_valid for item in rollouts)})
        except (RuntimeError, ValueError, TypeError, KeyError, OSError, pickle.UnpicklingError) as exc:
            row.update({"status": "incompatible_or_unreadable", "error": str(exc)[:500]})
        results.append(row)
    return results


def mechanism_probe() -> dict:
    instance = FJSPInstance((Job((Operation((OperationOption(0, 10), OperationOption(1, 1))),)),), 2)
    initial = [ScheduledOperation(0, 0, 0, 0, 10)]
    env = FJSPImprovementEnv(instance, initial_schedule=initial)
    move = ImprovementMove("coupled_reassign", 0, 1)
    returned = env._apply_move(move)
    alternative = [ScheduledOperation(0, 0, 1, 0, 1)]
    return {
        "initial_valid": validate_schedule(instance, initial).is_valid,
        "initial_makespan": 10, "returned_assignments": returned,
        "unchanged": returned == env._assignments,
        "potential_estimate": env._compute_improvement_potential(initial[0]),
        "alternative_valid": validate_schedule(instance, alternative).is_valid,
        "alternative_makespan": 1, "achievable_reduction": 9,
    }


def protocol_probe() -> dict:
    from fjsp.scheduler.local_search import simulated_annealing
    from fjsp.scheduler.validator import greedy_schedule

    env = FJSPDispatchEnv.from_file(ROOT / "data/instances/tiny_2x2.fjs")
    namespace = runpy.run_path(str(ROOT / "experiments/rl_baselines/l2i_baseline.py"), run_name="audit_l2i_module")
    try:
        namespace["train_l2i"](env, n_steps=1, seed=0)
    except Exception as exc:
        l2i = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    else:
        l2i = {"status": "one_step_completed"}
    initial = greedy_schedule(env.instance)
    initial_ms = validate_schedule(env.instance, initial).makespan
    sa = []
    for seed in range(3):
        schedule, value = simulated_annealing(env.instance, initial, max_total_iterations=20, seed=seed)
        sa.append({"seed": seed, "initial": initial_ms, "returned": value, "legal": validate_schedule(env.instance, schedule).is_valid})
    return {"l2i_one_step": l2i, "sa_tiny": sa}


def metrics() -> dict:
    final = read_json("data/results/sota_final.json")
    means = {key: mean(row[key] for row in final) for key in ("ef_makespan", "reinforce_best", "ils_best", "sa_makespan", "final_best", "lit_target")}
    gaps = [100 * (row["final_best"] - row["lit_target"]) / row["lit_target"] for row in final]
    ablations = {}
    for name in ("bari_full", "single_perspective", "binary_cp"):
        path = RESULTS / f"ablation_{name}/ablation_results.json"
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            ablations[name] = {"values": [row["best_makespan"] for row in rows], "mean": mean(row["best_makespan"] for row in rows), "improved": sum(row["best_makespan"] < row["initial_makespan"] for row in rows)}
    neural = {}
    for name, field, comparator in (
        ("neural_ils_benchmark.json", "neural_ils_makespan", "standard_ils_makespan"),
        ("neural_ils_v2_benchmark.json", "neural_ils_v2_makespan", "sota_makespan"),
        ("neural_ils_v2_benchmark_fast.json", "neural_ils_v2_makespan", "sota_makespan"),
    ):
        path = RESULTS / name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["results"]
            pairs = [row for row in rows if row.get(field) is not None and row.get(comparator) is not None]
            neural[name] = {"rows": len(rows), "wins": sum(row[field] < row[comparator] for row in pairs), "ties": sum(row[field] == row[comparator] for row in pairs), "losses": sum(row[field] > row[comparator] for row in pairs), "args": payload.get("args", {}), "values": [{"instance": row["instance"], "value": row[field], "reference": row[comparator]} for row in pairs]}
    cross = {}
    ef_lookup = {row["instance"]: row["ef_makespan"] for row in final}
    for folder, name in (("bari_cross", "cross_instance_results.json"), ("cross_instance", "evaluation_results.json"), ("cross_instance_v2", "evaluation_results.json")):
        path = RESULTS / folder / name
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            cross[folder] = {"rows": len(rows), "improved": sum(row.get("improvement", 0) > 0 for row in rows), "mean_improvement": mean(row.get("improvement", 0) for row in rows), "beats_historical_ef": sum(row.get("instance") in ef_lookup and row["best_makespan"] < ef_lookup[row["instance"]] for row in rows)}
    return {
        "final_rows": final, "stage_means": means,
        "mean_gap_percent": mean(gaps), "median_gap_percent": median(gaps),
        "mean_gap_excluding_mk13": mean(gap for row, gap in zip(final, gaps) if row["instance"] != "mk13"),
        "sum_gap_units": sum(row["final_best"] - row["lit_target"] for row in final),
        "stage_relative_percent": {
            "RL_vs_EF": 100 * (means["ef_makespan"] - means["reinforce_best"]) / means["ef_makespan"],
            "ILS_vs_RL": 100 * (means["reinforce_best"] - means["ils_best"]) / means["reinforce_best"],
            "SA_vs_ILS": 100 * (means["sa_makespan"] - means["ils_best"]) / means["ils_best"],
        },
        "sa_regressions": [row["instance"] for row in final if row["sa_makespan"] > row["ils_best"]],
        "sa_improvements": [row["instance"] for row in final if row["sa_makespan"] < row["ils_best"]],
        "tabu": read_json("data/results/tabu_results.json"),
        "ablations": ablations, "neural_comparisons": neural, "cross_runs": cross,
        "action_scorer_parameters": {str(size): sum(parameter.numel() for parameter in ActionScorer(8, size).parameters()) for size in (32, 64)},
    }


def claims(data: dict) -> list[dict]:
    schedules = {row["path"]: row for row in data["schedules"]}
    m = data["metrics"]
    l2i = read_json("data/results/l2i_baseline.json")
    lookup = {row["instance"]: row for row in m["final_rows"]}
    l2i_pairs = [{"instance": row["instance"], "l2i": row["l2i_best"], "pipeline": lookup[row["instance"]]["final_best"]} for row in l2i]
    def schedule(name: str) -> dict:
        return schedules.get("data/results/" + name, {"status": "missing_local_artifact"})
    definitions = [
        ("C01", "5.64% is the quality of the PA-REINFORCE policy itself", "fragile", {"historical_pipeline_gap_percent": m["mean_gap_percent"], "median": m["median_gap_percent"], "without_mk13": m["mean_gap_excluding_mk13"], "policy_only_gap_percent": mean(100 * (row["reinforce_best"] - row["lit_target"]) / row["lit_target"] for row in m["final_rows"])}, "Attribute the arithmetic to the historical composite pipeline and internal reference table, not the standalone policy or a current literature record."),
        ("C02", "The saved MK12 schedule substantiates makespan 508", "refuted", schedule("sota_mk12_schedule.json"), "Saved schedule validates at 524; 508 remains unsupported by this artifact."),
        ("C03", "MK13 416 is a new/current SOTA", "fragile", {"schedule": schedule("sota_mk13_schedule.json"), "internal_reference": lookup["mk13"]["lit_target"]}, "Retain legal makespan 416 and its difference from the stored reference; external novelty/rank is unverified."),
        ("C04", "The implemented policy has one hidden layer and 321 parameters", "refuted", m["action_scorer_parameters"], "Current model has two hidden layers, 1377 parameters at width 32 and 4801 at width 64. Draft features and argmin rule also disagree with source."),
        ("C05", "The archived evidence supports 5 seeds on all 15 instances and 375 comparable runs", "fragile", {"explicit_seed_rows": len(data["seeds"]), "explicit_seed_runs": sum(len(row["values"]) for row in data["seeds"]), "missing_instances": sorted(set(lookup) - {row["instance"] for row in data["seeds"]})}, "These explicit seed files cover 12 instances/60 runs; remaining runs need a unified manifest before asserting complete comparable coverage."),
        ("C06", "SA is generally a negative-contribution method", "fragile", {"regressions": m["sa_regressions"], "improvements": m["sa_improvements"], "stage_means": m["stage_means"], "current_probe": data["protocol_probe"].get("sa_tiny")}, "Observed historical endpoint regressions are descriptive, not a universal statement about SA; distinguish last endpoint from best-so-far and heterogeneous configurations."),
        ("C07", "Tabu search only improves MK15", "refuted", m["tabu"], "The recorded tabu runs improve their own starts on MK10, MK15, and MK09; MK09 matches an already better aggregate result."),
        ("C08", "15/15 wins over L2I establish superiority over the published method", "fragile", {"pairs": l2i_pairs, "wins": sum(row["pipeline"] < row["l2i"] for row in l2i_pairs), "wins_without_largest_difference": sum(row["pipeline"] < row["l2i"] for row in sorted(l2i_pairs, key=lambda row: row["l2i"] - row["pipeline"])[:-1]), "current_probe": data["protocol_probe"].get("l2i_one_step")}, "Restrict to this local operator-selection baseline; current one-step reproducibility, initialization, multi-start search and budgets require repair/reconciliation; published-method fidelity has not been verified."),
        ("C09", "Stage gains are ILS -12.1%, SA +6.1%, and final -7.2%", "refuted", {"means": m["stage_means"], "relative_percent": m["stage_relative_percent"]}, "The cited numbers are mean makespan-unit differences, not percentages; final drop is not attributable exclusively to tabu."),
        ("C10", "BC+AC has a saved legal MK01 schedule with makespan 49", "fragile" if schedule("bc_ac_mk01_best_schedule.json").get("status") == "missing_local_artifact" else ("verified" if schedule("bc_ac_mk01_best_schedule.json").get("makespan") == 49 and schedule("bc_ac_mk01_best_schedule.json").get("valid_for_filename_instance") else "refuted"), schedule("bc_ac_mk01_best_schedule.json"), "Scope to the saved schedule and historical run; new feature versions and multi-seed stability require separate checks."),
        ("C11", "HGT fine-tuning is a proven dead end", "fragile", data["hgt_probe"], "Report checkpoint reevaluations and their compatibility; a few configurations/checkpoints do not establish that the whole method family fails."),
        ("C12", "BARI main table improves EF on 4/5 instances", "refuted" if "bari_full" in m["ablations"] else "fragile", m["ablations"], "The local full-table artifact improves 3/5 instances; missing local artifacts cannot reproduce this count. Keep architectural generalization as a separate, still limited single-seed pilot observation."),
        ("C13", "Current coupled_reassign can realize a profitable machine reassignment", "refuted", data["mechanism_probe"], "The moving operation remains in other_ops and is added again on the target machine; decoding fails and broad exception handling returns the unchanged assignment."),
        ("C14", "The improvement-potential estimate is an upper bound on achievable reduction", "refuted", data["mechanism_probe"], "A legal alternative gains 9 while the estimate is 0; use heuristic-score wording until a valid bound is derived."),
    ]
    return [{"id": cid, "claim": claim, "verdict": verdict, "evidence": evidence, "resolution": resolution} for cid, claim, verdict, evidence, resolution in definitions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", help="Print one claim's reproducible check instead of writing reports.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    claim = args.claim
    data = {
        "files": file_inventory() if claim is None else [],
        "schedules": schedule_inventory() if claim is None or claim in {"C02", "C03", "C10"} else [],
        "seeds": seed_inventory() if claim is None or claim == "C05" else [],
        "metrics": metrics(),
        "mechanism_probe": mechanism_probe() if claim is None or claim in {"C13", "C14"} else {},
        "hgt_probe": hgt_probe() if claim is None or claim == "C11" else [],
        "protocol_probe": protocol_probe() if claim is None or claim in {"C06", "C08"} else {},
    }
    checked = claims(data)
    if args.claim:
        print(json.dumps(next(row for row in checked if row["id"] == args.claim), indent=2, ensure_ascii=True))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    data["drivers"] = driver_inventory(data["files"])
    data["signature_findings"] = call_signature_findings(data["files"])
    data["audit_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    data["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data["publication_revision"] = subprocess.check_output(["git", "rev-parse", "codex/engineering-foundation"], cwd=ROOT, text=True).strip()
    (OUT / "evidence.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "claims.json").write_text(json.dumps(checked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "ATTEMPTS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data["drivers"][0]))
        writer.writeheader()
        writer.writerows(data["drivers"])
    with (OUT / "ledger.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["iter", "claim", "verdict", "threat", "resolution"])
        for index, row in enumerate(checked, 1):
            writer.writerow([index, row["id"] + ": " + row["claim"], row["verdict"], "See attributable counterexample/recomputation in iter directory", row["resolution"]])
            directory = OUT / f"iter{index:02}"
            directory.mkdir(exist_ok=True)
            (directory / "out.json").write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (directory / "check.py").write_text(
                "import subprocess\nimport sys\nfrom pathlib import Path\n"
                "root = Path(__file__).resolve().parents[3]\n"
                f"subprocess.run([sys.executable, str(root / 'scripts/audit_research_history.py'), '--claim', '{row['id']}'], cwd=root, check=True)\n",
                encoding="utf-8",
            )
    print(f"Audited {len(data['files'])} files, {len(data['schedules'])} saved schedules, {len(checked)} primary claims.")
    print("Verdicts: " + json.dumps({verdict: sum(row["verdict"] == verdict for row in checked) for verdict in ("verified", "fragile", "refuted")}))
    print(f"Reports: {OUT}")


if __name__ == "__main__":
    main()
