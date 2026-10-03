"""FJSP-L2S Agent: PPO training for learning to improve FJSP solutions.

This agent implements the core innovation of the paper:
1. **Learning-to-Improve paradigm** for FJSP (first in literature)
2. **Cross-instance generalization** (train on small, test on large)
3. **Critical-path-aware action space** (reassign + swap moves)

Training flow
-------------
1. Sample an instance from the training distribution.
2. Generate an initial solution (e.g., greedy dispatch rule).
3. For each improvement step:
   a. Build the solution graph.
   b. Encode with FJSPImproveNet.
   c. Score candidate moves, sample from policy.
   d. Apply move, observe reward (-delta_makespan).
4. Collect trajectory, update with PPO.
5. Periodically evaluate on held-out instances.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from fjsp.env.improvement_env import FJSPImprovementEnv
from fjsp.parser.fjs_parser import FJSPInstance, parse_fjs
from fjsp.scheduler.validator import ScheduledOperation, earliest_finish_schedule, schedule_to_dict
from fjsp.scheduler.local_search import _assignments
from rl.models.fjsp_l2s import FJSPImproveNet


def _assignments_from_graph(graph) -> list[tuple[int, int, int]]:
    """Reconstruct (job, op, machine) assignments from a SolutionGraph.

    The operation_refs list gives (job, op) for each op node, and the
    assigned_machine feature is stored in op_features column 2.
    """
    machine_col = 2
    assignments = []
    n_machines = graph.machine_features.size(0)
    for i, (job, op) in enumerate(graph.operation_refs):
        machine_norm = graph.op_features[i, machine_col].item()
        # feature is normalized as machine / (num_machines - 1)
        denom = max(n_machines - 1, 1)
        machine = int(round(machine_norm * denom))
        machine = max(0, min(machine, n_machines - 1))
        assignments.append((job, op, machine))
    return assignments


class TrajectoryStep:
    """One step in a training trajectory."""

    __slots__ = ("graph", "moves", "assignments", "action", "log_prob",
                 "value", "reward", "done", "entropy")

    def __init__(
        self,
        graph,
        moves,
        assignments,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        entropy: float,
    ):
        self.graph = graph
        self.moves = moves
        self.assignments = assignments
        self.action = action
        self.log_prob = log_prob
        self.value = value
        self.reward = reward
        self.done = done
        self.entropy = entropy


class FJSPImproveAgent:
    """PPO agent for FJSP learning-to-improve.

    Parameters
    ----------
    net : FJSPImproveNet
        The actor-critic network.
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda for advantage estimation.
    clip_ratio : float
        PPO clipping parameter.
    entropy_coef : float
        Entropy bonus coefficient.
    value_coef : float
        Value loss coefficient.
    max_grad_norm : float
        Gradient clipping norm.
    """

    def __init__(
        self,
        net: FJSPImproveNet,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.net = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    def bc_pretrain(
        self,
        trajectories: list[dict],
        epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-3,
    ) -> list[float]:
        """Behavioral cloning pretraining from expert trajectories.

        Parameters
        ----------
        trajectories : list[dict]
            Each dict has key "trajectory" with a list of observation dicts,
            where each observation contains "graph", "valid_moves",
            "makespan", and "action".
        epochs : int
        batch_size : int
        lr : float

        Returns
        -------
        losses : list[float]
            Average cross-entropy loss per epoch.
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        losses: list[float] = []

        steps = []
        for traj in trajectories:
            steps.extend(traj["trajectory"])

        if not steps:
            return losses

        for epoch in range(epochs):
            # Shuffle steps
            import random
            random.shuffle(steps)
            epoch_losses: list[float] = []

            for i in range(0, len(steps), batch_size):
                batch = steps[i:i + batch_size]
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=next(self.net.parameters()).device)
                valid_count = 0

                for obs in batch:
                    graph = obs["graph"]
                    moves = obs["valid_moves"]
                    action = obs["action"]
                    if not moves or action < 0 or action >= len(moves):
                        continue

                    op_h, machine_h, _ = self.net.encode(graph)
                    assignments = _assignments_from_graph(graph)
                    scores = self.net.score_moves(op_h, machine_h, graph, moves, assignments)
                    if scores.numel() == 0:
                        continue

                    scores = torch.where(torch.isfinite(scores), scores, torch.tensor(-1e8, device=scores.device))
                    log_probs = torch.log_softmax(scores, dim=0)
                    batch_loss = batch_loss - log_probs[action]
                    valid_count += 1

                if valid_count == 0:
                    continue

                batch_loss = batch_loss / valid_count
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(batch_loss.item())

            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                losses.append(avg_loss)
                if (epoch + 1) % 5 == 0:
                    print(f"  BC epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")

        return losses

    def select_action(
        self,
        env: FJSPImprovementEnv,
        deterministic: bool = False,
    ) -> tuple[int, float, float, float]:
        """Select an action using the current policy.

        Returns
        -------
        action_idx, log_prob, value, entropy
        """
        obs = env.observe()
        graph = obs["graph"]
        moves = obs["valid_moves"]

        if not moves:
            return -1, 0.0, 0.0, 0.0

        op_h, machine_h, _ = self.net.encode(graph)
        assignments = _assignments(env.schedule)
        scores = self.net.score_moves(op_h, machine_h, graph, moves, assignments)

        if scores.numel() == 0:
            return -1, 0.0, 0.0, 0.0

        # Mask invalid scores
        scores = torch.where(torch.isfinite(scores), scores, torch.tensor(-1e8, device=scores.device))

        probs = torch.softmax(scores, dim=0)
        value = self.net.value(graph, op_h, machine_h).item()

        if deterministic:
            action_idx = probs.argmax().item()
            log_prob = torch.log(probs[action_idx] + 1e-10).item()
            entropy = 0.0
        else:
            dist = Categorical(probs)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx, device=scores.device)).item()
            entropy = dist.entropy().item()

        return action_idx, log_prob, value, entropy

    def collect_trajectory(
        self,
        env: FJSPImprovementEnv,
    ) -> list[TrajectoryStep]:
        """Collect one episode trajectory.

        Note: the caller is responsible for calling ``env.reset()`` with the
        desired initial schedule before invoking this method. This method
        does NOT reset the environment, so that diverse initial solutions
        set by the training loop are preserved.
        """
        trajectory: list[TrajectoryStep] = []

        while not env.done:
            action_idx, log_prob, value, entropy = self.select_action(env)
            if action_idx < 0:
                break

            obs = env.observe()
            moves = list(obs["valid_moves"])
            assignments = _assignments(env.schedule)
            graph = obs["graph"]

            next_obs, reward, done, info = env.step(action_idx)

            trajectory.append(TrajectoryStep(
                graph=graph,
                moves=moves,
                assignments=assignments,
                action=action_idx,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=done,
                entropy=entropy,
            ))

        return trajectory

    def compute_gae(
        self,
        trajectory: list[TrajectoryStep],
    ) -> tuple[list[float], list[float]]:
        """Compute GAE advantages and returns."""
        if not trajectory:
            return [], []

        values = [step.value for step in trajectory]
        rewards = [step.reward for step in trajectory]
        dones = [step.done for step in trajectory]

        advantages = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0.0
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)
            next_value = values[t]

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def ppo_update(
        self,
        trajectory: list[TrajectoryStep],
        k_epochs: int = 4,
        minibatch_size: int = 32,
    ) -> dict[str, float]:
        """Update the network using PPO."""
        if not trajectory:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        advantages, returns = self.compute_gae(trajectory)
        if not advantages:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Normalize advantages
        adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        if adv_tensor.numel() > 1:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        old_log_probs = torch.tensor([step.log_prob for step in trajectory], dtype=torch.float32)
        return_t = torch.tensor(returns, dtype=torch.float32)
        metrics: dict[str, list[float]] = defaultdict(list)

        for _ in range(k_epochs):
            # Re-evaluate all steps, keeping tensors in the computation graph
            new_log_prob_list = []
            new_value_list = []
            entropy_list = []

            for step in trajectory:
                op_h, machine_h, _ = self.net.encode(step.graph)
                scores = self.net.score_moves(op_h, machine_h, step.graph, step.moves, step.assignments)

                if scores.numel() == 0:
                    new_log_prob_list.append(torch.tensor(0.0, requires_grad=True))
                    new_value_list.append(torch.tensor(0.0, requires_grad=True))
                    entropy_list.append(torch.tensor(0.0, requires_grad=True))
                    continue

                scores = torch.where(torch.isfinite(scores), scores, torch.tensor(-1e8, device=scores.device))
                probs = torch.softmax(scores, dim=0)
                dist = Categorical(probs)

                action_t = torch.tensor(step.action, device=scores.device).clamp(0, probs.size(0) - 1)
                new_log_prob_list.append(dist.log_prob(action_t))
                new_value_list.append(self.net.value(step.graph, op_h, machine_h))
                entropy_list.append(dist.entropy())

            # Stack into tensors (these are in the computation graph)
            new_log_prob_t = torch.stack(new_log_prob_list)
            new_value_t = torch.cat(new_value_list)
            entropy_t = torch.stack(entropy_list)

            # Policy loss (PPO clip)
            ratio = torch.exp(new_log_prob_t - old_log_probs)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = ((new_value_t - return_t) ** 2).mean()

            # Entropy bonus
            entropy_bonus = entropy_t.mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optimizer.step()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy_bonus.item())

        return {k: sum(v) / len(v) for k, v in metrics.items()}


def train_fjsp_l2s(
    train_instances: list[str | Path | FJSPInstance],
    *,
    test_instances: list[str | Path | FJSPInstance] | None = None,
    episodes: int = 500,
    max_steps: int = 50,
    patience: int = 15,
    hidden_dim: int = 128,
    num_mp_rounds: int = 3,
    num_transformer_blocks: int = 2,
    num_heads: int = 8,
    ffn_dim: int = 256,
    use_dual_perspective: bool = True,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.02,
    k_epochs: int = 4,
    seed: int = 0,
    eval_every: int = 10,
    save_dir: str = "data/results",
    init_model: str | Path | None = None,
) -> tuple[FJSPImproveNet, list[dict]]:
    """Train FJSP-L2S agent on multiple instances.

    This is the main training function. It supports:
    - Multi-instance training (cross-instance generalization)
    - PPO with GAE
    - Periodic evaluation on test instances
    - Best model checkpointing
    - Diverse initial solution generation (ef, spt, neh, random, perturb_ef)
    - Dual-perspective encoder (BARI) or single-perspective (ablation)

    Parameters
    ----------
    train_instances : list
        Paths to FJSP instance files or FJSPInstance objects for training.
    test_instances : list, optional
        Paths or instances for evaluation. If None, uses train_instances.
    episodes : int
        Number of training episodes.
    max_steps : int
        Max improvement steps per episode.
    patience : int
        Early stopping patience (no improvement steps).
    hidden_dim, num_mp_rounds, etc.
        Network hyperparameters.
    use_dual_perspective : bool
        If True, use DualPerspectiveEncoder (BARI main model).
        If False, use single-perspective encoder (for ablation).
    seed : int
        Random seed.
    eval_every : int
        Evaluate every N episodes.
    save_dir : str
        Directory to save results.

    Returns
    -------
    net : FJSPImproveNet
        Trained network.
    history : list of dict
        Training history.
    """
    import random
    import numpy as np

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parse instances
    parsed_train = []
    for inst in train_instances:
        if isinstance(inst, FJSPInstance):
            parsed_train.append(inst)
        else:
            parsed_train.append(parse_fjs(inst))

    parsed_test = []
    if test_instances:
        for inst in test_instances:
            if isinstance(inst, FJSPInstance):
                parsed_test.append(inst)
            else:
                parsed_test.append(parse_fjs(inst))
    else:
        parsed_test = parsed_train

    # Create network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FJSPImproveNet(
        hidden_dim=hidden_dim,
        num_mp_rounds=num_mp_rounds,
        num_transformer_blocks=num_transformer_blocks,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        use_dual_perspective=use_dual_perspective,
    ).to(device)

    if init_model is not None:
        state_dict = torch.load(init_model, map_location=device)
        net.load_state_dict(state_dict)
        print(f"Loaded init model from {init_model}")

    agent = FJSPImproveAgent(
        net,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
    )

    # Create environments (no fixed initial schedule)
    envs = []
    for inst in parsed_train:
        envs.append(FJSPImprovementEnv(inst, max_steps=max_steps, patience=patience))

    # Pre-generate diverse initial solutions for each training instance
    from fjsp.scheduler.local_search import random_schedule, neh_construct
    from fjsp.scheduler.dispatch_rules import choose_spt, choose_earliest_finish
    from fjsp.env.dispatch_env import FJSPDispatchEnv

    def _generate_diverse_initial(inst: FJSPInstance, rng: random.Random) -> list[ScheduledOperation]:
        """Generate a diverse initial solution using different strategies."""
        strategy = rng.choice(["ef", "spt", "neh", "random", "perturb_ef"])

        if strategy == "ef":
            return earliest_finish_schedule(inst)
        elif strategy == "spt":
            env_d = FJSPDispatchEnv(inst)
            env_d.reset()
            while not env_d.done:
                env_d.step(choose_spt(env_d))
            return env_d.schedule
        elif strategy == "neh":
            return neh_construct(inst)
        elif strategy == "random":
            return random_schedule(inst, seed=rng.randint(0, 999999))
        else:  # perturb_ef
            base = earliest_finish_schedule(inst)
            # Randomly reassign some operations
            assignments = [(o.job, o.op, o.machine) for o in base]
            for i in range(len(assignments)):
                if rng.random() < 0.15:  # 15% chance to reassign
                    job_idx, op_idx, _ = assignments[i]
                    op = inst.jobs[job_idx].operations[op_idx]
                    new_machine = rng.choice([opt.machine for opt in op.options])
                    assignments[i] = (job_idx, op_idx, new_machine)
            from fjsp.scheduler.local_search import _recompute
            return _recompute(inst, assignments)

    # Training loop
    history: list[dict] = []
    best_eval_makespan = float("inf")
    best_schedule: list[ScheduledOperation] | None = None
    best_instance_idx = 0

    rng = random.Random(seed)

    for ep in range(episodes):
        # Sample an instance
        env_idx = rng.randint(0, len(envs) - 1)
        env = envs[env_idx]

        # Generate diverse initial schedule
        initial = _generate_diverse_initial(env.instance, rng)
        env.reset(initial_schedule=initial)

        # Collect trajectory
        trajectory = agent.collect_trajectory(env)

        # PPO update
        metrics = agent.ppo_update(trajectory, k_epochs=k_epochs)

        # Log
        ep_makespan = env.makespan
        ep_best = env.best_makespan
        ep_steps = env.step_count
        initial_makespan = max(o.end for o in initial)
        ep_improvement = initial_makespan - ep_makespan if trajectory else 0

        record = {
            "episode": ep,
            "instance_idx": env_idx,
            "final_makespan": ep_makespan,
            "best_makespan": ep_best,
            "improvement": ep_improvement,
            "steps": ep_steps,
            "trajectory_len": len(trajectory),
            **metrics,
        }
        history.append(record)

        if (ep + 1) % 10 == 0:
            avg_ms = sum(h["best_makespan"] for h in history[-10:]) / min(len(history), 10)
            print(f"Ep {ep+1}/{episodes} | best_ms={ep_best} | steps={ep_steps} | "
                  f"avg_ms(10)={avg_ms:.1f} | p_loss={metrics.get('policy_loss', 0):.4f}")

        # Periodic evaluation
        if (ep + 1) % eval_every == 0 and parsed_test:
            eval_results = evaluate_agent(net, parsed_test, device=device, max_steps=max_steps)
            avg_eval = sum(r["best_makespan"] for r in eval_results) / len(eval_results)

            if avg_eval < best_eval_makespan:
                best_eval_makespan = avg_eval
                # Save best model
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), save_path / "fjsp_l2s_best.pt")

                # Save best schedule
                best_result = min(eval_results, key=lambda r: r["best_makespan"])
                if best_result.get("schedule"):
                    with open(save_path / "fjsp_l2s_best_schedule.json", "w") as f:
                        json.dump(schedule_to_dict(best_result["schedule"]), f, indent=2)

            print(f"  [Eval] avg_makespan={avg_eval:.1f} | best_avg={best_eval_makespan:.1f}")

    return net, history


@torch.no_grad()
def evaluate_agent(
    net: FJSPImproveNet,
    instances: list[FJSPInstance],
    *,
    device: torch.device | str = "cpu",
    max_steps: int = 50,
    patience: int = 15,
    n_trials: int = 3,
) -> list[dict]:
    """Evaluate the agent on a list of instances.

    Returns
    -------
    results : list of dict
        Per-instance results with initial/best makespan and schedule.
    """
    net.eval()
    results = []

    for inst in instances:
        initial = earliest_finish_schedule(inst)
        initial_makespan = max(o.end for o in initial)

        best_makespan = initial_makespan
        best_schedule = initial

        for trial in range(n_trials):
            # Use EF for first trial, perturbed EF for subsequent trials
            if trial == 0:
                trial_initial = initial
            else:
                # Perturb EF by randomly reassigning some operations
                import random as _rng
                _r = _rng.Random(trial * 42)
                assignments = [(o.job, o.op, o.machine) for o in initial]
                for i in range(len(assignments)):
                    if _r.random() < 0.1 * trial:  # more perturbation for later trials
                        job_idx, op_idx, _ = assignments[i]
                        op = inst.jobs[job_idx].operations[op_idx]
                        new_machine = _r.choice([opt.machine for opt in op.options])
                        assignments[i] = (job_idx, op_idx, new_machine)
                from fjsp.scheduler.local_search import _recompute
                trial_initial = _recompute(inst, assignments)

            env = FJSPImprovementEnv(inst, initial_schedule=trial_initial, max_steps=max_steps, patience=patience)
            env.reset()

            while not env.done:
                action_idx, _, _, _ = FJSPImproveAgent(net).select_action(env, deterministic=True)
                if action_idx < 0:
                    break
                env.step(action_idx)

            if env.best_makespan < best_makespan:
                best_makespan = env.best_makespan
                best_schedule = env.schedule

        results.append({
            "instance": f"jobs={inst.job_count}_machines={inst.machine_count}",
            "initial_makespan": initial_makespan,
            "best_makespan": best_makespan,
            "improvement": initial_makespan - best_makespan,
            "schedule": best_schedule,
        })

    net.train()
    return results
