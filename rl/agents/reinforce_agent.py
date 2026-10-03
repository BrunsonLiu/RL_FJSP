from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.distributions import Categorical

from fjsp.env import DispatchAction, FJSPDispatchEnv
from fjsp.utils.scaling import instance_time_scale, scale_value
from rl.models.action_scorer import ActionScorer


FEATURE_DIM = 8


@dataclass
class EpisodeResult:
    makespan: int
    total_reward: float
    is_valid: bool
    log_prob_sum: torch.Tensor | None = None
    entropy_sum: torch.Tensor | float = 0.0


def action_features(env: FJSPDispatchEnv, action: DispatchAction, *, scale: int | None = None) -> list[float]:
    op_idx = env.job_next_op[action.job]
    job = env.instance.jobs[action.job]
    operation = job.operations[op_idx]
    duration = next(option.duration for option in operation.options if option.machine == action.machine)
    start = max(env.job_ready_time[action.job], env.machine_ready_time[action.machine])
    end = start + duration
    scale = instance_time_scale(env) if scale is None else scale

    return [
        scale_value(env.job_ready_time[action.job], scale),
        scale_value(env.machine_ready_time[action.machine], scale),
        scale_value(duration, scale),
        scale_value(start, scale),
        scale_value(end, scale),
        scale_value(op_idx, max(len(job.operations) - 1, 1)),
        scale_value(len(job.operations) - op_idx - 1, max(len(job.operations), 1)),
        scale_value(env.makespan, scale),
    ]


class ReinforceDispatchAgent:
    def __init__(self, model: ActionScorer, *, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    @classmethod
    def create(cls, *, hidden_dim: int = 64, device: str = "cpu") -> "ReinforceDispatchAgent":
        return cls(ActionScorer(FEATURE_DIM, hidden_dim=hidden_dim), device=device)

    def select_action(
        self,
        env: FJSPDispatchEnv,
        *,
        greedy: bool,
        rng: Random | None = None,
    ) -> tuple[DispatchAction, torch.Tensor | None]:
        actions = env.available_actions()
        if not actions:
            raise ValueError("No valid actions available.")

        scale = instance_time_scale(env)
        features = torch.tensor(
            [action_features(env, action, scale=scale) for action in actions],
            dtype=torch.float32,
            device=self.device,
        )
        logits = self.model(features)

        if greedy:
            index = int(torch.argmax(logits).item())
            return actions[index], None

        distribution = Categorical(logits=logits)
        if rng is not None:
            torch.manual_seed(rng.randrange(0, 2**31 - 1))
        sample = distribution.sample()
        index = int(sample.item())
        return actions[index], distribution.log_prob(sample)

    def rollout(self, env: FJSPDispatchEnv, *, greedy: bool = False, seed: int | None = None) -> EpisodeResult:
        env.reset()
        rng = Random(seed) if seed is not None else None
        log_probs: list[torch.Tensor] = []
        total_reward = 0.0
        entropy_sum: torch.Tensor | float = 0.0

        while not env.done:
            action, log_prob = self.select_action(env, greedy=greedy, rng=rng)
            if log_prob is not None:
                log_probs.append(log_prob)
                # Use the decision state and retain gradients for the entropy bonus.
                actions = env.available_actions()
                scale = instance_time_scale(env)
                features = torch.tensor(
                    [action_features(env, a, scale=scale) for a in actions],
                    dtype=torch.float32, device=self.device,
                )
                logits = self.model(features)
                entropy_sum = entropy_sum + Categorical(logits=logits).entropy()
            _, reward, _, _ = env.step(action)
            total_reward += reward

        validation = env.validate()
        log_prob_sum = torch.stack(log_probs).sum() if log_probs else None
        return EpisodeResult(
            makespan=validation.makespan,
            total_reward=total_reward,
            is_valid=validation.is_valid,
            log_prob_sum=log_prob_sum,
            entropy_sum=entropy_sum,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "feature_dim": FEATURE_DIM,
            "model_state": self.model.state_dict(),
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output)

    @classmethod
    def load(cls, path: str | Path, *, hidden_dim: int = 64, device: str = "cpu") -> "ReinforceDispatchAgent":
        agent = cls.create(hidden_dim=hidden_dim, device=device)
        payload = torch.load(path, map_location=device)
        state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
        agent.model.load_state_dict(state)
        return agent


def train_reinforce(
    env: FJSPDispatchEnv,
    *,
    episodes: int,
    lr: float = 3e-3,
    hidden_dim: int = 64,
    seed: int = 0,
    device: str = "cpu",
    entropy_coef: float = 0.01,
) -> tuple[ReinforceDispatchAgent, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = Random(seed)
    agent = ReinforceDispatchAgent.create(hidden_dim=hidden_dim, device=device)
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)
    baseline: float | None = None
    history: list[dict[str, float]] = []
    best_makespan: int | None = None
    best_episode = 0
    best_state: dict[str, torch.Tensor] | None = None

    for episode in range(1, episodes + 1):
        result = agent.rollout(env, greedy=False, seed=rng.randrange(0, 2**31 - 1))
        if not result.is_valid or result.log_prob_sum is None:
            raise RuntimeError("Training rollout produced an invalid or empty episode.")

        reward = -float(result.makespan)
        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
        advantage = reward - baseline
        # Entropy bonus for exploration (prevents premature convergence)
        loss = -result.log_prob_sum * advantage - entropy_coef * result.entropy_sum

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=1.0)
        optimizer.step()

        if episode == 1 or episode == episodes or episode % max(1, episodes // 10) == 0:
            greedy_result = agent.rollout(env, greedy=True)
            if best_makespan is None or greedy_result.makespan < best_makespan:
                best_makespan = greedy_result.makespan
                best_episode = episode
                best_state = deepcopy(agent.model.state_dict())
            history.append(
                {
                    "episode": float(episode),
                    "sample_makespan": float(result.makespan),
                    "greedy_makespan": float(greedy_result.makespan),
                    "best_greedy_makespan": float(best_makespan),
                    "best_episode": float(best_episode),
                    "baseline": float(baseline),
                    "loss": float(loss.detach().cpu().item()),
                }
            )

    if best_state is not None:
        agent.model.load_state_dict(best_state)
    return agent, history
