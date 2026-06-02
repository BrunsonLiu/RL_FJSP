from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.distributions import Categorical

from fjsp.env import DispatchAction, FJSPDispatchEnv
from fjsp.utils.scaling import instance_time_scale, scale_value
from rl.models.two_stage_actor_critic import TwoStageActorCriticNet


JOB_FEATURE_DIM = 8
MACHINE_FEATURE_DIM = 8
GLOBAL_FEATURE_DIM = 6


@dataclass(frozen=True)
class ActorCriticEpisode:
    makespan: int
    is_valid: bool
    log_probs: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    entropies: tuple[torch.Tensor, ...]


def global_features(env: FJSPDispatchEnv, *, scale: int | None = None) -> list[float]:
    scale = instance_time_scale(env) if scale is None else scale
    unfinished_jobs = sum(1 for job_idx, job in enumerate(env.instance.jobs) if env.job_next_op[job_idx] < len(job.operations))
    return [
        scale_value(env.makespan, scale),
        scale_value(env.remaining_operations, env.instance.operation_count),
        scale_value(unfinished_jobs, env.instance.job_count),
        scale_value(min(env.job_ready_time, default=0), scale),
        scale_value(max(env.job_ready_time, default=0), scale),
        scale_value(max(env.machine_ready_time, default=0), scale),
    ]


def job_features(env: FJSPDispatchEnv, job_idx: int, *, scale: int | None = None) -> list[float]:
    scale = instance_time_scale(env) if scale is None else scale
    job = env.instance.jobs[job_idx]
    op_idx = env.job_next_op[job_idx]
    operation = job.operations[op_idx]
    durations = [option.duration for option in operation.options]
    best_end = min(
        max(env.job_ready_time[job_idx], env.machine_ready_time[option.machine]) + option.duration
        for option in operation.options
    )
    return [
        scale_value(env.job_ready_time[job_idx], scale),
        scale_value(op_idx, max(len(job.operations) - 1, 1)),
        scale_value(len(job.operations) - op_idx, len(job.operations)),
        scale_value(min(durations), scale),
        scale_value(sum(durations) / len(durations), scale),
        scale_value(best_end, scale),
        scale_value(env.makespan, scale),
        scale_value(len(operation.options), env.instance.machine_count),
    ]


def machine_features(env: FJSPDispatchEnv, action: DispatchAction, *, scale: int | None = None) -> list[float]:
    scale = instance_time_scale(env) if scale is None else scale
    op_idx = env.job_next_op[action.job]
    operation = env.instance.jobs[action.job].operations[op_idx]
    duration = next(option.duration for option in operation.options if option.machine == action.machine)
    start = max(env.job_ready_time[action.job], env.machine_ready_time[action.machine])
    end = start + duration
    return [
        scale_value(env.job_ready_time[action.job], scale),
        scale_value(env.machine_ready_time[action.machine], scale),
        scale_value(duration, scale),
        scale_value(start, scale),
        scale_value(end, scale),
        scale_value(start - env.job_ready_time[action.job], scale),
        scale_value(action.machine, max(env.instance.machine_count - 1, 1)),
        scale_value(env.makespan, scale),
    ]


class TwoStageActorCriticAgent:
    def __init__(self, model: TwoStageActorCriticNet, *, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    @classmethod
    def create(cls, *, hidden_dim: int = 64, device: str = "cpu") -> "TwoStageActorCriticAgent":
        return cls(
            TwoStageActorCriticNet(
                job_feature_dim=JOB_FEATURE_DIM,
                machine_feature_dim=MACHINE_FEATURE_DIM,
                global_feature_dim=GLOBAL_FEATURE_DIM,
                hidden_dim=hidden_dim,
            ),
            device=device,
        )

    def select_action(
        self,
        env: FJSPDispatchEnv,
        *,
        greedy: bool,
        rng: Random | None = None,
    ) -> tuple[DispatchAction, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        scale = instance_time_scale(env)
        candidate_jobs = [
            job_idx
            for job_idx, job in enumerate(env.instance.jobs)
            if env.job_next_op[job_idx] < len(job.operations)
        ]
        if not candidate_jobs:
            raise ValueError("No valid jobs available.")

        job_tensor = torch.tensor(
            [job_features(env, job_idx, scale=scale) for job_idx in candidate_jobs],
            dtype=torch.float32,
            device=self.device,
        )
        job_logits = self.model.score_jobs(job_tensor)
        job_dist = Categorical(logits=job_logits)
        if greedy:
            job_pos = int(torch.argmax(job_logits).item())
            job_log_prob = None
            job_entropy = None
        else:
            if rng is not None:
                torch.manual_seed(rng.randrange(0, 2**31 - 1))
            job_sample = job_dist.sample()
            job_pos = int(job_sample.item())
            job_log_prob = job_dist.log_prob(job_sample)
            job_entropy = job_dist.entropy()
        selected_job = candidate_jobs[job_pos]

        machine_actions = [action for action in env.available_actions() if action.job == selected_job]
        machine_tensor = torch.tensor(
            [machine_features(env, action, scale=scale) for action in machine_actions],
            dtype=torch.float32,
            device=self.device,
        )
        machine_logits = self.model.score_machines(machine_tensor)
        machine_dist = Categorical(logits=machine_logits)
        if greedy:
            machine_pos = int(torch.argmax(machine_logits).item())
            log_prob = None
            entropy = None
        else:
            if rng is not None:
                torch.manual_seed(rng.randrange(0, 2**31 - 1))
            machine_sample = machine_dist.sample()
            machine_pos = int(machine_sample.item())
            machine_log_prob = machine_dist.log_prob(machine_sample)
            machine_entropy = machine_dist.entropy()
            log_prob = job_log_prob + machine_log_prob
            entropy = job_entropy + machine_entropy

        value_tensor = torch.tensor(global_features(env, scale=scale), dtype=torch.float32, device=self.device)
        value = self.model.value(value_tensor)
        return machine_actions[machine_pos], log_prob, value, entropy

    def rollout(self, env: FJSPDispatchEnv, *, greedy: bool = False, seed: int | None = None) -> ActorCriticEpisode:
        env.reset()
        rng = Random(seed) if seed is not None else None
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        while not env.done:
            action, log_prob, value, entropy = self.select_action(env, greedy=greedy, rng=rng)
            env.step(action)
            if log_prob is not None and value is not None and entropy is not None:
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)

        validation = env.validate()
        return ActorCriticEpisode(
            makespan=validation.makespan,
            is_valid=validation.is_valid,
            log_probs=tuple(log_probs),
            values=tuple(values),
            entropies=tuple(entropies),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "job_feature_dim": JOB_FEATURE_DIM,
            "machine_feature_dim": MACHINE_FEATURE_DIM,
            "global_feature_dim": GLOBAL_FEATURE_DIM,
            "model_state": self.model.state_dict(),
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output)

    @classmethod
    def load(cls, path: str | Path, *, hidden_dim: int = 64, device: str = "cpu") -> "TwoStageActorCriticAgent":
        agent = cls.create(hidden_dim=hidden_dim, device=device)
        payload = torch.load(path, map_location=device)
        state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
        agent.model.load_state_dict(state)
        return agent


def train_actor_critic(
    env: FJSPDispatchEnv,
    *,
    episodes: int,
    lr: float = 3e-3,
    hidden_dim: int = 64,
    seed: int = 0,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    device: str = "cpu",
) -> tuple[TwoStageActorCriticAgent, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = Random(seed)
    agent = TwoStageActorCriticAgent.create(hidden_dim=hidden_dim, device=device)
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    best_makespan: int | None = None
    best_episode = 0
    best_state: dict[str, torch.Tensor] | None = None

    for episode in range(1, episodes + 1):
        result = agent.rollout(env, greedy=False, seed=rng.randrange(0, 2**31 - 1))
        if not result.is_valid or not result.log_probs:
            raise RuntimeError("Actor-critic rollout produced an invalid or empty episode.")

        return_scale = float(instance_time_scale(env))
        returns = torch.full(
            (len(result.log_probs),),
            -float(result.makespan) / return_scale,
            dtype=torch.float32,
            device=agent.device,
        )
        log_probs = torch.stack(result.log_probs)
        values = torch.stack(result.values)
        entropies = torch.stack(result.entropies)
        advantages = returns - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss

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
                    "actor_loss": float(actor_loss.detach().cpu().item()),
                    "critic_loss": float(critic_loss.detach().cpu().item()),
                    "entropy": float(entropies.mean().detach().cpu().item()),
                }
            )

    if best_state is not None:
        agent.model.load_state_dict(best_state)
    return agent, history
