from __future__ import annotations

import torch
from torch import nn


class TwoStageActorCriticNet(nn.Module):
    """Two-stage masked actor with a state-value critic."""

    def __init__(
        self,
        *,
        job_feature_dim: int,
        machine_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.job_actor = nn.Sequential(
            nn.Linear(job_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.machine_actor = nn.Sequential(
            nn.Linear(machine_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def score_jobs(self, features: torch.Tensor) -> torch.Tensor:
        return self.job_actor(features).squeeze(-1)

    def score_machines(self, features: torch.Tensor) -> torch.Tensor:
        return self.machine_actor(features).squeeze(-1)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic(features).squeeze(-1)

