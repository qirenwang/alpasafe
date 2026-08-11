"""T26E-B pilot model: W_theta -> G_phi over Alpamayo candidate trajectories.

Frozen contract (T26E-B): the only model input is the candidate
``planned_trajectory`` [B,64,2]. W_theta (``CandidateTrajectoryConsequenceModel``)
encodes it to a latent consequence representation z [B,64] and predicts the
candidate-conditioned future ego trajectory as plan + residual (zero-initialized
residual head, the leakage-audited T26C convention). G_phi
(``OfficialSceneScoreHead``) predicts the official AlpaSim scene score from z
ONLY — there is deliberately no raw-plan-to-score bypass. The sigmoid constrains
the prediction to [0,1]; the training target stays identity-transformed.
"""

from __future__ import annotations

import torch
from torch import nn

PLAN_STEPS = 64
PLAN_DIM = 2
LATENT_DIM = 64


class CandidateTrajectoryConsequenceModel(nn.Module):
    """W_theta: candidate trajectory -> (latent consequence, future ego states)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(PLAN_STEPS * PLAN_DIM, LATENT_DIM),
            nn.GELU(),
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.GELU(),
        )
        self.future_residual_head = nn.Linear(LATENT_DIM, PLAN_STEPS * PLAN_DIM)
        # T26C plan-residual convention: start exactly at the plan-as-future baseline.
        nn.init.zeros_(self.future_residual_head.weight)
        nn.init.zeros_(self.future_residual_head.bias)

    def forward(
        self, plan_std: torch.Tensor, plan_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """plan_std/plan_raw: [B,64,2] -> (latent [B,64], future ego [B,64,2])."""
        z = self.encoder(plan_std.reshape(plan_std.shape[0], -1))
        residual = self.future_residual_head(z).reshape(-1, PLAN_STEPS, PLAN_DIM)
        return z, plan_raw + residual


class OfficialSceneScoreHead(nn.Module):
    """G_phi: latent consequence z [B,64] -> predicted official score [B] in [0,1]."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(LATENT_DIM, 1)

    def forward(self, latent_consequence: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(latent_consequence)).squeeze(-1)


class SafeWorldCandidatePilot(nn.Module):
    """Composed pilot: G_phi consumes W_theta's latent output (no bypass)."""

    def __init__(self) -> None:
        super().__init__()
        self.w_theta = CandidateTrajectoryConsequenceModel()
        self.g_phi = OfficialSceneScoreHead()

    def forward(self, plan_std: torch.Tensor, plan_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        latent, future = self.w_theta(plan_std, plan_raw)
        score = self.g_phi(latent)
        return {
            "latent_consequence": latent,
            "predicted_future_ego_states_ego_t0": future,
            "predicted_official_alpasim_scene_score": score,
        }
