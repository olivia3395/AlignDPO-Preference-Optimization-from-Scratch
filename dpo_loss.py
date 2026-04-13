"""
dpo_loss.py
-----------
Manual implementation of DPO (and variants) from the paper:
  Rafailov et al. (2023) "Direct Preference Optimization:
  Your Language Model is Secretly a Reward Model"

Also implements:
  - IPO  (Azar et al., 2023)
  - KTO  (Ethayarajh et al., 2023) — unpaired variant

Used in ablation experiments (compare_losses.py).
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Literal


# ── Core log-prob utility ─────────────────────────────────────────────────────

def get_log_probs(
    logits: Tensor,      # (B, T, V)
    labels: Tensor,      # (B, T)  — token ids, -100 for masked positions
) -> Tensor:
    """
    Compute per-token log-probs then sum over non-masked positions.
    Returns shape (B,).
    """
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)   # (B, T-1, V)
    target    = labels[:, 1:].clone()                        # (B, T-1)

    mask = (target != -100).float()
    target = target.clamp(min=0)                             # avoid index -100

    # Gather log-prob of the actual token at each position
    token_log_probs = log_probs.gather(
        dim=-1, index=target.unsqueeze(-1)
    ).squeeze(-1)                                            # (B, T-1)

    return (token_log_probs * mask).sum(dim=-1)              # (B,)


# ── DPO Loss ──────────────────────────────────────────────────────────────────

def dpo_loss(
    policy_chosen_logps:    Tensor,   # log π_θ(y_w | x)   shape (B,)
    policy_rejected_logps:  Tensor,   # log π_θ(y_l | x)   shape (B,)
    reference_chosen_logps: Tensor,   # log π_ref(y_w | x) shape (B,)
    reference_rejected_logps: Tensor, # log π_ref(y_l | x) shape (B,)
    beta: float = 0.1,
    loss_type: Literal["sigmoid", "ipo", "kto_pair"] = "sigmoid",
    label_smoothing: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns
    -------
    loss          : scalar
    chosen_reward : (B,)   — for logging
    rejected_reward: (B,)  — for logging
    """

    # ── Log-ratio: how much more/less likely vs reference ────────────────────
    chosen_ratio   = policy_chosen_logps   - reference_chosen_logps    # (B,)
    rejected_ratio = policy_rejected_logps - reference_rejected_logps  # (B,)

    # ── reward as defined in DPO paper ───────────────────────────────────────
    chosen_reward   = beta * chosen_ratio.detach()
    rejected_reward = beta * rejected_ratio.detach()

    # ── Loss variants ────────────────────────────────────────────────────────
    if loss_type == "sigmoid":
        # Standard DPO (Bradley-Terry preference model)
        # L = -E[ log σ( β(log(π/π_ref)(y_w) - log(π/π_ref)(y_l)) ) ]
        logits = chosen_ratio - rejected_ratio                  # (B,)
        if label_smoothing > 0:
            loss = (
                -F.logsigmoid( beta * logits) * (1 - label_smoothing)
                -F.logsigmoid(-beta * logits) * label_smoothing
            )
        else:
            loss = -F.logsigmoid(beta * logits)

    elif loss_type == "ipo":
        # IPO: identity function instead of log-sigmoid
        # Avoids reward over-optimisation by removing the log
        # L = (log(π/π_ref)(y_w) - log(π/π_ref)(y_l) - 1/(2β))²
        logits = chosen_ratio - rejected_ratio
        loss = (logits - 1.0 / (2.0 * beta)) ** 2

    elif loss_type == "kto_pair":
        # KTO (paired): does not require the two responses to be contrastive
        # Uses separate KL terms for chosen and rejected
        chosen_kl   = (policy_chosen_logps   - reference_chosen_logps).mean().clamp(min=0)
        rejected_kl = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_loss   = 1 - F.sigmoid(beta * (chosen_ratio   - chosen_kl))
        rejected_loss =     F.sigmoid(beta * (rejected_ratio - rejected_kl))
        loss = torch.cat([chosen_loss, rejected_loss], dim=0)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss.mean(), chosen_reward, rejected_reward


# ── Reward accuracy (auxiliary metric) ───────────────────────────────────────

def reward_accuracy(chosen_reward: Tensor, rejected_reward: Tensor) -> Tensor:
    """Fraction of pairs where chosen reward > rejected reward."""
    return (chosen_reward > rejected_reward).float().mean()


def reward_margin(chosen_reward: Tensor, rejected_reward: Tensor) -> Tensor:
    """Mean margin between chosen and rejected rewards."""
    return (chosen_reward - rejected_reward).mean()
