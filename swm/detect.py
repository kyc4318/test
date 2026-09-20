"""Detection: matched filtering, soft bits and raw reliability."""

from __future__ import annotations

import numpy as np
import torch


def matched_filter(
    Z_hat: torch.Tensor,
    carriers: torch.Tensor,
    ann: np.ndarray,
    channel: int,
    eps: float = 1e-8,
):
    """Z_hat: [1, C, H, W] complex, fftshifted spectrum of the inverted latents.

    Returns soft decisions ell [K_ind] and carrier squared norms.
    r_j = Re<Z_hat, Psi_j> ;  ell_j = r_j / (||Psi_j||^2 + eps).
    """
    z = Z_hat[0, channel][ann]  # [P] complex
    Psi = carriers[:, ann]      # [K_ind, P] real
    r = (Psi * z).sum(dim=1).real
    n2 = (Psi ** 2).sum(dim=1) + eps
    ell = r / n2
    return ell, n2


def reliability(
    Z_hat: torch.Tensor,
    carriers: torch.Tensor,
    ann: np.ndarray,
    channel: int,
    ell: torch.Tensor,
    chip: np.ndarray,
    K_ind: int,
    eps: float = 1e-8,
):
    """Per-bit raw reliability q_j = |ell_j| / (sqrt(nu_j) + eps), where
    nu_j is the local residual variance on chip j (residual after removing the
    estimated signal components)."""
    z = Z_hat[0, channel][ann]
    Psi = carriers[:, ann]
    res = z - (ell[:, None] * Psi).sum(dim=0)  # complex residual
    idx = torch.from_numpy(chip[ann] % K_ind)
    nu = torch.zeros(K_ind, dtype=torch.float32, device=z.device)
    for j in range(K_ind):
        m = idx == j
        if m.any():
            nu[j] = (res[m].abs() ** 2).mean()
    q = ell.abs() / (torch.sqrt(nu) + eps)
    return q, nu


def decode(
    Z_hat: torch.Tensor,
    carriers: torch.Tensor,
    ann: np.ndarray,
    channel: int,
    chip: np.ndarray,
    K_ind: int,
):
    """Full decode: soft ell, hard predictions and reliability."""
    ell, _ = matched_filter(Z_hat, carriers, ann, channel)
    pred = (ell > 0).long().cpu().numpy()
    q, nu = reliability(Z_hat, carriers, ann, channel, ell, chip, K_ind)
    return ell.cpu().numpy(), q.cpu().numpy(), pred
