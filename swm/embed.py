"""Embedding the sectorized watermark into the initial latents."""

from __future__ import annotations

import numpy as np
import torch


def build_watermark(
    chip: np.ndarray,
    ann: np.ndarray,
    C: torch.Tensor,
    bits: torch.Tensor,
    alpha: float = 1.0,
    total_energy: float | None = None,
) -> torch.Tensor:
    """W(omega) = alpha * sum_j bits[j] * C[j, s(omega)] on the annulus.

    bits: [K_ind] float (+-1).  Returns a real [size, size] tensor.  When
    ``total_energy`` is given the pattern is rescaled so that
    ``||W[ann]||^2 == total_energy`` (fixed total embedding energy, E_f).
    """
    K_ind = C.shape[0]
    # chip value on chip s: v_s = alpha * sum_j b_j * C[j, s] = alpha * (C^T b)_s
    v = alpha * (C.t() @ bits.float().to(C.device))  # [K_ind] chip values
    W = torch.zeros(chip.shape[0], chip.shape[1], dtype=torch.float32, device=C.device)
    idx = torch.from_numpy(chip)
    valid = idx >= 0
    W[valid] = v[idx[valid] % K_ind]
    if total_energy is not None:
        e = (W[valid] ** 2).sum()
        if e > 0:
            W[valid] = W[valid] * torch.sqrt(
                torch.tensor(total_energy, dtype=torch.float32, device=C.device) / e
            )
    return W


def inject(
    z_T: torch.Tensor,
    W: torch.Tensor,
    channel: int,
    ann: np.ndarray,
    mode: str = "replace",
):
    """Embed W into channel ``channel`` of the FFT spectrum of z_T.

    mode='replace': annulus coefficients are overwritten (Tree-Ring/METR style).
    mode='add':     W is added to the annulus coefficients (doc's additive form).
    Returns (z_w, Z_w): the watermarked real latents and its shifted spectrum.
    """
    # work in float32: torch.fft on float16 produces experimental ComplexHalf
    Z = torch.fft.fftshift(torch.fft.fft2(z_T.float()), dim=(-1, -2)).clone()
    a = torch.from_numpy(ann).to(z_T.device)
    Wc = W.to(Z.dtype)
    if mode == "replace":
        Z[0, channel][a] = Wc[a]
    elif mode == "add":
        Z[0, channel][a] = Z[0, channel][a] + Wc[a]
    else:
        raise ValueError(f"unknown injection mode: {mode}")
    z_w = torch.fft.ifft2(torch.fft.ifftshift(Z, dim=(-1, -2))).real
    return z_w.to(z_T.dtype), Z


def hermitian_error(Z: torch.Tensor) -> float:
    """Mean |Z - conj(partner(Z))| over the fftshifted spectrum.

    For a real signal the shifted spectrum satisfies
    Z[i, j] == conj(Z[(N-i)%N, (N-j)%N]).  The partner map is
    ``flip`` followed by a roll of 1 (flip gives N-1-i, roll gives N-i).
    Should be ~0 for a real signal.
    """
    partner = torch.roll(torch.flip(Z, dims=(-1, -2)), shifts=(1, 1), dims=(-1, -2))
    return (Z - torch.conj(partner)).abs().mean().item()
