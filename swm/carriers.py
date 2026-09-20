"""Sectorized frequency-domain carriers (first-phase method).

Coordinate conventions (same as Tree-Ring / METR / RingID):
  * working spectrum is ``torch.fft.fftshift(torch.fft.fft2(x))`` with DC at
    ``(size//2, size//2)``;
  * the carrier region is the annulus ``r1 <= rho <= r2``;
  * the annulus is split into ``K`` angular sectors; sector ``s`` and sector
    ``(s + K//2) % K`` are Hermitian-conjugate partners, hence only
    ``K_ind = K // 2`` independent chips exist.

Each independent bit is carried by one row of a ``K_ind x K_ind`` code
matrix C.  The watermark is piecewise constant on the chips:

    W(omega) = alpha * sum_j b_j * C[j, s(omega) % K_ind]    (omega in annulus)

Carrier types (Experiment 3 of the method document):
  * 'const' -> C = I      (sector-BPSK baseline, METR-style region mean)
  * 'rand'  -> keyed random +-1 matrix
  * 'walsh' -> Hadamard matrix (K_ind should be a power of two)
  * 'ortho' -> random matrix, weighted Gram-Schmidt orthogonalized on the
               actual sector supports (weights = sqrt(points per chip))
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import hadamard


def annulus_mask(size: int, r1: int, r2: int) -> np.ndarray:
    cy = cx = size // 2
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return (r >= r1) & (r <= r2)


def sector_chip_map(size: int, r1: int, r2: int, K: int) -> np.ndarray:
    """[size, size] int64: angular sector index in [0, K) on the annulus,
    else -1."""
    cy = cx = size // 2
    y, x = np.ogrid[:size, :size]
    ang = np.arctan2(y - cy, x - cx)  # [-pi, pi]
    s = np.floor((ang + np.pi) / (2.0 * np.pi / K)).astype(np.int64) % K
    m = annulus_mask(size, r1, r2)
    chip = np.full((size, size), -1, dtype=np.int64)
    chip[m] = s[m]
    return chip


def chip_point_counts(chip: np.ndarray, K: int) -> np.ndarray:
    """Points per independent chip (sector s plus its Hermitian partner)."""
    K_ind = K // 2
    N = np.zeros(K_ind, dtype=np.float64)
    for j in range(K_ind):
        N[j] = float(np.sum((chip == j) | (chip == (j + K_ind) % K)))
    return N


def make_code_matrix(
    K_ind: int, mode: str, seed: int = 0, weights: np.ndarray | None = None
) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    if mode == "const":
        C = np.eye(K_ind, dtype=np.float64)
    elif mode == "rand":
        C = rng.choice([-1.0, 1.0], size=(K_ind, K_ind))
    elif mode == "walsh":
        L = 1
        while L < K_ind:
            L *= 2
        C = hadamard(L, dtype=float)[:K_ind, :K_ind]
        if L != K_ind:
            print(
                f"[walsh] K_ind={K_ind} is not a power of two; "
                f"truncated {L}x{L} Hadamard (rows are not exactly orthogonal)."
            )
    elif mode == "ortho":
        C = rng.choice([-1.0, 1.0], size=(K_ind, K_ind)).astype(np.float64)
        if weights is not None:
            w2 = np.asarray(weights, dtype=np.float64)
            Q = np.zeros_like(C)
            for j in range(K_ind):
                v = C[j].copy()
                for i in range(j):
                    v -= np.sum(w2 * Q[i] * v) * Q[i]  # <Q_i, v>_w * Q_i
                nrm = np.sqrt(np.sum(w2 * v * v))
                Q[j] = v / max(nrm, 1e-12)
            C = Q
        else:
            C, _ = np.linalg.qr(C)
    else:
        raise ValueError(f"unknown carrier mode: {mode}")
    return torch.tensor(C, dtype=torch.float32)


def build_carriers(
    size: int, r1: int, r2: int, K: int, mode: str = "walsh",
    seed: int = 0, device: str = "cpu",
):
    """Build carriers [K_ind, size, size] (real, nonzero on the annulus).

    Returns (carriers, C, N, norms, ann, chip):
      carriers : [K_ind, size, size] float32 on `device`
      C        : [K_ind, K_ind] code matrix
      N        : per-chip point counts
      norms    : per-carrier squared L2 norms
      ann      : bool annulus mask
      chip     : int64 sector map (CPU numpy)
    """
    chip = sector_chip_map(size, r1, r2, K)
    K_ind = K // 2
    N = chip_point_counts(chip, K)
    C = make_code_matrix(K_ind, mode, seed=seed, weights=N)
    ann = chip >= 0
    carriers = torch.zeros(K_ind, size, size, dtype=torch.float32)
    ann_flat = ann.reshape(-1)
    idx = torch.from_numpy(chip[ann] % K_ind)  # [P]
    carriers_flat = carriers.view(K_ind, -1)
    carriers_flat[:, ann_flat] = C[:, idx]
    carriers = carriers.to(device)
    norms = (carriers ** 2).sum(dim=(-1, -2))
    return carriers, C, N, norms, ann, chip
