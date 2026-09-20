"""Dual-layer shift-identifiable sector carriers (phase coded, product fusion).

Each layer occupies its own radial band and its own *angular* partition; the
partitions are deliberately incommensurate (e.g. K=32 inner, K=30 outer) so
that a rotation cannot be an exact integer chip shift in both layers at once.
Carriers are Hermitian phase coded (breaks the structural 180-deg alias) and
the two layers are fused with a product, which multiplies the alias suppression
of the single layers.

Designs are produced by ``search_shift_identifiable_designs.py`` (CPU) and
stored as npz files with, per layer, ``edges``, ``C``, ``phi``, ``band`` and
``N``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LayerSpec:
    r_lo: float
    r_hi: float
    edges: np.ndarray
    C: np.ndarray
    phi: np.ndarray
    N: np.ndarray

    @property
    def K(self) -> int:
        return len(self.edges) - 1

    @property
    def M(self) -> int:
        return self.K // 2


def load_design(path: str) -> tuple[list[LayerSpec], dict]:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    layers = []
    i = 0
    while f"layer{i}_edges" in z.files:
        r_lo, r_hi = z[f"layer{i}_band"]
        layers.append(LayerSpec(float(r_lo), float(r_hi), z[f"layer{i}_edges"],
                                z[f"layer{i}_C"], z[f"layer{i}_phi"],
                                z[f"layer{i}_N"]))
        i += 1
    return layers, meta


class DualLayerWatermarker:
    """Embeds the same B-bit payload on every layer (diversity) and decodes
    each layer independently; sync uses the product of the layer scores."""

    def __init__(self, design_path: str, size: int = 64, channel: int = 3,
                 device: str = "cpu", energy_per_point: float = 1e4):
        self.specs, self.meta = load_design(design_path)
        self.size = size
        self.channel = channel
        self.device = device
        self.energy_per_point = energy_per_point
        cy = cx = size // 2
        yy, xx = np.ogrid[:size, :size]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        th = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0
        self.layers = []
        for spec in self.specs:
            band = (r >= spec.r_lo) & (r < spec.r_hi)
            q = np.searchsorted(spec.edges[1:], th[band], side="right") % spec.K
            sign = np.where(q < spec.M, 1.0, -1.0)
            qm = q % spec.M
            Psi = spec.C[:, qm] * np.exp(1j * spec.phi[qm] * sign)[None, :]
            self.layers.append({
                "spec": spec,
                "mask": band,
                "coords": np.argwhere(band),
                "q": q,
                "Psi": torch.from_numpy(Psi).to(device=device, dtype=torch.complex64),
                "norms": (np.abs(Psi) ** 2).sum(axis=1),
            })
            lay = self.layers[-1]
            lay["mask_t"] = torch.from_numpy(band).to(device)
            lay["norms_t"] = torch.from_numpy(lay["norms"]).float().to(device)
        self.n_points = sum(int(l["mask"].sum()) for l in self.layers)

    # ------------------------------------------------------------------ embed
    def embed(self, z_T: torch.Tensor, bits: np.ndarray) -> torch.Tensor:
        """bits: B-dim +/-1 vector, written on every layer (diversity mode)."""
        n_layers = len(self.layers)
        Z = torch.fft.fftshift(torch.fft.fft2(z_T.float(), dim=(-1, -2)),
                               dim=(-1, -2)).clone()
        bits_t = torch.as_tensor(np.asarray(bits)[:, None], dtype=torch.float32,
                                 device=z_T.device)
        # total energy E_f = eta * N_ann is shared equally across layers
        e_total = self.energy_per_point * self.n_points
        for lay in self.layers:
            b = bits_t.squeeze(-1).float()
            W = torch.complex(b, torch.zeros_like(b)) @ lay["Psi"]   # (P,) complex
            e = (W.abs() ** 2).sum()
            W = W * torch.sqrt(torch.tensor(e_total / n_layers,
                                            dtype=torch.float32) / (e + 1e-12))
            mask = torch.from_numpy(lay["mask"]).to(z_T.device)
            Z[0, self.channel][mask] = W
        z_w = torch.fft.ifft2(torch.fft.ifftshift(Z, dim=(-1, -2)),
                              dim=(-1, -2)).real
        return z_w.to(z_T.dtype)

    # ----------------------------------------------------------------- decode
    def decode_layers(self, z_hat: torch.Tensor):
        """Return per-layer ell (B,) and the mean|ell| score."""
        Z = torch.fft.fftshift(torch.fft.fft2(z_hat.float(), dim=(-1, -2)),
                               dim=(-1, -2))
        ells, scores = [], []
        for lay in self.layers:
            z = Z[0, self.channel][lay["mask_t"]]
            ell = (torch.conj(lay["Psi"]) * z[None, :]).sum(dim=1).real / lay["norms_t"]
            ells.append(ell)
            scores.append(ell.abs().mean())
        return ells, scores

    def score(self, z_hat: torch.Tensor) -> float:
        """Product fusion of the per-layer scores (scale invariant per layer)."""
        _, scores = self.decode_layers(z_hat)
        out = torch.ones((), device=z_hat.device)
        for s in scores:
            out = out * s
        return float(out)

    def hermitian_error(self) -> float:
        err = 0.0
        for lay in self.layers:
            coords = lay["coords"]
            lookup = -np.ones((self.size, self.size), dtype=np.int64)
            lookup[coords[:, 0], coords[:, 1]] = np.arange(len(coords))
            partner = (-coords) % self.size
            pidx = lookup[partner[:, 0], partner[:, 1]]
            ok = pidx >= 0
            Psi = lay["Psi"].cpu().numpy()
            err = max(err, float(np.abs(Psi[:, ok] - np.conj(Psi[:, pidx[ok]])).max()))
        return err

    def weighted_orth_error(self) -> float:
        err = 0.0
        for lay in self.layers:
            spec = lay["spec"]
            G = spec.C @ np.diag(spec.N) @ spec.C.T
            err = max(err, float(np.abs(G - np.eye(spec.C.shape[0])).max()))
        return err
