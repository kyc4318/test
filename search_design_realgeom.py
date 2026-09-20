"""Search carrier designs under the *real* rotation geometry (no diffusion).

The alias curve is measured the way the decoder actually sees it: the watermark
latent is spatially rotated with the same operator as the decoder (physical
centre, bilinear, zero fill), FFT-ed, and matched against the carriers.  This
includes the half-pixel phase ramp that breaks the structural degeneracy, so the
numbers transfer to the pipeline much better than an ideal chip-shift model.

Objective:  rho = max_{delta not near 0 mod 180} S(delta) / S(0)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.linalg import hadamard
from torchvision.transforms import functional as TF


def annulus(size=64, r1=10, r2=20):
    cy = cx = size // 2
    yy, xx = np.ogrid[:size, :size]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return (r >= r1) & (r <= r2)


def uniform_edges(K):
    return np.arange(K + 1) * (360.0 / K)


def widths_to_edges(w):
    half = np.concatenate([[0.0], np.cumsum(w)])
    return np.concatenate([half[:-1], half[:-1] + 180.0, [360.0]])


def random_widths(M, rng, min_deg=3.0):
    w = np.exp(rng.uniform(0.1, 0.5) * rng.randn(M))
    w = w / w.sum() * 180.0
    for _ in range(50):
        bad = w < min_deg
        if not bad.any():
            break
        ex = (min_deg - w[bad]).sum()
        w[bad] = min_deg
        ok = ~bad
        w[ok] -= ex * w[ok] / w[ok].sum()
    return w


class Layer:
    def __init__(self, size, r_lo, r_hi, edges, phi, seed, ann, th_full, device,
                 C=None):
        self.r_lo, self.r_hi = r_lo, r_hi
        self.edges = np.asarray(edges, float)
        self.K = len(self.edges) - 1
        self.M = self.K // 2
        self.phi = np.asarray(phi, float)
        self.size, self.device = size, device
        cy = size // 2
        yy, xx = np.ogrid[:size, :size]
        r = np.sqrt((xx - cy) ** 2 + (yy - cy) ** 2)
        self.band = (r >= r_lo) & (r < r_hi) & ann
        q = np.searchsorted(self.edges[1:], th_full, side="right") % self.K
        self.q = q
        sel = self.band[ann]
        self.sel = sel
        qs = q[sel]
        self.Nq = np.array([
            float(((qs % self.K == j) | (qs % self.K == j + self.M)).sum())
            for j in range(self.M)])
        if C is None:
            rng = np.random.RandomState(seed)
            A, _ = np.linalg.qr(rng.randn(8, self.M).T)
            self.C = A[:, :8].T / np.sqrt(self.Nq)[None, :]
        else:
            self.C = np.asarray(C, dtype=float)
        self.rebuild()

    def rebuild(self):
        sel = self.sel
        qs = self.q[sel]
        sign = np.where(qs < self.M, 1.0, -1.0)
        qm = qs % self.M
        Psi = np.zeros((self.C.shape[0], len(self.q)), np.complex64)
        Psi[:, sel] = self.C[:, qm] * np.exp(1j * self.phi[qm] * sign)
        self.Psi = torch.from_numpy(Psi).to(self.device)
        self.norms = (np.abs(Psi) ** 2).sum(axis=1)
        self.norms_t = torch.from_numpy(self.norms).to(self.device)

    def watermark_latent(self, bits, ann_flat):
        b = torch.as_tensor(bits, dtype=torch.float32, device=self.device)
        W = torch.complex(b, torch.zeros_like(b)) @ self.Psi
        spec = torch.zeros(self.size * self.size, dtype=torch.complex64,
                           device=self.device)
        spec[ann_flat] = W
        spec = spec.reshape(self.size, self.size)
        return torch.fft.ifft2(torch.fft.ifftshift(spec)[None], dim=(-1, -2)).real[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_step", type=float, default=2.0)
    ap.add_argument("--payloads", type=int, default=6)
    ap.add_argument("--random_designs", type=int, default=120)
    ap.add_argument("--refine_tries", type=int, default=5)
    ap.add_argument("--refine_steps", type=int, default=2)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    size = 64
    random_rot = 15.0  # not used; kept for clarity of the objective
    ann = annulus(size)
    coords = np.argwhere(ann)
    cy = cx = size // 2
    th_full = (np.degrees(np.arctan2(coords[:, 0] - cy, coords[:, 1] - cx))) % 360.0
    ann_flat = torch.from_numpy(ann.reshape(-1)).to(device)
    grid = np.arange(0.0, 360.0, args.grid_step)
    rng = np.random.RandomState(0)
    payloads = rng.choice([-1.0, 1.0], size=(args.payloads, 8))

    def make_layers(spec):
        """spec: list of (r_lo, r_hi, edges, phi, seed)"""
        return [Layer(size, r_lo, r_hi, e, p, s, ann, th_full, device)
                for (r_lo, r_hi, e, p, s) in spec]

    def alias_ratio(layers, fusion="prod", payloads=payloads, grid=grid):
        ratios = []
        for b in payloads:
            S_layer = []
            for lay in layers:
                zW = lay.watermark_latent(b, ann_flat)
                sc = []
                for g in grid:
                    zr = TF.rotate(zW[None, None], -float(g), fill=0)[0, 0]
                    Z = torch.fft.fftshift(torch.fft.fft2(zr, dim=(-1, -2)),
                                           dim=(-1, -2))
                    ell = (torch.conj(lay.Psi) @ Z.reshape(-1)[ann_flat]) / lay.norms_t
                    sc.append(float(ell.real.abs().mean()))
                S_layer.append(np.array(sc))
            S = np.stack(S_layer)
            if fusion == "prod":
                M = np.prod(S / S[:, :1], axis=0)
            elif fusion == "min":
                M = (S / S[:, :1]).min(axis=0)
            else:
                M = (S / S[:, :1]).mean(axis=0)
            far = np.abs((grid + 90) % 180 - 90) > 4.0
            ratios.append(M[far].max())
        return float(np.mean(ratios)), float(np.max(ratios))

    # ---------------- reference designs ------------------------------------ #
    print("=== reference designs")
    H = hadamard(16, dtype=float)
    from swm.carriers import build_carriers
    car_w, C_w, N_w, norms_w, ann_w, chip_w = build_carriers(size, 10, 20, 16,
                                                            "walsh", seed=0,
                                                            device=device)
    def alias_ratio_walsh(K=16):
        car, C, N, norms, annl, chip = build_carriers(size, 10, 20, K, "walsh",
                                                      seed=0, device=device)
        car, C, norms = car[:8].to(device), C[:8].to(device), norms[:8].to(device)
        annl_flat = torch.from_numpy(annl.reshape(-1)).to(device)
        ratios = []
        for b in payloads:
            v = C.t() @ torch.as_tensor(b, dtype=torch.float32, device=device)
            spec = torch.zeros(size * size, dtype=torch.complex64, device=device)
            spec[annl_flat] = v[torch.from_numpy(chip[annl] % (K // 2)).to(
                device)].to(torch.complex64)
            zW = torch.fft.ifft2(torch.fft.ifftshift(spec.reshape(size, size))[None],
                                 dim=(-1, -2)).real[0]
            sc = []
            for g in grid:
                zr = TF.rotate(zW[None, None], -float(g), fill=0)[0, 0]
                Z = torch.fft.fftshift(torch.fft.fft2(zr, dim=(-1, -2)), dim=(-1, -2))
                zc = Z.reshape(-1)[annl_flat]
                ell = (car.reshape(car.shape[0], -1)[:, annl_flat] * zc[None, :]
                       ).sum(dim=1).real / norms
                sc.append(float(ell.abs().mean()))
            sc = np.array(sc)
            far = np.abs((grid + 90) % 180 - 90) > 4.0
            ratios.append(sc[far].max() / sc[np.argmin(np.abs(grid))])
        return float(np.mean(ratios))

    r_walsh16 = alias_ratio_walsh(16)
    r_walsh32 = alias_ratio_walsh(32)
    print(f"  K16 Walsh : rho={r_walsh16:.3f}")
    print(f"  K32 Walsh : rho={r_walsh32:.3f}")

    # ---------------- random search ---------------------------------------- #
    print("\n=== random search")
    cands = []
    shell = [10.0, 15.811, 20.0]
    for k in range(args.random_designs):
        n_layers = 1 if k % 3 == 0 else 2
        spec = []
        for li in range(n_layers):
            r_lo, r_hi = (shell[0], shell[2]) if n_layers == 1 else (shell[li], shell[li + 1])
            style = rng.randint(0, 3)
            if style == 0:
                K = int(rng.choice([16, 20, 24, 28, 32]))
                edges = uniform_edges(K)
            elif style == 1:
                K = int(rng.choice([30, 32, 34, 36]))
                edges = uniform_edges(K)
            else:
                M = int(rng.choice([8, 10, 12, 14, 16]))
                edges = widths_to_edges(random_widths(M, rng))
                K = 2 * M
            phi = rng.uniform(-np.pi, np.pi, K // 2)
            spec.append((r_lo, r_hi, edges, phi, 100 + k * 7 + li))
        lay = make_layers(spec)
        m, mx = alias_ratio(lay)
        cands.append({"spec": spec, "rho": m, "rho_max": mx,
                      "fusion": "prod" if k % 2 else "min"})
    cands.sort(key=lambda c: c["rho"])
    for c in cands[:5]:
        print(f"  rho={c['rho']:.3f} (max {c['rho_max']:.3f}) "
              f"layers={len(c['spec'])} fusion={c['fusion']} "
              f"K={[len(s[2]) - 1 for s in c['spec']]}")

    # ---------------- greedy phase refinement ------------------------------ #
    print("\n=== phase refinement (top-3)")
    refined = []
    phase_rng = np.random.RandomState(12345)
    for c in cands[:3]:
        spec = [(r_lo, r_hi, e, p.copy(), s) for (r_lo, r_hi, e, p, s) in c["spec"]]
        lay = make_layers(spec)
        best = alias_ratio(lay, c["fusion"])[0]
        for step in range(args.refine_steps):
            for li, l in enumerate(lay):
                for m in range(l.M):
                    keep = l.phi[m]
                    for _ in range(args.refine_tries):
                        cand_phi = phase_rng.uniform(-np.pi, np.pi)
                        trial = l.phi.copy()
                        trial[m] = cand_phi
                        l.phi = trial
                        l.rebuild()
                        rho = alias_ratio(lay, c["fusion"])[0]
                        if rho < best - 1e-4:
                            best, keep = rho, cand_phi
                    l.phi[m] = keep
                    l.rebuild()
        refined.append({"spec": spec, "fusion": c["fusion"], "rho": best})
        print(f"  refined rho={best:.3f} fusion={c['fusion']} "
              f"K={[len(s[2]) - 1 for s in spec]} "
              f"bands={[round(s[1] - s[0], 2) for s in spec]}")

    # ---------------- save the best design --------------------------------- #
    refined.sort(key=lambda c: c["rho"])
    best = refined[0]
    os.makedirs(args.out_dir, exist_ok=True)
    arrs = {"meta": np.array(json.dumps(
        {"name": "searched_realgeom", "rho": best["rho"], "fusion": best["fusion"],
         "capacity_bits": 8}))}
    for i, (r_lo, r_hi, edges, phi, seed) in enumerate(best["spec"]):
        lay = Layer(size, r_lo, r_hi, edges, phi, seed, ann, th_full, device)
        arrs[f"layer{i}_edges"] = edges
        arrs[f"layer{i}_C"] = lay.C
        arrs[f"layer{i}_phi"] = phi
        arrs[f"layer{i}_band"] = np.array([r_lo, r_hi])
        arrs[f"layer{i}_N"] = lay.Nq
    np.savez(os.path.join(args.out_dir, "design_searched_realgeom.npz"), **arrs)
    with open(os.path.join(args.out_dir, "design_search_realgeom.json"), "w") as f:
        json.dump({"walsh16_rho": r_walsh16, "walsh32_rho": r_walsh32,
                   "random_top": [{k: v for k, v in c.items() if k != "spec"}
                                  for c in cands[:10]],
                   "refined": [{"rho": c["rho"], "fusion": c["fusion"],
                                "K": [len(s[2]) - 1 for s in c["spec"]],
                                "bands": [[s[0], s[1]] for s in c["spec"]]}
                               for c in refined]}, f, indent=2)
    print(f"\nsaved best design (rho={best['rho']:.3f}) -> "
          f"{args.out_dir}/design_searched_realgeom.npz")


if __name__ == "__main__":
    main()
