"""Measure the 180-degree alias A(pi) (and A(90)) for every ablation variant.

Complements ``ablate_design.py``: that script's metric

    rho_mod180 = max_{dist_180(delta, 0) > tol} A(delta)

excludes delta ~ 180 deg **by construction**, so it cannot see what the
Hermitian phase mask is for.  This script reports, for each variant, all three
numbers on the same real rotation geometry:

    rho_mod180   (reproduces the published ablation column)
    A_90         (the common integer-chip alias of the two grids)
    A_180        (the structural ambiguity of any real-valued Hermitian carrier)

    python measure_alias_pi.py --grid_step 1 --payloads 6
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.linalg import hadamard
from torchvision.transforms import functional as TF

from search_design_realgeom import Layer, annulus, uniform_edges
from swm.dual_layer import load_design


def walsh_code(B: int, M: int) -> np.ndarray:
    return (hadamard(M) / np.sqrt(M))[:B, :].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="results/design_searched_realgeom.npz")
    ap.add_argument("--grid_step", type=float, default=1.0)
    ap.add_argument("--payloads", type=int, default=6)
    ap.add_argument("--tol", type=float, default=4.0)
    ap.add_argument("--out", default="results/alias_pi_ablation.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    size = 64
    ann = annulus(size)
    coords = np.argwhere(ann)
    cy = cx = size // 2
    th = np.degrees(np.arctan2(coords[:, 0] - cy, coords[:, 1] - cx)) % 360.0
    ann_flat = np.flatnonzero(ann.reshape(-1))
    grid = np.arange(0.0, 360.0, args.grid_step)
    i90 = int(np.argmin(np.abs(grid - 90.0)))
    i180 = int(np.argmin(np.abs(grid - 180.0)))
    far = np.abs((grid + 90) % 180 - 90) > args.tol
    rng = np.random.RandomState(0)
    payloads = rng.choice([-1.0, 1.0], size=(args.payloads, 8))
    specs, meta = load_design(args.design)
    print("design meta:", meta)

    def mk(r_lo, r_hi, edges, phi, C):
        return Layer(size, r_lo, r_hi, edges, phi, 0, ann, th, args.device, C=C)

    def measure(layers, fusion="prod"):
        """Report, per variant, three metrics that need *different* rotations:

        rho_mod180 / A_90 / A_180   -- grid metric under the decoder's real
                                       rotation operator (torchvision bilinear,
                                       zero padding), i.e. what the pipeline sees
        A_180_exact                 -- A(pi) under the **exact** 180-deg rotation
                                       about the **physical** centre (index 31.5)
                                       = a pure array flip
        A_180_dft                   -- A(pi) under the **DFT-symmetric** rotation
                                       n -> -n (flip + roll by one sample), which
                                       carries no half-pixel phase ramp.  This is
                                       the operator Proposition 2 is about: for a
                                       real Hermitian carrier it must be exactly 1.

        Reporting all three separates two effects that the old single number
        conflated: the structural pi-ambiguity of real carriers (visible only in
        A_180_dft) and the half-pixel phase ramp of the physical rotation, which
        is Proposition 3 and attenuates *every* carrier.
        """
        rows = []
        for b in payloads:
            S = []
            S_exact = []
            S_dft = []
            for lay in layers:
                zW = lay.watermark_latent(b, ann_flat)
                sc = []
                for g in grid:
                    zr = TF.rotate(zW[None, None], -float(g), fill=0)[0, 0]
                    Z = torch.fft.fftshift(torch.fft.fft2(zr, dim=(-1, -2)),
                                           dim=(-1, -2))
                    ell = (torch.conj(lay.Psi) @ Z.reshape(-1)[ann_flat]) / lay.norms_t
                    sc.append(float(ell.real.abs().mean()))
                S.append(np.asarray(sc))
                # exact 180-deg rotation about the array centre = pure flip
                zf = torch.flip(zW, dims=(-1, -2))
                Zf = torch.fft.fftshift(torch.fft.fft2(zf, dim=(-1, -2)),
                                        dim=(-1, -2))
                ell_f = (torch.conj(lay.Psi) @ Zf.reshape(-1)[ann_flat]) / lay.norms_t
                S_exact.append(float(ell_f.real.abs().mean()))
                # DFT-symmetric 180-deg rotation n -> -n (flip + roll): no ramp
                zd = torch.roll(zf, shifts=(1, 1), dims=(-1, -2))
                Zd = torch.fft.fftshift(torch.fft.fft2(zd, dim=(-1, -2)),
                                        dim=(-1, -2))
                ell_d = (torch.conj(lay.Psi) @ Zd.reshape(-1)[ann_flat]) / lay.norms_t
                S_dft.append(float(ell_d.real.abs().mean()))
            S = np.stack(S)
            if fusion == "prod":
                M = np.prod(S / S[:, :1], axis=0)
                Me = float(np.prod(np.asarray(S_exact) / S[:, 0]))
                Md = float(np.prod(np.asarray(S_dft) / S[:, 0]))
            elif fusion == "min":
                M = (S / S[:, :1]).min(axis=0)
                Me = float((np.asarray(S_exact) / S[:, 0]).min())
                Md = float((np.asarray(S_dft) / S[:, 0]).min())
            else:
                M = (S / S[:, :1]).mean(axis=0)
                Me = float((np.asarray(S_exact) / S[:, 0]).mean())
                Md = float((np.asarray(S_dft) / S[:, 0]).mean())
            rows.append([float(M[far].max()), float(M[i90]), float(M[i180]), Me, Md])
        A = np.asarray(rows)
        return {"rho_mod180": float(A[:, 0].mean()),
                "A_90": float(A[:, 1].mean()),
                "A_180": float(A[:, 2].mean()),
                "A_180_max": float(A[:, 2].max()),
                "A_180_min": float(A[:, 2].min()),
                "A_180_center": float(A[:, 3].mean()),
                "A_180_dft": float(A[:, 4].mean())}

    sp1, sp2 = specs[0], specs[1]
    M1, M2 = sp1.M, sp2.M
    rng_phi = np.random.RandomState(7)
    rand_phi1 = rng_phi.uniform(-np.pi, np.pi, M1)
    rand_phi2 = rng_phi.uniform(-np.pi, np.pi, M2)
    K_same = sp1.K
    C_same = walsh_code(8, K_same // 2)

    variants = {
        "K16 Walsh (single layer, 8/8 full rate)":
            lambda: [mk(10.0, 20.0, uniform_edges(16), np.zeros(8), walsh_code(8, 8))],
        "K32 Walsh (single layer, 8/16)":
            lambda: [mk(10.0, 20.0, uniform_edges(32), np.zeros(16), walsh_code(8, 16))],
        "full design (two grids + phase, prod)":
            lambda: [mk(sp1.r_lo, sp1.r_hi, sp1.edges, sp1.phi, sp1.C),
                     mk(sp2.r_lo, sp2.r_hi, sp2.edges, sp2.phi, sp2.C)],
        "same design, min fusion":
            lambda: [mk(sp1.r_lo, sp1.r_hi, sp1.edges, sp1.phi, sp1.C),
                     mk(sp2.r_lo, sp2.r_hi, sp2.edges, sp2.phi, sp2.C)],
        "same design, mean fusion":
            lambda: [mk(sp1.r_lo, sp1.r_hi, sp1.edges, sp1.phi, sp1.C),
                     mk(sp2.r_lo, sp2.r_hi, sp2.edges, sp2.phi, sp2.C)],
        "same design, no phase mask (phi=0)":
            lambda: [mk(sp1.r_lo, sp1.r_hi, sp1.edges, np.zeros(M1), sp1.C),
                     mk(sp2.r_lo, sp2.r_hi, sp2.edges, np.zeros(M2), sp2.C)],
        "same design, random phase":
            lambda: [mk(sp1.r_lo, sp1.r_hi, sp1.edges, rand_phi1, sp1.C),
                     mk(sp2.r_lo, sp2.r_hi, sp2.edges, rand_phi2, sp2.C)],
        "inner layer only":
            lambda: [mk(sp1.r_lo, sp1.r_hi, sp1.edges, sp1.phi, sp1.C)],
        "outer layer only":
            lambda: [mk(sp2.r_lo, sp2.r_hi, sp2.edges, sp2.phi, sp2.C)],
        "two layers, same K":
            lambda: [mk(sp1.r_lo, sp1.r_hi, sp1.edges, sp1.phi, sp1.C),
                     mk(sp2.r_lo, sp2.r_hi, uniform_edges(K_same), np.zeros(K_same // 2),
                        C_same)],
    }
    fusion_of = {"same design, min fusion": "min",
                 "same design, mean fusion": "mean"}

    out = {}
    print(f"\n{'variant':<44}{'rho_m180':>10}{'A_90':>9}{'A_pi_tf':>9}"
          f"{'A_pi_ctr':>10}{'A_pi_dft':>10}")
    for name, build in variants.items():
        res = measure(build(), fusion_of.get(name, "prod"))
        out[name] = res
        print(f"{name:<44}{res['rho_mod180']:>10.3f}{res['A_90']:>9.3f}"
              f"{res['A_180']:>9.3f}{res['A_180_center']:>10.3f}"
              f"{res['A_180_dft']:>10.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "meta": meta, "variants": out}, f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
