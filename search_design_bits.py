"""Capacity / multi-key design search (independent script; CPU only).

``search_design_realgeom.py`` is the canonical search and is hard-wired to
B = 8 bits.  This script re-implements the same pipeline with

    --bits B          payload length (B <= K_l/2 for every layer)
    --seed S          random seed, so several independent keys can be produced

and writes an npz with exactly the same schema as the canonical design, so
``swm.dual_layer.load_design`` and every existing runner can load it unchanged.
The canonical script is not touched.

    python search_design_bits.py --bits 16 --seed 0 --out_name design_B16_s0
    python search_design_bits.py --bits 8  --seed 1 --out_name design_B8_s1
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torchvision.transforms import functional as TF

from search_design_realgeom import (Layer, annulus, random_widths, uniform_edges,
                                    widths_to_edges)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid_step", type=float, default=2.0)
    ap.add_argument("--payloads", type=int, default=6)
    ap.add_argument("--random_designs", type=int, default=120)
    ap.add_argument("--refine_tries", type=int, default=5)
    ap.add_argument("--refine_steps", type=int, default=2)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--out_name", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    B = args.bits
    device = args.device
    size = 64
    ann = annulus(size)
    coords = np.argwhere(ann)
    cy = cx = size // 2
    th_full = (np.degrees(np.arctan2(coords[:, 0] - cy, coords[:, 1] - cx))) % 360.0
    ann_flat = torch.from_numpy(ann.reshape(-1)).to(device)
    grid = np.arange(0.0, 360.0, args.grid_step)
    rng = np.random.RandomState(args.seed)
    payloads = rng.choice([-1.0, 1.0], size=(args.payloads, B))

    def code_for(lay, seed):
        """B x M orthonormal-in-the-weighted-sense code, normalized on chip sizes."""
        crng = np.random.RandomState(seed)
        A, _ = np.linalg.qr(crng.randn(B, lay.M).T)      # (M, B)
        return (A[:, :B].T / np.sqrt(lay.Nq)[None, :]).astype(float)

    def make_layers(spec):
        """spec: list of (r_lo, r_hi, edges, phi, seed); C is rebuilt for B bits."""
        layers = []
        for (r_lo, r_hi, e, p, s) in spec:
            lay = Layer(size, r_lo, r_hi, e, p, 0, ann, th_full, device)
            if lay.M < B:
                raise ValueError(f"layer with M={lay.M} cannot carry B={B} bits")
            lay.C = code_for(lay, s)
            lay.rebuild()
            layers.append(lay)
        return layers

    def alias_ratio(layers, fusion="prod"):
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
                S_layer.append(np.asarray(sc))
            S = np.stack(S_layer)
            if fusion == "prod":
                M = np.prod(S / S[:, :1], axis=0)
            elif fusion == "min":
                M = (S / S[:, :1]).min(axis=0)
            else:
                M = (S / S[:, :1]).mean(axis=0)
            far = np.abs((grid + 90) % 180 - 90) > 4.0
            ratios.append(float(M[far].max()))
        return float(np.mean(ratios)), float(np.max(ratios))

    def alias_pi(layers, fusion="prod"):
        i180 = int(np.argmin(np.abs(grid - 180.0)))
        i90 = int(np.argmin(np.abs(grid - 90.0)))
        vals = []
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
                S_layer.append(np.asarray(sc))
            S = np.stack(S_layer)
            M = (np.prod(S / S[:, :1], axis=0) if fusion == "prod"
                 else (S / S[:, :1]).mean(axis=0))
            vals.append([float(M[i90]), float(M[i180])])
        A = np.asarray(vals)
        return float(A[:, 0].mean()), float(A[:, 1].mean())

    print(f"=== search: B={B} seed={args.seed} designs={args.random_designs} ===")
    cands = []
    shell = [10.0, 15.811, 20.0]
    for k in range(args.random_designs):
        n_layers = 1 if k % 3 == 0 else 2
        spec = []
        ok = True
        for li in range(n_layers):
            r_lo, r_hi = ((shell[0], shell[2]) if n_layers == 1
                          else (shell[li], shell[li + 1]))
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
            if K // 2 < B:                     # cannot carry B bits
                ok = False
                break
            phi = rng.uniform(-np.pi, np.pi, K // 2)
            spec.append((r_lo, r_hi, edges, phi, 1000 + args.seed * 97 + k * 7 + li))
        if not ok:
            continue
        lay = make_layers(spec)
        m, mx = alias_ratio(lay)
        cands.append({"spec": spec, "rho": m, "rho_max": mx,
                      "fusion": "prod" if k % 2 else "min"})
    if not cands:
        raise SystemExit("no valid candidate for this B")
    cands.sort(key=lambda c: c["rho"])
    for c in cands[:5]:
        print(f"  rho={c['rho']:.3f} (max {c['rho_max']:.3f}) "
              f"layers={len(c['spec'])} fusion={c['fusion']} "
              f"K={[len(s[2]) - 1 for s in c['spec']]}")

    print("\n=== phase refinement (top-3) ===")
    refined = []
    phase_rng = np.random.RandomState(12345 + args.seed)
    for c in cands[:3]:
        spec = [(r_lo, r_hi, e, p.copy(), s) for (r_lo, r_hi, e, p, s) in c["spec"]]
        lay = make_layers(spec)
        best = alias_ratio(lay, c["fusion"])[0]
        for _ in range(args.refine_steps):
            for l in lay:
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
        a90, a180 = alias_pi(lay, c["fusion"])
        refined.append({"spec": spec, "fusion": c["fusion"], "rho": best,
                        "A_90": a90, "A_180": a180})
        print(f"  refined rho={best:.3f} A_90={a90:.3f} A_180={a180:.3f} "
              f"fusion={c['fusion']} K={[len(s[2]) - 1 for s in spec]}")

    refined.sort(key=lambda c: c["rho"])
    best = refined[0]
    os.makedirs(args.out_dir, exist_ok=True)
    name = args.out_name or f"design_B{B}_s{args.seed}"
    arrs = {"meta": np.array(json.dumps(
        {"name": name, "rho": best["rho"], "fusion": best["fusion"],
         "capacity_bits": B, "seed": args.seed, "A_90": best["A_90"],
         "A_180": best["A_180"]}))}
    for i, (r_lo, r_hi, edges, phi, seed) in enumerate(best["spec"]):
        lay = make_layers([(r_lo, r_hi, edges, phi, seed)])[0]
        arrs[f"layer{i}_edges"] = edges
        arrs[f"layer{i}_C"] = lay.C
        arrs[f"layer{i}_phi"] = phi
        arrs[f"layer{i}_band"] = np.array([r_lo, r_hi])
        arrs[f"layer{i}_N"] = lay.Nq
    path = os.path.join(args.out_dir, f"{name}.npz")
    np.savez(path, **arrs)
    with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
        json.dump({"bits": B, "seed": args.seed, "best": {
            k: v for k, v in best.items() if k != "spec"},
            "K": [len(s[2]) - 1 for s in best["spec"]],
            "fusion": best["fusion"]}, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
