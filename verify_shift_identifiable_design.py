"""Reload a saved shift-identifiable design and verify it independently.

Checks
  1. weighted orthogonality C diag(N) C^T = I and Hermitian completion
  2. clean decodability: sign(ell_j(0)) == b_j for every message, every layer
  3. alias profile A(delta) and the interior/half/other worst-case values
  4. noise robustness of the synchroniser

    python verify_shift_identifiable_design.py --npz results/design_0_*.npz
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

import search_shift_identifiable_designs as S


def load_design(path: str) -> S.Design:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    layers = []
    i = 0
    while f"layer{i}_edges" in z.files:
        r_lo, r_hi = z[f"layer{i}_band"]
        lay = S.Layer(r_lo, r_hi, z[f"layer{i}_edges"], z[f"layer{i}_phi"], seed=0,
                      label=f"L{i}")
        # the saved code matrix is authoritative (phase search may have run)
        lay.C = z[f"layer{i}_C"]
        lay.N = z[f"layer{i}_N"]
        lay.norms = np.array([
            float((np.abs(lay.C[j]) ** 2 * lay.N).sum())
            for j in range(lay.C.shape[0])
        ])
        lay.Psi = lay._carriers()
        Psi_s = lay.Psi[:, lay.order]
        lay.csum = np.concatenate(
            [np.zeros((lay.C.shape[0], 1), dtype=complex),
             np.cumsum(np.conj(Psi_s), axis=1)],
            axis=1,
        )
        layers.append(lay)
        i += 1
    return S.Design(layers, fusion=meta["fusion"], name=meta["name"],
                    capacity_bits=meta["capacity_bits"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/design_0_*.npz")
    ap.add_argument("--step", type=float, default=0.5)
    args = ap.parse_args()
    for path in sorted(glob.glob(args.npz)):
        design = load_design(path)
        B = design.layers[0].C.shape[0]
        msgs = S.messages_matrix(B, limit=256 if B > 8 else 0)
        deltas = np.arange(0.0, 360.0 + 1e-9, args.step)
        print(f"=== {path}")
        for lay in design.layers:
            ell0 = lay.scores(msgs, np.array([0.0]))[0]
            print(f"  {lay.label}: K={lay.K} M={lay.M} band=[{lay.r_lo:.2f},"
                  f"{lay.r_hi:.2f}) pts={lay.n_points} orth_err={lay.weighted_orth_error():.1e} "
                  f"herm_err={lay.hermitian_error():.1e} phase={np.any(np.abs(lay.phi)>1e-9)}")
            print(f"        clean S(0): mean={ell0.mean():.6f} min={ell0.min():.6f}")
            # bit correctness at delta = 0 (sign of ell must equal the payload)
            V = lay.chip_values(msgs)
            ell = (lay.G(0.0) @ V) / lay.norms[:, None]
            ok = np.mean((ell.real > 0) == (msgs > 0))
            print(f"        clean bit accuracy at delta=0: {ok:.6f}")
        row = S.summarise(design, msgs, deltas)
        print(f"  A(45/90/135/180/270) = {row['A(45)']:.3f}/{row['A(90)']:.3f}/"
              f"{row['A(135)']:.3f}/{row['A(180)']:.3f}/{row['A(270)']:.3f}")
        print(f"  A_int={row['A_int_worst']:.3f} (mean {row['A_int_mean']:.3f})  "
              f"A_half={row['A_half_worst']:.3f}  A_other={row['A_other_worst']:.3f}")
        print(f"  sync margin (1 - A_int) = {1 - row['A_int_worst']:.3f}   "
              f"p_alias(.05)={row['p_alias']['0.05']:.3f}")
        ells = [S.layer_ell(lay, msgs[:, :64], np.arange(0.0, 360.0, 1.0))
                for lay in design.layers]
        rng = np.random.RandomState(0)
        nr = S.noise_sync_fail(design, ells, np.arange(0.0, 360.0, 1.0),
                               [0.0, 0.05, 0.1, 0.2], [1.0, 0.12, 0.05], 3, rng)
        for n in nr:
            if n["sigma"] in (0.0, 0.05, 0.1):
                print(f"    noise scale={n['scale']:<5} sigma={n['sigma']:<5} "
                      f"fail(mod180)={n['sync_fail_mod180']:.3f} "
                      f"fail(mod360)={n['sync_fail_mod360']:.3f}")
        print()


if __name__ == "__main__":
    main()
