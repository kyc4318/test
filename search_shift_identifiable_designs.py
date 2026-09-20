"""Shift-identifiable sector codebook design search (CPU, no diffusion).

Motivation
----------
Orthogonality is necessary but NOT sufficient for decoder-native rotation
synchronisation: a full-rate orthogonal sector codebook still has

    A(180 deg) == 1        exactly, for every real Hermitian carrier,

because the chip pairing q <-> q+M forces the same carrier value on both
Hermitian halves, so a 180 deg rotation maps the watermark onto itself.
The binding requirement is *shift identifiability*: the alias curve

    A(delta) = mean_j |ell_j(delta)| / mean_j |ell_j(0)|        (per message)
    ell(delta) = G(delta) V / ||Psi||^2

must be small for every delta that is not a legitimate alignment.

Model
-----
Exact chip-domain rotation on the real 64x64 annulus grid (r1=10, r2=20):
a point p at angle th takes the watermark value of the chip that contains
th - delta.  The watermark is piecewise constant in angle, so this needs no
interpolation and is exact for arbitrary (also fractional) delta.  Carriers
may be real (phi = 0) or Hermitian phase coded (phi != 0), the latter being
the only way to break the structural A(180) = 1.

Designs
-------
 * angular partition type   : uniform | quasi-periodic | random widths
 * number of sectors K      : M = K/2 independent chips
 * radial layers            : 1 or 2 bands, each with its own partition
 * carrier                  : real | phase (phi searched)
 * fusion across layers     : min | product | mean

Run
---
    python search_shift_identifiable_designs.py --out_dir results
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np

SIZE = 64
R1, R2 = 10, 20
B = 8                       # payload bits per layer
TAU = (0.05, 0.10)          # alias tolerance for p_alias


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def annulus_geometry(size: int = SIZE, r1: int = R1, r2: int = R2):
    y, x = np.ogrid[:size, :size]
    c = size // 2
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2)
    th = (np.degrees(np.arctan2(y - c, x - c)) + 360.0) % 360.0
    mask = (r >= r1) & (r <= r2)
    return r, th, mask


R_GRID, TH_GRID, ANN = annulus_geometry()


def _annulus_index_maps():
    """Flat annulus index -> partner index (-p), used for Hermitian checks."""
    coords = np.argwhere(ANN)
    lookup = -np.ones((SIZE, SIZE), dtype=np.int64)
    lookup[coords[:, 0], coords[:, 1]] = np.arange(len(coords))
    partner = (-coords) % SIZE
    return coords, lookup[partner[:, 0], partner[:, 1]]


ANN_COORDS, ANN_PARTNER = _annulus_index_maps()


def messages_matrix(B: int, limit: int = 0, seed: int = 0) -> np.ndarray:
    """All 2^B messages as +/-1, or `limit` random ones, shape (B, n)."""
    if limit > 0 and (1 << B) > limit:
        rng = np.random.RandomState(seed)
        return rng.choice([-1.0, 1.0], size=(B, limit))
    idx = np.arange(1 << B, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(B, dtype=np.int64)) & 1
    return np.where(bits.T > 0, 1.0, -1.0)


# --------------------------------------------------------------------------- #
# angular partitions (always 180 deg periodic, required by Hermitian pairing)
# --------------------------------------------------------------------------- #
def uniform_edges(K: int) -> np.ndarray:
    return np.arange(K + 1) * (360.0 / K)


def widths_to_edges(widths_half: np.ndarray) -> np.ndarray:
    """widths over [0,180); edges are repeated for the conjugate half."""
    half = np.concatenate([[0.0], np.cumsum(widths_half)])
    assert abs(half[-1] - 180.0) < 1e-9
    return np.concatenate([half[:-1], half[:-1] + 180.0, [360.0]])


def aperiodic_widths(M: int, mode: str, rng: np.random.RandomState,
                     min_deg: float = 3.0) -> np.ndarray:
    target = 180.0 / M
    if mode == "quasi":
        a = rng.uniform(0.15, 0.5)
        psi = rng.uniform(0, 2 * np.pi)
        m = np.arange(M)
        w = 1.0 + a * np.cos(2 * np.pi * 0.61803398875 * m + psi)
    else:  # log-normal random
        w = np.exp(rng.uniform(0.1, 0.5) * rng.randn(M))
    w = w / w.sum() * 180.0
    # enforce a minimum chip width by iterative transfer
    for _ in range(50):
        too_small = w < min_deg
        if not too_small.any():
            break
        excess = (min_deg - w[too_small]).sum()
        w[too_small] = min_deg
        big = ~too_small
        w[big] -= excess * w[big] / w[big].sum()
    assert abs(w.sum() - 180.0) < 1e-9
    return w


# --------------------------------------------------------------------------- #
# one radial layer
# --------------------------------------------------------------------------- #
class Layer:
    """A radial band carrying B bits on M = K/2 independent angular chips."""

    def __init__(self, r_lo, r_hi, edges, phi, seed, label=""):
        self.label = label
        self.r_lo, self.r_hi = r_lo, r_hi
        self.edges = np.asarray(edges, dtype=float)
        self.K = len(self.edges) - 1
        self.M = self.K // 2
        self.phi = np.asarray(phi, dtype=float)          # length M
        self.in_band = (R_GRID >= r_lo) & (R_GRID < r_hi) & ANN
        self.n_points = int(self.in_band.sum())
        self.pts = np.argwhere(self.in_band)             # (n, 2)
        lookup = -np.ones((SIZE, SIZE), dtype=np.int64)
        lookup[self.pts[:, 0], self.pts[:, 1]] = np.arange(len(self.pts))
        partner = (-self.pts) % SIZE
        self.partner_idx = lookup[partner[:, 0], partner[:, 1]]
        assert (self.partner_idx >= 0).all(), "radial band must be closed under p -> -p"
        self.th = TH_GRID[self.in_band]
        self.order = np.argsort(self.th)
        self.th_sorted = self.th[self.order]
        # chip index of every in-band point at delta = 0
        self.chip0 = self._chip_index(self.th, 0.0)
        self.C, self.N = self._codebook(seed)
        self.Psi = self._carriers()                      # (B, n) complex
        self.norms = np.array([
            float((np.abs(self.C[j]) ** 2 * self.N).sum())
            for j in range(self.C.shape[0])
        ])
        # cumulative sums for fast G(delta)
        Psi_s = self.Psi[:, self.order]
        self.csum = np.concatenate(
            [np.zeros((B, 1), dtype=complex), np.cumsum(np.conj(Psi_s), axis=1)],
            axis=1,
        )

    # -- partition ---------------------------------------------------------- #
    def _chip_index(self, theta_deg, delta: float) -> np.ndarray:
        t = (np.asarray(theta_deg) - delta) % 360.0
        return np.searchsorted(self.edges[1:], t, side="right").astype(np.int64) % self.K

    def hermitian_error(self) -> float:
        """max |Psi(p) - conj(Psi(-p))| over the annulus (0 for valid carriers)."""
        return float(np.abs(self.Psi - np.conj(self.Psi[:, self.partner_idx])).max())

    # -- codebook ----------------------------------------------------------- #
    def _codebook(self, seed):
        rng = np.random.RandomState(seed)
        A, _ = np.linalg.qr(rng.randn(B, self.M).T)
        A = A[:, :B].T                                   # B x M, orthonormal rows
        N = np.array([
            float(((self.chip0 == j) | (self.chip0 == j + self.M)).sum())
            for j in range(self.M)
        ])
        C = A / np.sqrt(N)[None, :]                      # C diag(N) C^T = I
        return C, N

    def weighted_orth_error(self) -> float:
        G = self.C @ np.diag(self.N) @ self.C.T
        return float(np.abs(G - np.eye(self.C.shape[0])).max())

    # -- carriers ----------------------------------------------------------- #
    def _carriers(self):
        q = self.chip0 % self.M
        sign = np.where(self.chip0 < self.M, 1.0, -1.0)
        phase = np.exp(1j * self.phi[q] * sign)
        return self.C[:, q] * phase[None, :]

    # -- chip values and alias ---------------------------------------------- #
    def chip_values(self, msgs: np.ndarray) -> np.ndarray:
        """(K, n_msg) complex watermark chip values for each message."""
        u = self.C.T @ msgs                               # (M, n_msg)
        sign = np.where(np.arange(self.K) < self.M, 1.0, -1.0)
        phase = np.exp(1j * self.phi[np.arange(self.K) % self.M] * sign)
        return u[np.arange(self.K) % self.M, :] * phase[:, None]

    def set_phase(self, phi: np.ndarray) -> None:
        """Replace the phase mask and rebuild carriers / cumulative sums."""
        self.phi = np.asarray(phi, dtype=float)
        self.Psi = self._carriers()
        Psi_s = self.Psi[:, self.order]
        self.csum = np.concatenate(
            [np.zeros((self.C.shape[0], 1), dtype=complex),
             np.cumsum(np.conj(Psi_s), axis=1)],
            axis=1,
        )

    def G(self, delta: float) -> np.ndarray:
        """G[j, q] = sum over points p with chip(p, delta) == q of conj(Psi_j(p))."""
        edges = (self.edges + delta) % 360.0
        lo = np.searchsorted(self.th_sorted, edges[:-1], side="left")
        hi = np.searchsorted(self.th_sorted, edges[1:], side="left")
        n = len(self.th_sorted)
        wrap = edges[1:] <= edges[:-1]
        lo_v, hi_v = self.csum[:, lo], self.csum[:, hi]     # (B, K) each
        direct = hi_v - lo_v
        wrapped = (self.csum[:, n][:, None] - lo_v) + hi_v
        return np.where(wrap[None, :], wrapped, direct)

    def scores(self, msgs: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """(n_delta, n_msg) mean|ell| at each candidate alignment delta."""
        V = self.chip_values(msgs)                        # (K, n_msg)
        out = np.empty((len(deltas), msgs.shape[1]))
        for i, d in enumerate(deltas):
            ell = (self.G(float(d)) @ V) / self.norms[:, None]
            out[i] = np.abs(ell).mean(axis=0)
        return out


# --------------------------------------------------------------------------- #
# designs
# --------------------------------------------------------------------------- #
class Design:
    def __init__(self, layers, fusion="min", name="", capacity_bits=None,
                 diversity=True):
        self.layers = layers
        self.fusion = fusion
        self.name = name
        self.diversity = diversity
        self.capacity_bits = capacity_bits or sum(B for _ in layers)

    def alias(self, msgs, deltas, msgs2=None):
        msg_list = msgs2 if msgs2 is not None else [msgs] * len(self.layers)
        curves = [lay.scores(m, deltas) for lay, m in zip(self.layers, msg_list)]
        A = [c / c[0:1, :] for c in curves]
        if self.fusion == "min":
            return np.minimum.reduce(A)
        if self.fusion == "prod":
            return np.prod(np.stack(A), axis=0)
        return np.mean(np.stack(A), axis=0)


def summarise(design: Design, msgs, deltas, msgs2=None) -> dict:
    A = design.alias(msgs, deltas, msgs2)
    int_mask = (deltas >= 10.0) & (deltas <= 170.0)
    half_mask = (deltas >= 170.0) & (deltas <= 190.0)
    other_mask = (deltas > 190.0) & (deltas <= 350.0)
    A_int = A[int_mask].max(axis=0)
    A_half = A[half_mask].max(axis=0)
    A_other = A[other_mask].max(axis=0)
    at = lambda t: float(A[np.argmin(np.abs(deltas - t))].mean())
    out = {
        "name": design.name,
        "fusion": design.fusion,
        "capacity_bits": design.capacity_bits,
        "layers": [
            {
                "label": lay.label,
                "K": lay.K,
                "M": lay.M,
                "r_band": [lay.r_lo, lay.r_hi],
                "n_points": lay.n_points,
                "points_per_bit": lay.n_points / lay.C.shape[0],
                "orth_error": lay.weighted_orth_error(),
                "hermitian_error": lay.hermitian_error(),
                "phase": bool(np.any(np.abs(lay.phi) > 1e-9)),
            }
            for lay in design.layers
        ],
        "A(45)": at(45.0), "A(90)": at(90.0), "A(135)": at(135.0),
        "A(180)": at(180.0), "A(270)": at(270.0),
        "A_int_mean": float(A_int.mean()), "A_int_worst": float(A_int.max()),
        "A_half_mean": float(A_half.mean()), "A_half_worst": float(A_half.max()),
        "A_other_mean": float(A_other.mean()), "A_other_worst": float(A_other.max()),
        "sync_margin_mean": float(1.0 - A_int.mean()),
        "p_alias": {
            str(t): float(np.mean(A_int >= 1.0 - t)) for t in TAU
        },
        "clean_score_dev": float(np.abs(A[0] - 1.0).max()),
    }
    return out


# --------------------------------------------------------------------------- #
# noise robustness of the *synchroniser* (per-bit Gaussian model)
# --------------------------------------------------------------------------- #
def layer_ell(lay: Layer, msgs: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """(n_delta, B, n_msg) noiseless matched-filter outputs of one layer."""
    V = lay.chip_values(msgs)
    out = np.empty((len(deltas), msgs.shape[0], msgs.shape[1]), dtype=complex)
    for i, d in enumerate(deltas):
        out[i] = (lay.G(float(d)) @ V) / lay.norms[:, None]
    return out


def fuse(stack: np.ndarray, fusion: str) -> np.ndarray:
    """stack: (n_layer, n_delta, n_msg) -> (n_delta, n_msg)."""
    if fusion == "min":
        return stack.min(axis=0)
    if fusion == "prod":
        return stack.prod(axis=0)
    return stack.mean(axis=0)


def save_design(path: str, design: Design, meta: dict) -> None:
    """Persist a design (edges / code matrices / phase masks) for the pipeline."""
    arrs = {"meta": np.array(json.dumps(meta))}
    for i, lay in enumerate(design.layers):
        arrs[f"layer{i}_edges"] = lay.edges
        arrs[f"layer{i}_C"] = lay.C
        arrs[f"layer{i}_phi"] = lay.phi
        arrs[f"layer{i}_band"] = np.array([lay.r_lo, lay.r_hi])
        arrs[f"layer{i}_N"] = lay.N
    np.savez(path, **arrs)


def noise_sync_fail(design: Design, ell_layers, deltas, sigmas, scales,
                    n_trials: int, rng, tol_deg: float = 2.0,
                    nuisance: float = 0.0) -> list[dict]:
    """P(argmax outside valid set) for each (signal scale, per-bit noise sigma).

    scale mimics the channel attenuation of the true peak (1.0 cleaner than
    the real pipeline; 0.12 rot; 0.05 rot+noise), sigma is the per-bit noise
    std in units where the unattenuated true response is 1.
    """
    comb = 0.5 + 0.5 * np.cos(np.deg2rad(4.0 * deltas))       # 4-fold content nuisance
    rows = []
    for scale in scales:
        for sigma in sigmas:
            f180 = f360 = 0
            total = 0
            margin = []
            for _ in range(n_trials):
                stack = []
                for e in ell_layers:
                    noise = (rng.randn(*e.shape) + 1j * rng.randn(*e.shape))
                    noise /= np.sqrt(2.0)
                    stack.append(np.abs(scale * e + sigma * noise).mean(axis=1))
                A = fuse(np.stack(stack), design.fusion)
                if nuisance > 0.0:
                    A = A + nuisance * comb[:, None]
                A = A / A[0:1, :]
                pk = np.argmax(A, axis=0)
                ang = deltas[pk]
                d180 = np.abs((ang + 90.0) % 180.0 - 90.0)
                d360 = np.abs((ang + 180.0) % 360.0 - 180.0)
                f180 += int((d180 > tol_deg).sum())
                f360 += int((d360 > tol_deg).sum())
                total += A.shape[1]
                margin.append(float(np.mean(A[0] - A.max(axis=0))))
            rows.append({
                "scale": scale, "sigma": sigma,
                "sync_fail_mod180": f180 / total,
                "sync_fail_mod360": f360 / total,
                "margin_mean": float(np.mean(margin)),
            })
    return rows


# --------------------------------------------------------------------------- #
# design generators
# --------------------------------------------------------------------------- #
def single_layer(name, K, widths=None, phi=None, seed=0, r_lo=R1, r_hi=R2 + 1e-6):
    edges = uniform_edges(K) if widths is None else widths_to_edges(widths)
    M = (len(edges) - 1) // 2
    if phi is None:
        phi = np.zeros(M)
    return Design([Layer(r_lo, r_hi, edges, phi, seed, label=name)], name=name)


def two_layer(name, K1, K2, split, fusion="min", phi1=None, phi2=None, seed=0,
              widths1=None, widths2=None, diversity=True):
    e1 = uniform_edges(K1) if widths1 is None else widths_to_edges(widths1)
    e2 = uniform_edges(K2) if widths2 is None else widths_to_edges(widths2)
    p1 = np.zeros(K1 // 2) if phi1 is None else np.asarray(phi1, dtype=float)
    p2 = np.zeros(K2 // 2) if phi2 is None else np.asarray(phi2, dtype=float)
    l1 = Layer(R1, split, e1, p1, seed, label="inner")
    l2 = Layer(split, R2 + 1e-6, e2, p2, seed + 1, label="outer")
    return Design([l1, l2], fusion=fusion, name=name,
                  capacity_bits=(B if diversity else 2 * B), diversity=diversity)


# --------------------------------------------------------------------------- #
# experiment
# --------------------------------------------------------------------------- #
def main() -> None:
    global B
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--msg_b", type=int, default=B)
    ap.add_argument("--search_step", type=float, default=1.0)
    ap.add_argument("--search_msgs", type=int, default=64)
    ap.add_argument("--msg_limit", type=int, default=256,
                    help="cap on the number of messages evaluated per layer "
                         "(0 = enumerate all 2^B; needed for B > 8)")
    ap.add_argument("--only_anchors", action="store_true",
                    help="evaluate only the anchors/ablations (no search)")
    ap.add_argument("--full_step", type=float, default=0.5)
    ap.add_argument("--full_msgs", type=int, default=256)
    ap.add_argument("--random_widths", type=int, default=120)
    ap.add_argument("--quasi_widths", type=int, default=24)
    ap.add_argument("--phase_candidates", type=int, default=200)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--noise_sigmas", default="0,0.05,0.1,0.2,0.4,0.8")
    ap.add_argument("--noise_scales", default="1.0,0.12,0.05")
    ap.add_argument("--noise_trials", type=int, default=3)
    ap.add_argument("--noise_msgs", type=int, default=64)
    ap.add_argument("--nuisance", type=float, default=0.0,
                    help="optional 4-fold content-nuisance floor added to the "
                         "sync score (0 disables); the real pipeline has ~0.07")
    ap.add_argument("--capacity_mode", action="store_true", default=True)
    ap.add_argument("--refine_steps", type=int, default=1,
                    help="greedy phase-mask refinement passes on the best 2-layer design")
    ap.add_argument("--refine_tries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    B = args.msg_b
    rng = np.random.RandomState(args.seed)
    all_msgs = messages_matrix(B, limit=args.msg_limit, seed=args.seed)
    search_msgs = all_msgs[:, : min(args.search_msgs, all_msgs.shape[1])]
    full_msgs = all_msgs[:, : min(args.full_msgs, all_msgs.shape[1])]
    d_search = np.arange(0.0, 360.0 + 1e-9, args.search_step)
    d_full = np.arange(0.0, 360.0 + 1e-9, args.full_step)

    t0 = time.time()
    fast, full = [], []

    def evaluate_fast(design):
        row = summarise(design, search_msgs, d_search)
        row["_design"] = design
        fast.append(row)
        return row

    def rank(row):
        """Lower is better: interior aliases first, then the 180-deg partner."""
        return (row["A_int_worst"] + 0.5 * row["A_half_worst"]
                + 0.25 * row["A_other_worst"])

	# ---------------- anchors --------------------------------------------- #
    anchors = [
        single_layer("anchor_uniform_K32_real", 32),
        single_layer("anchor_uniform_K32_phase_rand", 32,
                     phi=rng.uniform(-np.pi, np.pi, 16)),
    ]
    if B <= 15:
        anchors.append(single_layer("anchor_uniform_K30_real", 30))
    for d in anchors:
        evaluate_fast(d)
    print(f"[anchors] done  ({time.time()-t0:.1f}s)")

    # ---------------- stage 1: single layer geometry ---------------------- #
    Ks = [] if args.only_anchors else [24, 26, 28, 30, 32, 34, 36, 38, 40, 44, 48, 56, 64]
    for K in Ks:
        if K // 2 < B:
            continue
        evaluate_fast(single_layer(f"uni_K{K}_real", K))
        w = aperiodic_widths(K // 2, "quasi", rng)
        evaluate_fast(single_layer(f"quasi_K{K}_real", K, widths=w))
    for i in range(0 if args.only_anchors else args.quasi_widths):
        w = aperiodic_widths(16, "quasi", rng)
        evaluate_fast(single_layer(f"quasi16_{i}_real", 32, widths=w))
    for i in range(0 if args.only_anchors else args.random_widths):
        w = aperiodic_widths(16, "rand", rng)
        if B > 16:
            continue
        evaluate_fast(single_layer(f"rand16_{i}_real", 32, widths=w))
    print(f"[stage 1 geometry] {len(fast)} designs  ({time.time()-t0:.1f}s)")

    # ---------------- stage 2: two radial layers -------------------------- #
    split_choices = {
        "eqwidth": 0.5 * (R1 + R2),
        "eqarea": float(np.sqrt(0.5 * (R1 ** 2 + R2 ** 2))),
        "eqlog": float(np.sqrt(R1 * R2)),
    }
    pair_choices = [(32, 30), (30, 32), (32, 34), (34, 32), (30, 36), (32, 32),
                    (36, 32), (34, 30), (30, 30)]
    for sname, split in split_choices.items():
        for K1, K2 in pair_choices:
            if min(K1, K2) // 2 < B:
                continue
            for fusion in ("min", "prod"):
                if args.only_anchors and sname != "eqarea":
                    continue
                evaluate_fast(two_layer(
                    f"2L_{sname}_K{K1}xK{K2}_{fusion}_real", K1, K2, split,
                    fusion=fusion))
    split = split_choices["eqarea"]
    for i in range(0 if args.only_anchors else args.quasi_widths):
        w1 = aperiodic_widths(16, "quasi", rng)
        w2 = aperiodic_widths(15, "quasi", rng)
        evaluate_fast(two_layer(f"2L_aper_{i}_min_real", 32, 30, split,
                                fusion="min", widths1=w1, widths2=w2))
    print(f"[stage 2 layers] {len(fast)} designs  ({time.time()-t0:.1f}s)")

    # ---------------- stage 3: phase on the best geometries --------------- #
    n_phase_each = 0 if args.only_anchors else max(1, args.phase_candidates // 2)
    for tag, K in [("uni32", 32), ("uni30", 30)]:
        if K // 2 < B:
            continue
        for c in range(n_phase_each):
            phi = rng.uniform(-np.pi, np.pi, K // 2)
            evaluate_fast(single_layer(f"phase_{tag}_{c}", K, phi=phi))
    # two-layer phase variants on the best pair
    phase_pair = (32, 34) if B > 15 else (32, 30)
    for c in range(0 if args.only_anchors else args.phase_candidates // 8):
        phi1 = rng.uniform(-np.pi, np.pi, phase_pair[0] // 2)
        phi2 = rng.uniform(-np.pi, np.pi, phase_pair[1] // 2)
        for fusion in ("min", "prod", "mean"):
            evaluate_fast(two_layer(f"2L_phase_{c}_{fusion}", *phase_pair,
                                    split_choices["eqarea"], fusion=fusion,
                                    phi1=phi1, phi2=phi2))
    print(f"[stage 3 phase] {len(fast)} designs  ({time.time()-t0:.1f}s)")

    # ---------------- stage 3b: greedy phase refinement -------------------- #
    refine_info = None
    if args.refine_steps > 0:
        base = min([r for r in fast if r["name"].startswith("2L_phase")], key=rank)
        design = copy.deepcopy(base["_design"])
        before = rank(base)
        for step in range(args.refine_steps):
            for lay in design.layers:
                for m in range(lay.M):
                    keep = lay.phi[m]
                    for _ in range(args.refine_tries):
                        cand = rng.uniform(-np.pi, np.pi)
                        trial = lay.phi.copy()
                        trial[m] = cand
                        lay.set_phase(trial)
                        score = rank(summarise(design, search_msgs, d_search))
                        if score < before - 1e-6:
                            before = score
                            keep = cand
                    trial = lay.phi.copy()
                    trial[m] = keep
                    lay.set_phase(trial)
        row = summarise(design, search_msgs, d_search)
        row["name"] = "refined_2L_phase_prod"
        row["_design"] = design
        fast.append(row)
        refine_info = {"start_from": base["name"], "rank_before": float(before),
                       "rank_after": float(rank(row)),
                       "A_int_before": base["A_int_worst"],
                       "A_int_after": row["A_int_worst"]}
        print(f"[stage 3b refine] {base['name']} -> A_int="
              f"{row['A_int_worst']:.3f} A_half={row['A_half_worst']:.3f} "
              f"({time.time()-t0:.1f}s)")

    # ---------------- full-resolution re-evaluation of the finalists ------- #
    # explicit ablation family: single layer real/phase, two layers real/phase,
    # and the naive two-layer design that shares one angular grid.
    split0 = float(np.sqrt(0.5 * (R1 ** 2 + R2 ** 2)))
    ablations = [
        single_layer("abl_1L_K32_real", 32),
        two_layer("abl_2L_K32xK32_real_min", 32, 32, split0, fusion="min"),
    ]
    if B <= 15:
        ablations += [
            single_layer("abl_1L_K30_real", 30),
            two_layer("abl_2L_K32xK30_real_min", 32, 30, split0, fusion="min"),
            two_layer("abl_2L_K32xK30_real_prod", 32, 30, split0, fusion="prod"),
        ]
    for d in ablations:
        d._row = summarise(d, full_msgs, d_full)
    cand_1L = [r for r in fast if r["name"].startswith("phase_uni32")]
    cand_2L = [r for r in fast if r["name"].startswith("2L_phase")]
    best_phase_1L = min(cand_1L, key=rank) if cand_1L else None
    best_phase_2L = min(cand_2L, key=rank) if cand_2L else None

    finalists = sorted(fast, key=rank)[: args.top_k]
    finalists += [r for r in fast if r["name"].startswith("anchor")]
    finalists += [r for r in (best_phase_1L, best_phase_2L) if r is not None]
    finalists += [{"name": d.name, "_design": d, "A_int_worst": d._row["A_int_worst"],
                   "A_half_worst": d._row["A_half_worst"],
                   "A_other_worst": d._row["A_other_worst"]} for d in ablations]
    seen = set()
    full_designs = []
    for row in finalists:
        if row["name"] in seen:
            continue
        seen.add(row["name"])
        design = row["_design"]
        out = summarise(design, full_msgs, d_full)
        out["name"] = row["name"]
        out["search_rank"] = rank(row)
        full.append(out)
        full_designs.append(design)

    # ---- noise robustness of the synchroniser ----------------------------- #
    sigmas = [float(x) for x in args.noise_sigmas.split(",")]
    scales = [float(x) for x in args.noise_scales.split(",")]
    noise_msgs = full_msgs[:, : min(args.noise_msgs, full_msgs.shape[1])]
    d_noise = np.arange(0.0, 360.0 + 1e-9, 1.0)
    for out, design in zip(full, full_designs):
        ells = [layer_ell(lay, noise_msgs, d_noise) for lay in design.layers]
        out["noise"] = noise_sync_fail(design, ells, d_noise, sigmas, scales,
                                       args.noise_trials, rng)
        if args.nuisance > 0.0:
            out["noise_with_nuisance"] = noise_sync_fail(
                design, ells, d_noise, sigmas, scales, args.noise_trials, rng,
                nuisance=args.nuisance)
        # capacity mode: the two layers carry independent payloads
        if args.capacity_mode and len(design.layers) > 1:
            perm = rng.permutation(full_msgs.shape[1])
            second = full_msgs[:, perm]
            A = design.alias(full_msgs, d_full, msgs2=[full_msgs, second])
            m = (d_full >= 10.0) & (d_full <= 170.0)
            out["capacity_mode_alias"] = {
                "A_int_worst": float(A[m].max()),
                "A_half_worst": float(A[(d_full >= 170) & (d_full <= 190)].max()),
                "capacity_bits": 2 * B,
            }
    order = sorted(range(len(full)),
                   key=lambda i: full[i]["A_int_worst"] + 0.5 * full[i]["A_half_worst"])
    full = [full[i] for i in order]
    print(f"[noise eval] {len(full)} finalists  ({time.time()-t0:.1f}s)")

    os.makedirs(args.out_dir, exist_ok=True)
    blob = {
        "config": vars(args),
        "baselines": {"R1": R1, "R2": R2, "size": SIZE, "B": B,
                      "n_annulus_points": int(ANN.sum())},
        "search_table": [{k: v for k, v in r.items() if k != "_design"}
                         for r in sorted(fast, key=rank)],
        "finalists": full,
        "refine": refine_info,
        "elapsed_sec": time.time() - t0,
    }
    out_file = os.path.join(args.out_dir, "shift_identifiable_search.json")
    with open(out_file, "w") as f:
        json.dump(blob, f, indent=2)
    # persist the best designs (and the current-pipeline anchor) for the pipeline
    order_idx = sorted(range(len(full_designs)),
                       key=lambda i: full[i]["A_int_worst"] + 0.5 * full[i]["A_half_worst"])
    saved = []
    for rank_i, idx in enumerate(order_idx[:3]):
        name = full[idx]["name"]
        path = os.path.join(args.out_dir, f"design_{rank_i}_{name}.npz")
        save_design(path, full_designs[idx], {"name": name,
                                              "A_int": full[idx]["A_int_worst"],
                                              "A_half": full[idx]["A_half_worst"],
                                              "fusion": full[idx]["fusion"],
                                              "capacity_bits": full[idx]["capacity_bits"]})
        saved.append(path)

    print()
    header = (f"{'design':<34}{'cap':>4}{'A45':>7}{'A90':>7}{'A135':>7}"
              f"{'A180':>7}{'A270':>7}{'Aint':>7}{'Ahalf':>7}{'p.05':>6}")
    print(header)
    print("-" * len(header))
    for r in full:
        print(f"{r['name']:<34}{r['capacity_bits']:>4}{r['A(45)']:>7.3f}{r['A(90)']:>7.3f}"
              f"{r['A(135)']:>7.3f}{r['A(180)']:>7.3f}{r['A(270)']:>7.3f}"
              f"{r['A_int_worst']:>7.3f}{r['A_half_worst']:>7.3f}"
              f"{r['p_alias']['0.05']:>6.2f}")
    print()
    print(f"searched {len(fast)} designs, {len(full)} re-evaluated at "
          f"{args.full_step} deg / {full_msgs.shape[1]} messages "
          f"in {time.time()-t0:.1f}s")
    print(f"saved -> {out_file}")


if __name__ == "__main__":
    main()
