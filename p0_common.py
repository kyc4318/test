"""Shared helpers for the P0 evidence-gate experiments.

Additive by construction: nothing here is imported by the canonical
``run_sync_search.py`` / ``swm`` core, and none of those files are modified.
The P0 suite answers the four questions the N=50 pilot cannot:

  controls   does "max over angles" inflate the null / wrong-key score, and is
             the FPR still controlled when the threshold comes from an
             independent calibration split?
  rotation   does the synchroniser hold on continuous and negative angles,
             including angles that fall between search-grid points?
  identity   can the method enter a key-identification table at all, and at
             what key space and decode cost?
  spectrum   where does the inversion error go when rotation+noise collapses
             the true peak (attenuation vs out-of-subspace leakage)?

Every P0 script writes rows with one common record schema::

    protocol_version, config_hash, image_id, prompt_id, seed, key_id,
    payload_bits, case, true_angle, est_angle, est_error_deg, scores_topk,
    decoded_bits, detection_score, runtime_sec

so that thresholds, ROC curves, angle-error statistics and confidence
intervals can all be recomputed offline from the raw JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

PROTOCOL_VERSION = "p0-2026-09-18"


def configure_stdio() -> None:
    """Let the reports print non-ASCII on consoles whose codec is not UTF-8.

    The GPU box is UTF-8, but a Windows console defaults to GBK and would raise
    inside ``print`` when a report line contains a Greek letter or an em dash.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


configure_stdio()


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover - only on exotic consoles
        import sys

        buf = getattr(sys.stdout, "buffer", None)
        if buf is None:
            return
        buf.write(text.encode("utf-8", "replace") + b"\n")


# --------------------------------------------------------------------------- #
# angles
# --------------------------------------------------------------------------- #
def wrap180(x: float) -> float:
    """Map an angle error into (-90, 90] -- the carrier's symmetry period."""
    return float((float(x) + 90.0) % 180.0 - 90.0)


def wrap360(x: float) -> float:
    """Map an angle error into (-180, 180] -- no symmetry assumed."""
    return float((float(x) + 180.0) % 360.0 - 180.0)


def align_error(est: float, true: float, period: float = 180.0) -> float:
    """Minimal alignment error modulo the carrier symmetry period.

    .. warning::
       The default ``period=180`` assumes the carrier is **real-valued**, i.e.
       that a 180-degree flip is an equivalent alignment (Proposition 2 of the
       manuscript).  SectorSync's complex phase mask exists precisely to break
       that equivalence, so for this carrier the mod-180 convention *understates*
       failure: an estimate at ``true + 180`` is scored 0 error while decoding
       the wrong bits.  Use :func:`align_error360` for pass/fail decisions and
       keep mod-180 only to stay comparable with older numbers.
    """
    d = (float(est) - float(true)) % period
    return float(min(d, period - d))


def align_error360(est: float, true: float) -> float:
    """Full-circle alignment error ``|wrap360(est - true)|`` in [0, 180]."""
    return abs(wrap360(float(est) - float(true)))


def signed_error(est: float, true: float) -> float:
    return wrap180(float(est) - float(true))


def signed_error360(est: float, true: float) -> float:
    return wrap360(float(est) - float(true))


def is_synced(est: float, true: float, tol: float = 2.0,
              period: float = 180.0) -> float:
    """1.0 when the estimate is within ``tol`` degrees, in the given period."""
    return float(align_error(est, true, period=period) <= tol)


def is_synced360(est: float, true: float, tol: float = 2.0) -> float:
    """1.0 when the estimate is within ``tol`` degrees on the *full* circle."""
    return float(align_error360(est, true) <= tol)


def case_rotation_angle(case: str, theta: float) -> float:
    """The rotation (degrees) that ``apply_case`` actually applies.

    ``rot`` (bare) uses the per-sample ``theta``; ``rot75`` style tokens use
    their literal angle; every other token applies no rotation at all.
    """
    angle = 0.0
    for token in case.split("+"):
        if token == "rot":
            angle += float(theta)
            continue
        m = re.fullmatch(r"rot(-?\d+(?:\.\d+)?)", token)
        if m:
            angle += float(m.group(1))
    return angle


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(p), float(max(0.0, c - h)), float(min(1.0, c + h))


def bootstrap_ci(values: Sequence[float], stat=np.mean, n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05):
    """Image-level bootstrap interval for any scalar statistic."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.size == 1:
        v = float(stat(arr))
        return {"mean": v, "ci": [v, v], "n": 1}
    rng = np.random.RandomState(seed)
    n = arr.size
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        vals[b] = float(stat(arr[rng.randint(0, n, n)]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(stat(arr)), "ci": [float(lo), float(hi)], "n": int(n)}


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float],
                        n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap interval for mean(a - b) on paired per-image values."""
    x = np.asarray(list(a), dtype=np.float64)
    y = np.asarray(list(b), dtype=np.float64)
    n = min(x.size, y.size)
    if n == 0:
        return None
    d = x[:n] - y[:n]
    rng = np.random.RandomState(seed)
    vals = np.array([float(d[rng.randint(0, n, n)].mean()) for _ in range(n_boot)])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(d.mean()), "ci": [float(lo), float(hi)], "n": int(n)}


def _cluster_groups(groups: Sequence) -> List[np.ndarray]:
    """Map a per-record cluster id to the record indices of each cluster."""
    buckets: Dict[str, List[int]] = {}
    for idx, g in enumerate(groups):
        buckets.setdefault(str(g), []).append(idx)
    return [np.asarray(v, dtype=np.int64) for v in buckets.values()]


def clustered_stat_ci(values: Sequence[float], groups: Sequence,
                      stat=np.mean, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> Dict:
    """Cluster (image-level) bootstrap for any scalar statistic.

    ``bootstrap_ci`` resamples *records*.  When records are repeated
    measurements of the same image -- the rotation suites store 8-27 records per
    image, and the same 30 images are reused across sub-runs -- that treats them
    as independent draws and reports intervals that are far too narrow.  Here the
    resampling unit is the cluster (normally ``image_id``): a cluster drawn into
    a bootstrap replicate contributes *all* of its records.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return None
    clusters = _cluster_groups(groups)
    rng = np.random.RandomState(seed)
    n_c = len(clusters)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.randint(0, n_c, n_c)
        idx = np.concatenate([clusters[j] for j in pick])
        vals[b] = float(stat(arr[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(stat(arr)), "ci": [float(lo), float(hi)],
            "n": int(arr.size), "n_clusters": int(n_c)}


def clustered_rate_ci(flags: Sequence[float], groups: Sequence,
                      n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> Dict:
    """Cluster bootstrap for a 0/1 rate (e.g. the sync-failure rate).

    Same rationale as :func:`clustered_stat_ci`: resampling records for a rate
    that is only replicated across ~30 images understates the uncertainty by
    roughly the square root of the records-per-image ratio.

    Degenerate case: when *no* cluster (or every cluster) contains an event,
    every bootstrap replicate is identical and the interval collapses to a
    point -- which reads as "zero failures, zero uncertainty".  In that case the
    reported interval falls back to a Wilson interval at the **cluster** level
    (number of clusters containing at least one event out of the number of
    clusters), which is conservative but informative.  ``ci_degenerate`` and
    ``wilson_on_clusters`` are recorded so a reader can tell which happened.
    """
    arr = (np.asarray(list(flags), dtype=np.float64) > 0.5).astype(np.float64)
    if arr.size == 0:
        return None
    res = clustered_stat_ci(arr, groups, stat=np.mean, n_boot=n_boot,
                            seed=seed, alpha=alpha)
    res["k"] = int(arr.sum())
    clusters = _cluster_groups(groups)
    per_cluster = np.asarray([1.0 if arr[idx].max() > 0 else 0.0
                              for idx in clusters], dtype=np.float64)
    kc, nc = int(per_cluster.sum()), len(clusters)
    _, lo_c, hi_c = wilson(kc, nc)
    res["clusters_with_event"] = kc
    res["wilson_on_clusters"] = [lo_c, hi_c]
    res["ci_degenerate"] = bool(res["ci"][0] == res["ci"][1])
    if res["ci_degenerate"]:
        res["ci"] = [lo_c, hi_c]
    return res


def calibrate_threshold(neg: Sequence[float], fpr: float, eps: float = 1e-12):
    """Threshold tau with P(neg > tau) <= fpr, higher score = watermarked.

    Uses the conformal order statistic, so the condition holds by construction
    on the calibration split and does not depend on a distributional model.
    ``k = floor(fpr * n)`` nulls are allowed to exceed tau and the threshold is
    placed just above the (n-k)-th largest score, which keeps the largest
    achievable TPR subject to that constraint.  At ``n=50, fpr=0.01`` this is
    ``k=0``, i.e. tau = max(null) and a realised FPR of exactly 0/50.
    """
    arr = np.sort(np.asarray(list(neg), dtype=np.float64))
    n = arr.size
    if n == 0:
        return float("inf")
    k = max(0, min(int(math.floor(fpr * n)), n - 1))
    return float(arr[n - k - 1]) + eps


def rate_above(values: Sequence[float], tau: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float((arr > tau).mean()) if arr.size else float("nan")


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Rank-based ROC-AUC (identical to sklearn's tie-averaged value).

    Implemented directly so the P0 controls do not depend on scikit-learn.
    """
    p = np.asarray(list(pos), dtype=np.float64)
    n = np.asarray(list(neg), dtype=np.float64)
    if p.size == 0 or n.size == 0:
        return float("nan")
    all_scores = np.concatenate([p, n])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty(all_scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, all_scores.size + 1, dtype=np.float64)
    sorted_scores = all_scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = float(ranks[:p.size].sum())
    return (r_pos - p.size * (p.size + 1) / 2.0) / (p.size * n.size)


# --------------------------------------------------------------------------- #
# provenance / IO
# --------------------------------------------------------------------------- #
def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def config_hash(obj) -> str:
    blob = json.dumps(to_jsonable(obj), sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def provenance(extra: Optional[Dict] = None) -> Dict:
    import platform
    import sys

    info = {
        "protocol_version": PROTOCOL_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else None)
    except Exception:  # pragma: no cover - torch missing on a CPU-only box
        pass
    try:
        import subprocess

        root = os.path.dirname(os.path.abspath(__file__))
        info["git_commit"] = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    if extra:
        info.update(to_jsonable(extra))
    return info


class ResumeLog:
    """Append-only JSONL writer with resume support.

    Each row must carry a ``uid`` (or ``image_id``/``case``/``method``).  On
    restart the existing file is scanned and finished uids are skipped, so a
    long GPU job can be interrupted without losing work.

    Pass ``fingerprint=run_meta`` to get a guard against the failure mode this
    feature introduces: a *smoke* run and a *full* run pointed at the same
    ``--out_dir`` share uids, so the resume logic silently keeps the smoke rows
    (with the smoke configuration) and skips those uids in the later run.  The
    guard stores a hash of the stable part of the config next to the rows and
    refuses to resume when it differs.
    """

    #: keys whose values legitimately change between runs of the same config
    VOLATILE_KEYS = ("provenance", "created", "timestamp", "elapsed_sec",
                     "runtime_sec")

    def __init__(self, path: str, keys: Sequence[str] = ("uid",),
                 fingerprint: Optional[Dict] = None):
        self.path = path
        self.keys = tuple(keys)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._guard_fingerprint(path, fingerprint)
        self.done = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.done.add(tuple(str(rec.get(k)) for k in self.keys))
        self._fh = open(path, "a", encoding="utf-8")

    @classmethod
    def _stable(cls, obj):
        """Drop volatile keys so a restart of the same config hashes identically."""
        if isinstance(obj, dict):
            return {k: cls._stable(v) for k, v in obj.items()
                    if k not in cls.VOLATILE_KEYS}
        if isinstance(obj, (list, tuple)):
            return [cls._stable(v) for v in obj]
        return obj

    @classmethod
    def _guard_fingerprint(cls, path: str, fingerprint: Optional[Dict]) -> None:
        """Fail loudly when resuming a run with a different configuration."""
        if fingerprint is None:
            return
        side = path + ".fingerprint.json"
        cur = config_hash(cls._stable(fingerprint))
        has_rows = os.path.exists(path) and os.path.getsize(path) > 0
        prev = None
        if os.path.exists(side):
            try:
                with open(side, encoding="utf-8") as f:
                    prev = json.load(f).get("fingerprint")
            except Exception:
                prev = None
        if prev is None and has_rows:
            safe_print(
                f"[ResumeLog] WARNING: {path} already contains rows but has no "
                f"fingerprint sidecar, so the configuration cannot be verified. "
                f"Use a fresh --out_dir if the existing rows may come from "
                f"different parameters.")
        elif prev is not None and prev != cur:
            raise SystemExit(
                "[ResumeLog] refusing to resume:\n"
                f"  rows        : {path}\n"
                f"  written with: fingerprint {prev}\n"
                f"  current run : fingerprint {cur}\n"
                "This guards against mixing a smoke run and a full run in one "
                "--out_dir (the uids collide and the older rows are kept). "
                "Use a different --out_dir, or delete the rows file to redo it.")
        try:
            write_json(side, {"fingerprint": cur,
                              "config": to_jsonable(cls._stable(fingerprint))})
        except Exception as exc:  # pragma: no cover - never block a run on this
            safe_print(f"[ResumeLog] WARNING: could not write {side} ({exc})")

    def has(self, row: Dict) -> bool:
        return tuple(str(row.get(k)) for k in self.keys) in self.done

    def append(self, row: Dict) -> None:
        key = tuple(str(row.get(k)) for k in self.keys)
        self.done.add(key)
        self._fh.write(json.dumps(to_jsonable(row)) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_rows(path: str) -> List[Dict]:
    """Read a JSONL row file (tolerates a whole-run JSON blob too)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def make_config(args, device: str):
    from configs import SectorConfig

    return SectorConfig(
        model_id=args.model_id,
        dataset=args.dataset,
        num_inference_steps=args.steps,
        test_num_inference_steps=args.steps,
        device=device,
    )


def build_pipe(cfg, device: str):
    from run_exp import load_pipeline

    return load_pipeline(cfg, device)


def resolve_device(arg: str = "cuda") -> str:
    import torch

    return arg if (arg == "cpu" or torch.cuda.is_available()) else "cpu"


# --------------------------------------------------------------------------- #
# ours (SectorSync) -- same decode path as run_paper_compare.OursMethod
# --------------------------------------------------------------------------- #
class OursCore:
    """Thin wrapper that exposes the *landscape* the main runner hides.

    ``run_paper_compare.OursMethod`` returns only the arg-max angle and the
    bits there.  The P0 controls need every angle's score, the per-angle
    matched-filter output, and the ability to decode the same spectrum with a
    second (wrong) codebook.
    """

    def __init__(self, design: str, device: str = "cpu",
                 energy_per_point: float = 1e4, size: int = 64,
                 channel: int = 3):
        from swm.dual_layer import DualLayerWatermarker

        self.design = design
        self.device = device
        self.channel = channel
        self.energy_per_point = float(energy_per_point)
        self.dl = DualLayerWatermarker(design, size=size, channel=channel,
                                       device=device,
                                       energy_per_point=self.energy_per_point)
        self.n_bits = int(self.dl.layers[0]["spec"].C.shape[0])
        self.specs = [lay["spec"] for lay in self.dl.layers]

    # -- encode --------------------------------------------------------- #
    def embed(self, z_no, bits):
        return self.dl.embed(z_no, np.asarray(bits, dtype=np.float32))

    # -- decode --------------------------------------------------------- #
    def ell_at(self, z_hat, angle: float) -> np.ndarray:
        """Per-layer matched-filter output after anti-rotating by ``angle``."""
        from run_paper_compare import rotate_latent

        ells = self.dl.decode_layers(rotate_latent(z_hat, -float(angle)))[0]
        return np.stack([e.detach().cpu().numpy().astype(np.float64)
                         for e in ells])

    def ell_mean(self, z_hat, angle: float) -> np.ndarray:
        """Layer-averaged soft decisions (what the main table decodes)."""
        return self.ell_at(z_hat, angle).mean(axis=0)

    def score_at(self, z_hat, angle: float) -> float:
        """Product fusion of the per-layer mean|ell| scores."""
        from run_paper_compare import rotate_latent

        _, sc = self.dl.decode_layers(rotate_latent(z_hat, -float(angle)))
        out = 1.0
        for s in sc:
            out *= float(s)
        return out

    def landscape(self, z_hat, angles: Sequence[float]) -> np.ndarray:
        return np.asarray([self.score_at(z_hat, a) for a in angles],
                          dtype=np.float64)

    def decode(self, z_hat, angles: Sequence[float], topk: int = 5) -> Dict:
        angles = np.asarray(angles, dtype=np.float64)
        S = self.landscape(z_hat, angles)
        order = np.argsort(-S)[:max(1, int(topk))]
        best = int(np.argmax(S))
        est = float(angles[best])
        ell = self.ell_mean(z_hat, est)
        return {
            "est_angle": est,
            "detection_score": float(S[best]),
            "scores_topk": [{"angle": float(angles[i]), "score": float(S[i])}
                            for i in order],
            "ell": ell,
            "decoded_bits": (ell > 0).astype(int).tolist(),
        }


def search_grid(step: float, lo: float = 0.0, hi: float = 360.0) -> np.ndarray:
    n = int(round((hi - lo) / float(step)))
    return lo + float(step) * np.arange(n, dtype=np.float64)


# --------------------------------------------------------------------------- #
# image-level geometric ops (P0 controls need variants of the rotation token)
# --------------------------------------------------------------------------- #
def rotate_image(img, deg: float, resample: str = "bilinear", fill: int = 0):
    from PIL import Image

    mode = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST}[resample]
    return img.rotate(float(deg), resample=mode, fillcolor=tuple([fill] * 3))


def inscribed_crop_after_rotation(img, deg: float):
    """Rotate, then crop the largest inscribed square and resize back.

    Removes the zero-filled wedges that a naive ``rotate`` leaves in the
    corners, which is the standard way to test whether the synchroniser is
    reading the rotation off the padding rather than off the carrier.
    """
    from PIL import Image

    w, h = img.size
    rad = math.radians(abs(float(deg)) % 90.0)
    scale = math.cos(rad) + math.sin(rad)
    side = int(round(min(w, h) / max(scale, 1e-6)))
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize(
        (w, h), Image.LANCZOS)


def add_gaussian_noise(img, sigma: float, seed: int):
    if sigma <= 0:
        return img
    from PIL import Image

    rng = np.random.RandomState(seed)
    arr = np.asarray(img).astype(np.float64)
    arr = arr + rng.normal(0.0, float(sigma) * 255.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def bits_from_rng(rng: np.random.RandomState, n_bits: int) -> np.ndarray:
    return rng.randint(0, 2, int(n_bits)).astype(np.float32) * 2.0 - 1.0


def bit_acc(true_bits, pred_bits) -> float:
    t = np.asarray(true_bits).reshape(-1)
    p = np.asarray(pred_bits).reshape(-1)[:t.size]
    return float((t == p).mean())


def perfect_match(true_bits, pred_bits) -> float:
    t = np.asarray(true_bits).reshape(-1)
    p = np.asarray(pred_bits).reshape(-1)[:t.size]
    return float(np.all(t == p))


@dataclass
class SplitPlan:
    """Disjoint calibration / test index ranges.

    The P0 gate requires the detection threshold to come from images the test
    statistic never touches; contiguous disjoint ranges keep that auditable.
    """

    start: int = 0
    calib_n: int = 50
    test_n: int = 50
    calib_offset: int = 0
    test_offset: int = 1000

    def indices(self, split: str) -> np.ndarray:
        if split == "calib":
            return np.arange(self.calib_offset + self.start,
                             self.calib_offset + self.start + self.calib_n)
        if split == "test":
            return np.arange(self.test_offset + self.start,
                             self.test_offset + self.start + self.test_n)
        raise ValueError(f"unknown split {split!r}")

    def label(self, i: int) -> str:
        if i in set(self.indices("calib").tolist()):
            return "calib"
        if i in set(self.indices("test").tolist()):
            return "test"
        return "other"

    def as_dict(self) -> Dict:
        return {"start": self.start, "calib_n": self.calib_n,
                "test_n": self.test_n, "calib_offset": self.calib_offset,
                "test_offset": self.test_offset,
                "calib_ids": self.indices("calib").tolist(),
                "test_ids": self.indices("test").tolist()}
