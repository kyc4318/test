"""Wilson / bootstrap confidence intervals for the N=50 pilot tables.

The draft's N=50 tables report point estimates only.  At N=50 an empirical FPR
has resolution 1/50 = 2%, so TPR@1%FPR is an ROC interpolation, and a PMR such
as 48/50 = 0.960 has a 95% Wilson interval of roughly [0.865, 0.989].  This
script recomputes every table entry with an interval and writes a companion
markdown file, without touching the raw runs.

    python add_confidence_intervals.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import OrderedDict

import numpy as np

LABEL = {
    "ours": "SectorSync (ours)",
    "gs256": "Gaussian Shading 256b",
    "gs8": "Gaussian Shading 8b",
    "sfw_hstr": "SFWMark HSTR",
    "sfw_hsqr": "SFWMark HSQR",
}
CASE_ORDER = [
    "clean", "jpeg25", "jpeg75", "noise0.05", "noise0.1", "blur4", "blur5",
    "median5", "bright6.0", "contrast0.5", "bm3d0.1", "resize0.9", "resize1.25",
    "rot15", "rot30", "rot45", "rot60", "rot75", "rot90",
    "rot+noise0.05", "rot75+noise0.05", "rot75+noise0.1", "rot75+jpeg75",
    "rot45+resize1.25", "rot30+blur4", "cc0.5", "cc0.7", "rc0.7",
    "vaeb3", "vaec3", "regen300", "regen600",
]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def auc(pos, neg):
    from sklearn.metrics import auc as _auc
    from sklearn.metrics import roc_curve

    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    fpr, tpr, _ = roc_curve(y, np.concatenate([neg, pos]))
    return float(_auc(fpr, tpr))


def boot_ci(fn, n: int, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.RandomState(seed)
    vals = [fn(rng.randint(0, n, n)) for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/paper_*/*.json")
    ap.add_argument("--out", default="results/paper_main_tables_ci.md")
    ap.add_argument("--json_out", default="results/paper_main_tables_ci.json")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    runs = OrderedDict()
    for path in sorted(glob.glob(args.runs)):
        with open(path) as f:
            blob = json.load(f)
        runs[blob.get("method", os.path.basename(path)[:-5])] = blob
    if not runs:
        print(f"no runs matched {args.runs!r}")
        return
    cases = []
    for b in runs.values():
        for c in b.get("cases", []):
            if c not in cases:
                cases.append(c)
    cases = [c for c in CASE_ORDER if c in cases] + [c for c in cases
                                                      if c not in CASE_ORDER]

    out = {}
    lines = ["# 主表（N=50 pilot）——带 95% 置信区间", "",
             "PMR 与失锁率用 Wilson 区间；BitAcc 与 AUC 用 image-level bootstrap"
             f"（{args.n_boot} 次重采样）。$N=50$ 时经验 FPR 分辨率为 $1/50=2\\%$，"
             "故 TPR@1%FPR 为 ROC 插值，TPR@$10^{-3}/10^{-6}$ 不可经验估计。", ""]

    for method, blob in runs.items():
        rows = {}
        lines += [f"## {LABEL.get(method, method)}", "",
                  "| case | n | BitAcc [95% CI] | PMR [95% CI] | AUC [95% CI] | "
                  "TPR@1%FPR |", "|---|---:|---|---|---|---:|"]
        for case in cases:
            sub = [r for r in blob["rows"] if r["case"] == case]
            if not sub:
                continue
            n = len(sub)
            det = blob["summary"].get(case, {}).get("detection", {})
            entry = {"n": n}
            # BitAcc / PMR: ours & GS use different field names
            ba = [r.get("wm_bit_acc", r.get("wm_bit_acc_native")) for r in sub]
            pm = [r.get("wm_perfect", r.get("wm_native_perfect")) for r in sub]
            if all(v is not None for v in ba):
                arr = np.asarray(ba, dtype=float)
                lo, hi = boot_ci(lambda idx: float(arr[idx].mean()), n, args.n_boot)
                entry["bit_acc"] = {"mean": float(arr.mean()), "ci": [lo, hi]}
            if all(v is not None for v in pm):
                k = int(np.sum(np.asarray(pm, dtype=float)))
                p, lo, hi = wilson(k, n)
                entry["pmr"] = {"mean": p, "ci": [lo, hi], "k": k}
            pos = np.asarray([r["wm_score"] for r in sub], dtype=float)
            neg = np.asarray([r["null_score"] for r in sub], dtype=float)
            if len(pos) and len(neg):
                a = auc(pos, neg)
                lo, hi = boot_ci(
                    lambda idx: auc(pos[idx], neg[idx]), n, args.n_boot)
                entry["auc"] = {"mean": a, "ci": [lo, hi]}
                entry["tpr_at_1fpr"] = det.get("tpr_at_fpr_0.01")
            if "wm_est_angle" in sub[0]:
                errs = np.asarray([
                    abs((r["wm_est_angle"] - _attack_angle(r) + 90) % 180 - 90)
                    for r in sub])
                k = int(np.sum(errs > 2.0))
                p, lo, hi = wilson(k, n)
                entry["sync_fail"] = {"mean": p, "ci": [lo, hi], "k": k}
            rows[case] = entry

            def fmt(d, nd=4):
                if not d:
                    return "--"
                return f"{d['mean']:.{nd}f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"

            lines.append(
                f"| {case} | {n} | {fmt(entry.get('bit_acc'))} | "
                f"{fmt(entry.get('pmr'), 3)} | {fmt(entry.get('auc'), 3)} | "
                f"{entry.get('tpr_at_1fpr', float('nan')):.3f} |")
        lines.append("")
        out[method] = rows

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n".join(lines[:60]))
    print(f"\n-> {args.out}")


def _attack_angle(row):
    import re

    m = re.search(r"rot(\d+(?:\.\d+)?)", row["case"])
    return float(m.group(1)) if m else float(row["theta"])


if __name__ == "__main__":
    main()
