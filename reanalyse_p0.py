"""Offline re-analysis of the P0 rows: mod-360 sync metrics + clustered CIs.

This is a **CPU-only, torch-free** re-analysis.  It reads the per-sample
``rows.jsonl`` files that the P0 runs already wrote and recomputes the two
quantities that an external source-code review showed were defined
inconsistently:

1. **mod-360 vs mod-180 synchronization.**  ``p0_common.is_synced`` defaults to
   ``period=180``, which assumes a 180-degree flip is an equivalent alignment --
   true for a *real* Hermitian carrier, which is exactly what SectorSync's
   complex phase mask breaks.  A hypothesis at ``true + 180`` is a genuine
   failure (it decodes the wrong bits) yet mod-180 scores it as 0 error.  We
   re-score every row on the full circle and report how many rows move from
   "synced" to "failed".

2. **Record-level vs image-clustered intervals.**  The rotation stacks contain
   8-27 rows per image and reuse the same images across sub-runs, so resampling
   *records* understates the uncertainty.  We report image-level block-bootstrap
   intervals alongside the record-level ones.

Nothing here needs the diffusion model, a GPU, or the original run scripts --
only numpy and the rows files.

    python reanalyse_p0.py --runs_dir runs --out_dir results/reanalysis
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

from p0_common import (align_error, align_error360, case_rotation_angle,
                       clustered_rate_ci, clustered_stat_ci, load_rows,
                       provenance, safe_print, wilson, write_json)

#: rows files to look for, and the columns that define a "cell" to group on
STAGES = {
    "controls": {
        "glob": "p0_controls/*/rows.jsonl",
        "group": ("case", "condition"),
        # sync is only meaningful for watermarked rows
        "keep_conditions": ("wm",),
    },
    "rotation": {
        "glob": "p0_rotation/*/rows.jsonl",
        "group": ("sigma",),
    },
    "operators": {
        "glob": "p0_operators/*/rows.jsonl",
        "group": ("attack_op", "decode_op"),
    },
    "pareto": {
        "glob": "pareto/rows.jsonl",
        "group": ("cell", "case"),
    },
    # NOTE: ``identity`` is deliberately excluded.  Its rows store
    # ``condition`` in {pos, null, unregistered} and, more importantly, the
    # ``true_angle`` field written before 2026-09-21 held the raw random draw
    # ``theta`` rather than the angle actually applied for literal tokens such
    # as ``rot45`` -- so any sync statistic recomputed from those rows would be
    # wrong.  The identity benchmark's own metrics (Id-Acc, margin) do not use
    # sync labels, so nothing is lost.  The script is fixed for future runs.
}

#: stages whose stored rows cannot be re-scored, with the reason (printed in the
#  report so a reader does not wonder why they are missing)
EXCLUDED = {
    "identity": "行内的 `true_angle` 在本轮修复前存的是随机 θ 而非实际施加角"
                "（`rot45`/`rot75` 这类字面角会失真），因此无法据此重算同步口径；"
                "其 Id-Acc / margin 不依赖同步标签，故不影响结论。",
}


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "--"
    if isinstance(v, float) and not np.isfinite(v):
        return "--"
    return f"{float(v):.{nd}f}"


def _ci(d: Optional[Dict], key: str = "ci") -> str:
    if not d or key not in d or d[key] is None:
        return "--"
    lo, hi = d[key]
    return f"[{lo:.4f}, {hi:.4f}]"


def _clus(d: Optional[Dict], label: str = "图有失锁") -> str:
    """Format a clustered interval, flagging the degenerate-bootstrap fallback."""
    if not d:
        return "--"
    txt = _ci(d)
    if d.get("ci_degenerate"):
        txt += "†"
    if "clusters_with_event" in d:
        txt += f" ({d['clusters_with_event']}/{d['n_clusters']} {label})"
    return txt


def analyse(rows: List[Dict], tol: float) -> Optional[Dict]:
    """Recompute sync statistics for one group of rows.

    Both ``mod180`` and ``mod360`` verdicts are derived from the stored
    ``est_angle`` / ``true_angle``, so rows written *before* the 360-degree
    helpers existed can still be re-scored -- that is the whole point of keeping
    the raw angles in every row.
    """
    rows = [r for r in rows if "est_angle" in r and "true_angle" in r]
    if not rows:
        return None
    groups = [r.get("image_id", i) for i, r in enumerate(rows)]
    est = np.asarray([float(r["est_angle"]) for r in rows])
    true = np.asarray([float(r["true_angle"]) for r in rows])
    err180 = np.asarray([align_error(e, t) for e, t in zip(est, true)])
    err360 = np.asarray([align_error360(e, t) for e, t in zip(est, true)])
    fail180 = (err180 > tol).astype(np.float64)
    fail360 = (err360 > tol).astype(np.float64)
    p180, lo180, hi180 = wilson(int(fail180.sum()), len(rows))
    p360, lo360, hi360 = wilson(int(fail360.sum()), len(rows))
    out = {
        "n": len(rows),
        "n_images": len({str(g) for g in groups}),
        # mod-180 (the convention the old reports used)
        "fail180_rate": p180, "fail180_wilson": [lo180, hi180],
        "fail180_k": int(fail180.sum()),
        # mod-360 (re-scored, full circle)
        "fail360_rate": p360, "fail360_wilson": [lo360, hi360],
        "fail360_k": int(fail360.sum()),
        # image-clustered intervals for both
        "fail180_clustered": clustered_rate_ci(fail180, groups),
        "fail360_clustered": clustered_rate_ci(fail360, groups),
        # how many rows change verdict, and where the failures sit
        "rows_flipped_to_fail": int(((fail180 == 0) & (fail360 == 1)).sum()),
        "err360_median": float(np.median(err360)),
        "err360_p90": float(np.percentile(err360, 90)),
        "err360_max": float(err360.max()),
        "err180_median": float(np.median(err180)),
        "err180_p90": float(np.percentile(err180, 90)),
        "n_antipodal": int(((err180 <= tol) & (err360 > 90.0)).sum()),
    }
    for key, name in (("bit_acc", "bit_acc"), ("perfect", "pmr")):
        vals = [r[key] for r in rows if key in r]
        if vals:
            vals = np.asarray(vals, dtype=np.float64)
            out[f"{name}_mean"] = float(vals.mean())
            out[f"{name}_clustered"] = clustered_stat_ci(vals, groups)
    return out


def group_key(row: Dict, cols) -> str:
    parts = []
    for c in cols:
        if c in row:
            parts.append(f"{c}={row[c]}")
    return "|".join(parts) if parts else "all"


def leak_table(rows: List[Dict], tol: float) -> Dict[str, Dict]:
    """Angle-lock rate per (case, negative-class).

    For a *watermarked* image the synchronizer is supposed to find the true
    angle.  For a **null** image (no watermark) rotated by the *same operator*,
    finding the true angle would mean the angle is readable from the attack
    itself -- the padding wedges, the interpolation footprint, or the black
    border -- rather than from the carrier.  So the null lock rate is the
    padding-independence test that does not depend on any particular fill rule.
    Wrong-key rows are reported alongside as a second negative class.
    """
    out: Dict[str, Dict] = {}
    for r in rows:
        if "est_angle" not in r or "true_angle" not in r:
            continue
        cond = str(r.get("condition", "?"))
        cls = "wrongkey" if cond.startswith("wrongkey") else cond
        key = f"{r.get('case', '?')}|{cls}"
        out.setdefault(key, []).append(r)
    res: Dict[str, Dict] = {}
    for key, sub in sorted(out.items()):
        groups = [r.get("image_id", i) for i, r in enumerate(sub)]
        err = np.asarray([align_error360(float(r["est_angle"]),
                                         float(r["true_angle"])) for r in sub])
        locked = (err <= tol).astype(np.float64)
        p, lo, hi = wilson(int(locked.sum()), len(sub))
        res[key] = {
            "n": len(sub),
            "n_images": len({str(g) for g in groups}),
            "lock_rate": p, "lock_wilson": [lo, hi],
            "lock_clustered": clustered_rate_ci(locked, groups),
            "err_median": float(np.median(err)),
            "err_p90": float(np.percentile(err, 90)),
        }
    return res


def analyse_main_table(path: str, tol: float) -> Optional[Dict]:
    """Re-score the unified-comparison run (``runs/paper_*/ours.json``).

    That file stores one row per (image, case) with ``wm_est_angle``, ``case``
    and the per-image ``theta``; the angle actually applied is
    ``case_rotation_angle(case, theta)``.  This is the flagship synchronization
    evidence (search on vs off), so it is worth checking on the full circle too.
    """
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    rows = blob.get("rows") or []
    pseudo = []
    for r in rows:
        if "wm_est_angle" not in r or "case" not in r:
            continue
        pseudo.append({
            "image_id": r.get("index", len(pseudo)),
            "case": r["case"],
            "true_angle": float(case_rotation_angle(r["case"],
                                                    float(r.get("theta", 0.0)))),
            "est_angle": float(r["wm_est_angle"]),
            "bit_acc": r.get("wm_bit_acc"),
            "perfect": r.get("wm_perfect"),
        })
    if not pseudo:
        return None
    buckets: Dict[str, List[Dict]] = {}
    for r in pseudo:
        buckets.setdefault(f"case={r['case']}", []).append(r)
    return {"overall": analyse(pseudo, tol),
            "cells": {k: analyse(v, tol) for k, v in sorted(buckets.items())}}


def resolve_path(arg: str, runs_dir: str) -> Optional[str]:
    """Resolve a rows path given as-is, relative to runs_dir, or to its parent."""
    cands = [arg] if os.path.isabs(arg) else [
        arg,
        os.path.join(runs_dir, arg),
        os.path.join(os.path.dirname(os.path.abspath(runs_dir)), arg),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_dir", default="results/reanalysis")
    ap.add_argument("--sync_tol", type=float, default=2.0)
    ap.add_argument("--report_path", default="results/P0_reanalysis_mod360.md")
    ap.add_argument("--leak_rows", default="p0_controls/step2_N50/rows.jsonl",
                    help="rows file whose null/wrong-key conditions are used for "
                         "the angle-lock (padding) test; resolved as given, then "
                         "relative to --runs_dir, then relative to its parent")
    ap.add_argument("--main_table_runs",
                    default="paper_ours/ours.json",
                    help="comma list of unified-comparison runs (runs/<x>/ours.json) "
                         "to re-score; resolved like --leak_rows")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary: Dict[str, Dict] = {}
    lines: List[str] = []

    for stage, spec in STAGES.items():
        pattern = os.path.join(args.runs_dir, spec["glob"])
        files = sorted(glob.glob(pattern))
        if not files:
            safe_print(f"[skip] {stage}: no rows at {pattern}")
            continue
        stage_out: Dict[str, Dict] = {}
        for path in files:
            rows = load_rows(path)
            keep = spec.get("keep_conditions")
            if keep:
                rows = [r for r in rows if r.get("condition") in keep]
            if not rows:
                continue
            tag = os.path.relpath(path, args.runs_dir)
            buckets: Dict[str, List[Dict]] = {}
            for r in rows:
                buckets.setdefault(group_key(r, spec["group"]), []).append(r)
            stage_out[tag] = {
                "overall": analyse(rows, args.sync_tol),
                "cells": {k: analyse(v, args.sync_tol)
                          for k, v in sorted(buckets.items())},
            }
            safe_print(f"[ok] {stage}: {tag} -> {len(rows)} rows, "
                       f"{len(buckets)} cells")
        if stage_out:
            summary[stage] = stage_out

    # ---- report ---------------------------------------------------------
    lines += ["# P0 行级重算：mod-360 同步判定 + 按图像聚类的区间", "",
              f"容差 `sync_tol = {args.sync_tol:g}°`；数据来源为各阶段已有的 "
              "`rows.jsonl`（**无 GPU、无扩散**，纯离线重算）。", "",
              "说明：`mod180` 是旧报告使用的口径（假设 180° 翻转等价）；"
              "`mod360` 是全圆口径。**判定同步应看 mod360**，"
              "因为复相位掩码正是为打破 180° 等价性而设计。"
              "`逐行` 区间把每条记录当独立样本，`按图像聚类` 才是可用于泛化断言的区间。",
              "",
              "† = 聚类 bootstrap 退化（没有或**全部**图像含失锁时，重采样得到同一个值），"
              "此时区间改用**簇级 Wilson**（含失锁的图像数 / 总图像数），偏保守但有信息量；"
              "原始退化区间可从 JSON 的 `ci_degenerate` 字段读出。", ""]
    if EXCLUDED:
        lines += ["**未纳入的阶段**：", ""]
        for st, why in EXCLUDED.items():
            lines.append(f"* `{st}`：{why}")
        lines.append("")

    for stage, stage_out in summary.items():
        lines += [f"## {stage}", ""]
        for tag, data in stage_out.items():
            o = data["overall"]
            if o is None:
                continue
            lines += [f"### `{tag}`", "",
                      f"记录 {o['n']} 条，来自 **{o['n_images']} 张图**。", "",
                      "| 口径 | 失锁数 | 失锁率 | 95%CI | 区间类型 |",
                      "|---|---:|---:|---|---|",
                      f"| mod180 | {o['fail180_k']} | {o['fail180_rate']:.4f} | "
                      f"[{o['fail180_wilson'][0]:.4f}, {o['fail180_wilson'][1]:.4f}] | 逐行 |",
                      f"| mod180 | {o['fail180_k']} | {o['fail180_rate']:.4f} | "
                      f"{_clus(o['fail180_clustered'])} | **按图像聚类** |",
                      f"| **mod360** | {o['fail360_k']} | {o['fail360_rate']:.4f} | "
                      f"[{o['fail360_wilson'][0]:.4f}, {o['fail360_wilson'][1]:.4f}] | 逐行 |",
                      f"| **mod360** | {o['fail360_k']} | {o['fail360_rate']:.4f} | "
                      f"{_clus(o['fail360_clustered'])} | **按图像聚类** |", ""]
            lines += [f"改判：{o['rows_flipped_to_fail']} 条从 mod180 的「成功」"
                      f"改为 mod360 的「失败」；其中真正的对径锁定 "
                      f"（err180 ≤ tol 且 err360 > 90°）有 **{o['n_antipodal']}** 条。",
                      "",
                      f"角误差：mod360 中位 {o['err360_median']:.2f}°、"
                      f"p90 {o['err360_p90']:.2f}°、max {o['err360_max']:.2f}°"
                      f"（mod180 中位 {o['err180_median']:.2f}°）。"]
            if "pmr_mean" in o:
                lines.append(f"PMR {o['pmr_mean']:.3f}（聚类 CI "
                             f"{_ci(o.get('pmr_clustered'))}）；"
                             f"BitAcc {o['bit_acc_mean']:.4f}（聚类 CI "
                             f"{_ci(o.get('bit_acc_clustered'))}）。")
            lines.append("")

            cells = {k: v for k, v in data["cells"].items() if v}
            if len(cells) > 1:
                lines += ["| cell | n | n_img | 失锁率 mod180(逐行) | "
                          "失锁率 mod360(逐行) | 失锁率 mod360(聚类 CI) | 改判 | 对径 |",
                          "|---|---:|---:|---:|---:|---|---:|---:|"]
                for k, c in cells.items():
                    lines.append(
                        f"| `{k}` | {c['n']} | {c['n_images']} | "
                        f"{c['fail180_rate']:.4f} | {c['fail360_rate']:.4f} | "
                        f"{c['fail360_rate']:.4f} {_clus(c['fail360_clustered'])} | "
                        f"{c['rows_flipped_to_fail']} | {c['n_antipodal']} |")
                lines.append("")

    # ---- angle-leak / padding test --------------------------------------
    leak = None
    leak_path = resolve_path(args.leak_rows, args.runs_dir)
    if leak_path is None:
        safe_print(f"[warn] leak rows not found: {args.leak_rows}")
    else:
        safe_print(f"[ok] leak test on {leak_path}")
        leak = leak_table(load_rows(leak_path), args.sync_tol)
        lines += ["", "---", "",
                  "## 附：角度锁定率 —— 「同步是否在读攻击算子本身」的检验", "",
                  f"数据：`{args.leak_rows}`。对所有条件（水印 / 无水印 / 错密钥）"
                  f"用**同一 mod360 口径**统计「估计角落在真实角 ±{args.sync_tol:g}° 内」"
                  "的比例。", "",
                  "读法：**水印图**应该锁定真实角；**无水印图**被同一个算子（同样的"
                  "零填充黑楔、同样的插值）旋转，如果它也能锁定真实角，"
                  "说明角度信息来自攻击算子而非载波。这条检验**不依赖任何填充"
                  "算子的几何假设**（它只要求攻击算子对 wm 与 null 一致），"
                  "因此它是「同步是否读黑边」的正面证据。", "",
                  "| case \\| 条件 | n | n_img | 锁定率 | 聚类 CI | 角误差 中位/p90 |",
                  "|---|---:|---:|---:|---|---|"]
        for key, v in leak.items():
            lines.append(f"| `{key}` | {v['n']} | {v['n_images']} | "
                         f"{v['lock_rate']:.4f} | "
                         f"{_clus(v['lock_clustered'], label='图锁定真角')} | "
                         f"{v['err_median']:.2f} / {v['err_p90']:.2f} |")
        lines += ["", "注意 `clean` / `jpeg25` 等格的 `true_angle = 0`，"
                  "其 null 锁定率只是「估计角碰巧落在 0° 附近」的比例，"
                  "**不构成泄漏证据**；有意义的只有真正施加了旋转的格"
                  "（`rot45` / `rot75` / `rot+noise0.05`）。", ""]

    # ---- flagship main-table runs ---------------------------------------
    main_results = []
    for spec in [s for s in args.main_table_runs.split(",") if s]:
        p = resolve_path(spec, args.runs_dir)
        if p is None:
            safe_print(f"[warn] main-table run not found: {spec}")
            continue
        res = analyse_main_table(p, args.sync_tol)
        if res:
            safe_print(f"[ok] main table: {p}")
            main_results.append((os.path.basename(os.path.dirname(p)), res))
    if main_results:
        lines += ["", "---", "",
                  "## 附：主对比表（同步开关消融）的 mod-360 复核", "",
                  "主表最常被引用的那组证据（同载波、同攻击，只切换解码端是否做角度搜索）"
                  "同样用 mod-360 与 mod-180 分别复核。", ""]
        for tag, data in main_results:
            o = data["overall"]
            lines += [f"### `{tag}`（{o['n']} 条记录 / {o['n_images']} 张图）", "",
                      "**改判 "
                      f"{o['rows_flipped_to_fail']} 条**，其中对径锁定 "
                      f"**{o['n_antipodal']}** 条。", "",
                      "| case | n | 失锁 mod180 | 失锁 **mod360** | 角误差360 中位/p90 | 聚类 CI (mod360) |",
                      "|---|---:|---:|---:|---|---|"]
            for k, c in data["cells"].items():
                if not c:
                    continue
                lines.append(f"| `{k.split('=', 1)[1]}` | {c['n']} | "
                             f"{c['fail180_k']} ({c['fail180_rate']:.3f}) | "
                             f"{c['fail360_k']} ({c['fail360_rate']:.3f}) | "
                             f"{c['err360_median']:.2f} / {c['err360_p90']:.2f} | "
                             f"{_clus(c['fail360_clustered'])} |")
            lines.append("")

    lines += ["## 产物", "",
              f"* 逐阶段 JSON：`{args.out_dir}/mod360_summary.json`",
              f"* 本报告：`{args.report_path}`", ""]

    write_json(os.path.join(args.out_dir, "mod360_summary.json"),
               {"config": vars(args), "provenance": provenance(),
                "summary": summary, "leak": leak})
    os.makedirs(os.path.dirname(os.path.abspath(args.report_path)),
                exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    safe_print(f"\n-> {args.report_path}")


if __name__ == "__main__":
    main()
