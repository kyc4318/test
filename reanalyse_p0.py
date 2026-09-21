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
                       clustered_paired_diff, clustered_rate_ci,
                       clustered_stat_ci, load_rows, provenance, safe_print,
                       wilson, write_json)

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


def _rate_cols(d: Optional[Dict]) -> tuple:
    """Two columns for a rate: the record-level and the image-level estimand.

    They are *different quantities* and must never share a cell (a point estimate
    for one with the interval of the other is meaningless).  ``†`` marks a
    record-level cluster bootstrap that degenerated to a point.
    """
    if not d:
        return "--", "--"
    rec = (f"{d['rate_record']:.4f} "
           f"[{d['ci_record'][0]:.4f}, {d['ci_record'][1]:.4f}]")
    if d.get("ci_record_degenerate"):
        rec += "†"
    img = (f"{d['n_images_with_event']}/{d['n_clusters']} "
           f"= {d['rate_image']:.4f} "
           f"[{d['ci_image'][0]:.4f}, {d['ci_image'][1]:.4f}]")
    return rec, img


def _pair_keys(rows: List[Dict], extra: Sequence[str] = ()) -> Dict:
    """Index rows by the key that identifies a paired replicate."""
    out = {}
    for r in rows:
        k = (str(r.get("image_id")),
             f"{float(r.get('true_angle', 0.0)):.6f}",
             f"{r.get('sigma', '')}") + tuple(str(r.get(c, "")) for c in extra)
        out[k] = r
    return out


def _pci(d: Optional[Dict]) -> str:
    """Format a paired difference with its clustered interval.

    ``†`` marks a degenerate interval (every pair identical, so the bootstrap
    returns a point): that is "no evidence of a difference", not "proven equal".
    """
    if not d:
        return "--"
    txt = f"{d['mean']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"
    if d.get("ci_degenerate"):
        txt += "†"
    return txt


def _row_flags(r: Dict, tol: float):
    """(bit_acc, fail360) for one row, or None when it is not re-scorable."""
    if "est_angle" not in r or "true_angle" not in r:
        return None
    fail = float(align_error360(float(r["est_angle"]),
                               float(r["true_angle"])) > tol)
    ba = r.get("bit_acc")
    return (float(ba) if ba is not None else float("nan"), fail)


def paired_vs_reference(rows: List[Dict], tol: float, cell_cols=("attack_op",
                                                                 "decode_op"),
                        ref: Optional[tuple] = None) -> List[Dict]:
    """Paired difference of every cell against the reference cell.

    Records are paired on (image, true angle, sigma, *cell_cols* other than the
    varied ones) -- i.e. the same image under the same attack and angle, decoded
    two ways -- and the difference is bootstrapped over images.  Two intervals
    overlapping is *not* a test; a paired interval is.
    """
    cells = sorted({tuple(str(r.get(c, "")) for c in cell_cols) for r in rows})
    if not cells:
        return []
    if ref is None:
        ref = cells[0]
    index = {}
    for r in rows:
        fl = _row_flags(r, tol)
        if fl is None:
            continue
        index.setdefault(tuple(str(r.get(c, "")) for c in cell_cols), {})
        key = _pair_keys([r])
        index[tuple(str(r.get(c, "")) for c in cell_cols)].update(key)
    if ref not in index:
        return []
    out = []
    for cell in cells:
        if cell == ref or cell not in index:
            continue
        common = sorted(set(index[ref]) & set(index[cell]))
        if not common:
            continue
        groups = [k[0] for k in common]
        cell_flags = np.asarray([_row_flags(index[cell][k], tol) for k in
                                 common], dtype=np.float64)
        ref_flags = np.asarray([_row_flags(index[ref][k], tol) for k in
                                common], dtype=np.float64)
        zeros = np.zeros(len(common), dtype=np.float64)
        out.append({
            "cell": "|".join(cell), "ref": "|".join(ref),
            "n_pairs": len(common),
            "bitacc_cell": float(cell_flags[:, 0].mean()),
            "bitacc_ref": float(ref_flags[:, 0].mean()),
            "fail_cell": float(cell_flags[:, 1].mean()),
            "fail_ref": float(ref_flags[:, 1].mean()),
            "d_bitacc": clustered_paired_diff(cell_flags[:, 0],
                                              ref_flags[:, 0], groups),
            "d_fail": clustered_paired_diff(cell_flags[:, 1],
                                            ref_flags[:, 1], groups),
        })
    return out


def paired_b8_vs_b16(rows: List[Dict], tol: float) -> List[Dict]:
    """Paired B=16 minus B=8 comparison inside the capacity/energy Pareto run.

    The run reuses the same image / eta / attack across designs, so the pair key
    is (eta, case, image).  PMR is reported next to BitAcc because PMR is
    inherently stricter at B=16 (``p^16`` vs ``p^8``) and therefore cannot on its
    own support a claim about synchronization or per-bit reliability.
    """
    out = []
    etas = sorted({str(r.get("eta", "")) for r in rows})
    cases = sorted({str(r.get("case", "")) for r in rows})
    for eta in etas:
        for case in cases:
            sub = [r for r in rows
                   if str(r.get("eta", "")) == eta and str(r.get("case", "")) == case]
            by_bits = {}
            for r in sub:
                fl = _row_flags(r, tol)
                if fl is None:
                    continue
                b = int(r.get("n_bits", 0))
                by_bits.setdefault(b, {})[
                    str(r.get("image_id"))] = (fl[0], fl[1], r.get("perfect"))
            if 8 not in by_bits or 16 not in by_bits:
                continue
            common = sorted(set(by_bits[8]) & set(by_bits[16]))
            if not common:
                continue
            ba16 = np.asarray([by_bits[16][k][0] for k in common])
            ba8 = np.asarray([by_bits[8][k][0] for k in common])
            fl16 = np.asarray([by_bits[16][k][1] for k in common])
            fl8 = np.asarray([by_bits[8][k][1] for k in common])
            pm16 = np.asarray([float(by_bits[16][k][2] or 0) for k in common])
            pm8 = np.asarray([float(by_bits[8][k][2] or 0) for k in common])
            out.append({
                "eta": eta, "case": case, "n_pairs": len(common),
                "bitacc_8": float(ba8.mean()), "bitacc_16": float(ba16.mean()),
                "pmr_8": float(pm8.mean()), "pmr_16": float(pm16.mean()),
                "fail_8": float(fl8.mean()), "fail_16": float(fl16.mean()),
                # BER is just 1 - BitAcc; quoted explicitly because PMR is
                # inherently stricter at B=16 (p^16 vs p^8) and must not carry a
                # capacity claim on its own.
                "ber_8": float(1.0 - ba8.mean()),
                "ber_16": float(1.0 - ba16.mean()),
                "d_bitacc": clustered_paired_diff(ba16, ba8, common),
                "d_fail": clustered_paired_diff(fl16, fl8, common),
                "d_pmr": clustered_paired_diff(pm16, pm8, common),
            })
    return out


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
                      "两个**互不相同的统计量**分列（不可混用）：记录级 = 失败的"
                      "条件数 / 全部条件数；图像级 = 至少失败一次的图像数 / 图像数。", "",
                      "| 口径 | 记录级失锁率 [95%CI，按图像聚类] | "
                      "**图像级**失锁（≥1 次失败）[Wilson 95%CI] |",
                      "|---|---|---|"]
            for tagi, key in (("mod180", "fail180"), ("**mod360**", "fail360")):
                rec, img = _rate_cols(o[f"{key}_clustered"])
                lines.append(f"| {tagi} | {rec} | {img} |")
            lines += ["",
                      f"逐行（记录级、非聚类）区间仅作对照：mod180 "
                      f"[{o['fail180_wilson'][0]:.4f}, {o['fail180_wilson'][1]:.4f}]、"
                      f"mod360 [{o['fail360_wilson'][0]:.4f}, "
                      f"{o['fail360_wilson'][1]:.4f}]。", ""]
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
                          "失锁率 mod360(逐行) | mod360 记录级 [聚类 CI] | "
                          "mod360 图像级 | 改判 | 对径 |",
                          "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
                for k, c in cells.items():
                    rec, img = _rate_cols(c["fail360_clustered"])
                    lines.append(
                        f"| `{k}` | {c['n']} | {c['n_images']} | "
                        f"{c['fail180_rate']:.4f} | {c['fail360_rate']:.4f} | "
                        f"{rec} | {img} | "
                        f"{c['rows_flipped_to_fail']} | {c['n_antipodal']} |")
                lines.append("")

    # ---- paired operator comparison -------------------------------------
    ops_files = sorted(glob.glob(os.path.join(args.runs_dir,
                                              "p0_operators/*/rows.jsonl")))
    if ops_files:
        ops_rows = []
        for p in ops_files:
            ops_rows.extend(load_rows(p))
        paired = paired_vs_reference(ops_rows, args.sync_tol,
                                     ref=("pil_bilinear", "tv_nearest_c31.5"))
        if paired:
            safe_print(f"[ok] paired operator comparison: {len(paired)} cells")
            lines += ["", "---", "",
                      "## 附：operator 的配对差（相对参考格 `pil_bilinear` × "
                      "`tv_nearest_c31.5`）", "",
                      "两个 Wilson 区间重叠**不是**差异检验。这里每一对都是**同一张图、"
                      "同一个角度、同一次反演**下的两种实现，因此给出**配对差**"
                      "及其按图像聚类的 bootstrap 区间（配对差 = cell − 参考格；"
                      "Δ失锁率为正则表示该 cell 更差）。", "",
                      "| cell | n_pairs | BitAcc cell→ref | ΔBitAcc [聚类CI] | "
                      "失锁率 cell→ref | Δ失锁率 [聚类CI] | 不一致对数 |",
                      "|---|---:|---|---|---|---|---:|"]
            for e in paired:
                db, df = e["d_bitacc"], e["d_fail"]
                lines.append(
                    f"| `{e['cell']}` | {e['n_pairs']} | "
                    f"{e['bitacc_cell']:.4f} → {e['bitacc_ref']:.4f} | "
                    f"{_pci(db)} | "
                    f"{e['fail_cell']:.4f} → {e['fail_ref']:.4f} | "
                    f"{_pci(df)} | "
                    f"{df['n_discordant']} |")
            lines += ["", "读法：Δ 的聚类区间**不含 0** 才说明该 cell 与参考格有"
                      "可分辨差异；区间跨 0 时应写成「未观察到差异」，"
                      "而不是「两者相同」。`†` = 所有配对完全相同，区间退化为一点，"
                      "同样只能读作「无差异证据」。", ""]

    # ---- paired B8 vs B16 ----------------------------------------------
    pareto_file = resolve_path("pareto/rows.jsonl", args.runs_dir)
    if pareto_file:
        pr = paired_b8_vs_b16(load_rows(pareto_file), args.sync_tol)
        if pr:
            safe_print(f"[ok] paired B8 vs B16 comparison: {len(pr)} cells")
            lines += ["", "---", "",
                      "## 附：B8 vs B16 的配对差（同图/同 η/同攻击）", "",
                      "**PMR 不能单独支撑「容量增加削弱同步」的结论**：若单比特"
                      "正确率同为 p，理想独立近似下 PMR(B8)≈p⁸、PMR(B16)≈p¹⁶，"
                      "消息翻倍本身就会压低全对率。因此这里同时给 BitAcc（以及"
                      "BER）、失锁率与 PMR，并给**配对差**（B16 − B8）与"
                      "按图像聚类的区间。", "",
                      "| η | case | n_pairs | BitAcc 8→16 | ΔBitAcc [聚类CI] | "
                      "BER 8→16 | 失锁率 8→16 | Δ失锁率 [聚类CI] | "
                      "PMR 8→16 | ΔPMR [聚类CI] |",
                      "|---:|---|---:|---|---|---|---|---|---|---|"]
            for e in pr:
                db, df, dp = e["d_bitacc"], e["d_fail"], e["d_pmr"]
                lines.append(
                    f"| {float(e['eta']):g} | `{e['case']}` | {e['n_pairs']} | "
                    f"{e['bitacc_8']:.4f} → {e['bitacc_16']:.4f} | "
                    f"{_pci(db)} | "
                    f"{e['ber_8']:.4f} → {e['ber_16']:.4f} | "
                    f"{e['fail_8']:.4f} → {e['fail_16']:.4f} | "
                    f"{_pci(df)} | "
                    f"{e['pmr_8']:.3f} → {e['pmr_16']:.3f} | "
                    f"{_pci(dp)} |")
            lines += ["", "判读：只有在 **ΔBitAcc 与 Δ失锁率的聚类区间都不含 0** "
                      "（即单比特可靠性与同步本身都退化）时，才能说容量增加"
                      "确实损害了同步/逐比特恢复；若只有 ΔPMR 显著，"
                      "那更可能是 p^B 的固有惩罚。", ""]

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
                  "锁定率同样分**记录级**与**图像级**两列（含义见上）。", "",
                  "| case \\| 条件 | n | n_img | 记录级锁定率 [聚类CI] | "
                  "图像级（≥1 次锁定） | 角误差 中位/p90 |",
                  "|---|---:|---:|---|---|---|"]
        for key, v in leak.items():
            rec, img = _rate_cols(v["lock_clustered"])
            lines.append(f"| `{key}` | {v['n']} | {v['n_images']} | "
                         f"{rec} | {img} | "
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
