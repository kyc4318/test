"""Assemble the comparison tables from the unified runs.

Reads runs/paper_*/<method>.json written by run_paper_compare.py and produces
results/paper_main_tables.md (+ .json).  Tables are task-partitioned:

  T1 payload   bit payload methods (ours, ours-nosync, GS-256, GS-8)
  T2 detection presence detection for every ported method
  T3 identify  key-identification methods (SFWMark HSTR / HSQR)

RingID / MaXsive / Tree-Ring are run by their own official scripts; their
numbers are appended from results/*.json when present.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import OrderedDict

import numpy as np

LABEL = {
    "ours": "SectorSync (ours)",
    "ours_nosync": "ours, no sync search",
    "gs256": "Gaussian Shading 256b",
    "gs8": "Gaussian Shading 8b",
    "sfw_hstr": "SFWMark HSTR (2048 keys)",
    "sfw_hsqr": "SFWMark HSQR (2048 keys)",
}
CASE_ORDER = [
    "clean", "jpeg25", "jpeg75", "noise0.05", "noise0.1", "blur4", "blur5",
    "median5", "bright6.0", "contrast0.5", "bm3d0.1", "resize0.9", "resize1.25",
    "rot15", "rot30", "rot45", "rot60", "rot75", "rot90",
    "rot+noise0.05", "rot75+noise0.05", "rot75+noise0.1", "rot75+jpeg75",
    "rot45+resize1.25", "rot30+blur4", "cc0.5", "cc0.7", "rc0.7",
    "vaeb3", "vaec3", "regen300", "regen600",
]


def load_runs(pattern: str) -> "OrderedDict[str, dict]":
    runs = OrderedDict()
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            blob = json.load(f)
        name = blob.get("method") or os.path.basename(path)[:-5]
        blob["_path"] = path
        runs[name] = blob
    return runs


def order_cases(cases):
    known = [c for c in CASE_ORDER if c in cases]
    return known + [c for c in cases if c not in known]


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/paper_*/*.json")
    ap.add_argument("--out", default="results/paper_main_tables.md")
    ap.add_argument("--json_out", default="results/paper_main_tables.json")
    ap.add_argument("--identity", default="runs/identity/summary.json",
                    help="P0 identity benchmark summary (adds ours to T3)")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        print(f"no runs matched {args.runs!r}")
        return
    cases = []
    for b in runs.values():
        for c in b.get("cases", []):
            if c not in cases:
                cases.append(c)
    cases = order_cases(cases)

    def s(method, case, key):
        b = runs.get(method)
        return None if b is None else b["summary"].get(case, {}).get(key)

    def det(method, case, key):
        b = runs.get(method)
        if b is None:
            return None
        return b["summary"].get(case, {}).get("detection", {}).get(key)

    payload = [m for m in runs if m not in ("sfw_hstr", "sfw_hsqr")]
    identify = [m for m in runs if m in ("sfw_hstr", "sfw_hsqr")]

    def bitacc_cell(m, case):
        ba = s(m, case, "wm_bit_acc")
        pm = s(m, case, "wm_perfect")
        if ba is None:
            ba = s(m, case, "wm_bit_acc_native")
            pm = s(m, case, "wm_native_perfect")
        if ba is None:
            return "--"
        return f"{fmt(ba, 4)} / {fmt(pm)}"

    lines = ["# SectorSync vs open-source baselines -- unified protocol", ""]
    lines += [f"Methods: {', '.join(LABEL.get(m, m) for m in runs)}.",
              "",
              "Protocol: SD2.1-base, 512x512, DDIM-50 generation and inversion, "
              "COCO-5k prompts, one inversion per attacked image, "
              "per-sample paired attack realisation.",
              ""]

    lines += ["## T1 payload tracing (bit payload)", "",
              "Cell = `BitAcc / PMR`; PMR is the fraction of images with every "
              "bit correct (exhaustive identification over 2^B messages).", ""]
    lines += ["| case | " + " | ".join(LABEL.get(m, m) for m in payload) + " |",
              "|" + "---|" * (len(payload) + 1)]
    for case in cases:
        lines.append(f"| {case} | " + " | ".join(bitacc_cell(m, case)
                                                  for m in payload) + " |")
    lines.append("")

    lines += ["## T2 presence detection (watermark vs no watermark)", "",
              "Cell = `AUC / TPR@1%FPR`.", ""]
    lines += ["| case | " + " | ".join(LABEL.get(m, m) for m in runs) + " |",
              "|" + "---|" * (len(runs) + 1)]
    for case in cases:
        cells = []
        for m in runs:
            auc, tpr = det(m, case, "auc"), det(m, case, "tpr_at_fpr_0.01")
            cells.append(f"{fmt(auc)} / {fmt(tpr)}" if auc is not None else "--")
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
    lines.append("")

    if identify:
        lines += ["## T3 identification (key task, own key space)", "",
                  "Cell = `Id-Acc / TPR@1%FPR`; SFWMark uses 2048 keys.", ""]
        lines += ["| case | " + " | ".join(LABEL.get(m, m) for m in identify) + " |",
                  "|" + "---|" * (len(identify) + 1)]
        for case in cases:
            cells = []
            for m in identify:
                ida, tpr = s(m, case, "wm_id_correct"), det(m, case, "tpr_at_fpr_0.01")
                cells.append(f"{fmt(ida)} / {fmt(tpr)}" if ida is not None else "--")
            lines.append(f"| {case} | " + " | ".join(cells) + " |")
        lines.append("")
        lines += ["Note: our PMR over 2^8 messages is the same *quantity* as "
                  "Id-Acc over a key space, with B = 8 instead of 11 key bits; "
                  "the capacities are quoted with every table.", ""]

    # T3b: the P0 identity benchmark, which runs our method through the *same*
    # registry protocol as SFWMark instead of describing the equivalence.
    if os.path.exists(args.identity):
        with open(args.identity) as f:
            ident = json.load(f)
        isum = ident.get("summary", {})
        if isum:
            lines += ["## T3b identification incl. ours (P0 registry protocol)",
                      "",
                      "Cell = `Id-Acc@1`; our key space is `2^B`, so the "
                      "registered count is reported with every row.", "",
                      "| method | key space | registered | case | Id-Acc@1 "
                      "[95% CI] | margin (pos) | AUC(pos vs null) | TPR@1%FPR |",
                      "|---|---:|---:|---|---|---:|---:|---:|"]
            for m, e in isum.items():
                for case, c in e.get("cases", {}).items():
                    a = c.get("id_acc_soft") or {}
                    acc = (f"{a.get('mean', float('nan')):.3f} "
                           f"[{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]"
                           if a.get("ci") else "--")
                    lines.append(
                        f"| {m} | {e.get('key_space')} | "
                        f"{e.get('n_keys_registered')} | {case} | {acc} | "
                        f"{fmt(c.get('margin_mean_pos'))} | "
                        f"{fmt(c.get('auc_pos_vs_null'))} | "
                        f"{fmt(c.get('tpr_at_calibrated_fpr'))} |")
            lines.append("")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(args.json_out, "w") as f:
        json.dump({m: {"path": b["_path"], "summary": b["summary"]}
                   for m, b in runs.items()}, f, indent=2)
    print(text)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
