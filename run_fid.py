"""FID between image folders (e.g. real COCO, generated no_w, generated w).

Usage: python run_fid.py DIR_A DIR_B [DIR_C ...]

The literature reports FID at 5000-10000 images per method; at N <= 50 the
estimate is dominated by finite-sample bias and must not be quoted as a
comparison.  This wrapper therefore counts each folder, refuses to run below
``--require_n`` (default 0 = off) and prints an explicit warning below
``--warn_min_n`` (default 1000).  Results are written to JSON with the folder
sizes and environment provenance so a table can be rebuilt later.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

from pytorch_fid.fid_score import calculate_fid_given_paths


def count_images(d: str) -> int:
    pats = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG")
    return sum(len(glob.glob(os.path.join(d, p))) for p in pats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--warn_min_n", type=int, default=1000)
    ap.add_argument("--require_n", type=int, default=0,
                    help="abort if any folder has fewer than this many images")
    ap.add_argument("--ref_only", action="store_true",
                    help="only compare every folder against the first one")
    ap.add_argument("--json_out", default="")
    args = ap.parse_args()
    paths = [os.path.abspath(d) for d in args.dirs]
    for d in paths:
        assert os.path.isdir(d), f"not a directory: {d}"
    counts = {d: count_images(d) for d in paths}
    for d, n in counts.items():
        print(f"{os.path.basename(d):>24}: {n} images")
        if n < args.require_n:
            raise SystemExit(
                f"folder {d} has {n} images < --require_n {args.require_n}")
        if n < args.warn_min_n:
            print(f"[warn] {os.path.basename(d)} has only {n} images; FID at "
                  f"this sample size is not comparable across methods "
                  f"(literature uses 5000-10000).")

    results = []
    for i in range(len(paths)):
        js = [0] if args.ref_only else range(i + 1, len(paths))
        for j in js:
            if i == j:
                continue
            fid = calculate_fid_given_paths(
                [paths[i], paths[j]],
                batch_size=args.batch_size,
                device=args.device,
                dims=2048,
                num_workers=args.num_workers,
            )
            print(
                f"FID({os.path.basename(paths[i])} vs "
                f"{os.path.basename(paths[j])}) = {fid:.3f}"
            )
            results.append({
                "a": paths[i], "b": paths[j], "fid": float(fid),
                "n_a": counts[paths[i]], "n_b": counts[paths[j]],
            })
    if args.json_out:
        payload = {"results": results, "counts": counts,
                   "batch_size": args.batch_size, "device": args.device,
                   "dims": 2048, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "warn_min_n": args.warn_min_n}
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)),
                    exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"-> {args.json_out}")


if __name__ == "__main__":
    main()
