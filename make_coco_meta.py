"""Build a COCO prompt meta file from the raw MS-COCO 5k test CSV.

``get_dataset("coco:<path>")`` expects
``{"annotations": [{"image_id", "file_name", "caption"}, ...]}``.  The raw file
from ``nlphuji/mscoco_2014_5k_test_image_text_retrieval`` stores five captions
per image in a JSON list inside the ``raw`` column, so this script flattens it to
one caption per image (the first), matching the format *and* the ordering of the
existing 1000-caption ``meta_data.json``.

That matters for scale: the instance shipped with only 1000 captions, which is
why the P0 queues had to pass explicit disjoint index offsets (calib 0..N-1,
test 500..).  With the full 5000 there is room for the literature protocol
(1000 watermarked + 1000 unwatermarked, plus disjoint calibration splits).

    python make_coco_meta.py /root/autodl-tmp/data/coco5k_raw/test_5k_mscoco_2014.csv \
        /root/autodl-tmp/data/coco5k/meta_data_5k.json \
        --verify_against /root/autodl-tmp/data/coco5k/meta_data.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os


def read_records(csv_path: str) -> list:
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            captions = json.loads(row["raw"])
            if not captions:
                continue
            records.append({"image_id": int(row["cocoid"]),
                            "file_name": row["filename"],
                            "caption": captions[0].strip()})
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("out_path")
    ap.add_argument("--verify_against", default="",
                    help="existing meta file: report how many of its entries are "
                         "reproduced byte-for-byte by this conversion")
    args = ap.parse_args()

    records = read_records(args.csv_path)
    if not records:
        raise SystemExit(f"no records parsed from {args.csv_path}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump({"annotations": records}, f)
    print(f"{len(records)} captions -> {args.out_path}")
    print(f"  first : {records[0]['file_name']} | {records[0]['caption'][:60]}")
    print(f"  last  : {records[-1]['file_name']} | {records[-1]['caption'][:60]}")

    if args.verify_against and os.path.exists(args.verify_against):
        with open(args.verify_against, encoding="utf-8") as f:
            old = json.load(f)["annotations"]
        n = min(len(old), len(records))
        same = sum(
            1 for a, b in zip(old[:n], records[:n])
            if a["image_id"] == b["image_id"]
            and a["file_name"] == b["file_name"]
            and a["caption"].strip() == b["caption"].strip())
        print(f"  vs {os.path.basename(args.verify_against)}: "
              f"{same}/{n} of its entries reproduced exactly")


if __name__ == "__main__":
    main()
