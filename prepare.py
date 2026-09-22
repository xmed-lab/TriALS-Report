#!/usr/bin/env python3
"""Build the inputs for feature extraction from the released dataset alone.

  python prepare.py series      --data ./TriALS-Report --work ./work
  vision-engine process ...     (one run per list, see README)
  python prepare.py rate-inputs --data ./TriALS-Report --work ./work
  rate-extract ...              (one run per model and list, see README)
  python prepare.py features    --data ./TriALS-Report --work ./work --model pillar0
  python prepare.py radar-inputs --data ./TriALS-Report --work ./work   (test CTs named for DAMO RADAR)

Lists are <center>_<split>: center1_train, center1_val, center1_test, center2_test.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

CENTERS = {"Center 1": "center1", "Center 2": "center2"}
PREFIX = "non_contrast_"


def cohort(data):
    """{list_name: [(patient_id, split, ct_path)]} read from each labels.csv."""
    lists = {}
    for center, key in CENTERS.items():
        with open(os.path.join(data, center, "labels.csv")) as fh:
            for r in csv.DictReader(fh):
                path = os.path.abspath(os.path.join(data, center, r["image"]))
                lists.setdefault(f"{key}_{r['split']}", []).append((r["patient_id"], r["split"], path))
    return lists


def cmd_series(args):
    out = os.path.join(args.work, "series")
    os.makedirs(out, exist_ok=True)
    missing = []
    for name, rows in cohort(args.data).items():
        with open(os.path.join(out, f"{name}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["series_path"])
            for _, _, path in rows:
                if not os.path.exists(path):
                    missing.append(path)
                w.writerow([path])
        print(f"{name:<14} {len(rows):5d} volumes -> {out}/{name}.csv")
    if missing:
        sys.exit(f"{len(missing)} CT files listed in labels.csv are missing, e.g. {missing[:3]}")


def cmd_rate_inputs(args):
    out = os.path.join(args.work, "rate")
    os.makedirs(out, exist_ok=True)
    for name, rows in cohort(args.data).items():
        mapping = os.path.join(args.work, "cache", name, "mapping.csv")
        if not os.path.exists(mapping):
            print(f"{name:<14} skipped (no {mapping})")
            continue
        cache = {}
        with open(mapping) as fh:
            for m in csv.DictReader(fh):
                pid = os.path.basename(m["source_path"])[:-len(".nii.gz")]
                p = m["output_path"]
                cache[pid] = p if os.path.isabs(p) else os.path.abspath(os.path.join(os.path.dirname(mapping), p))
        absent = [pid for pid, _, _ in rows if pid not in cache]
        if absent:
            sys.exit(f"{name}: {len(absent)} patients were not preprocessed, e.g. {absent[:5]}")
        with open(os.path.join(out, f"{name}.jsonl"), "w") as fh:
            for pid, _, _ in rows:
                fh.write(json.dumps({"sample_name": PREFIX + pid, "nii_path": None,
                                     "report_metadata": "FINDINGS: "}) + "\n")
        with open(os.path.join(out, f"manifest_{name}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sample_name", "image_cache_path"])
            w.writerows([PREFIX + pid, cache[pid]] for pid, _, _ in rows)
        print(f"{name:<14} {len(rows):5d} samples -> {out}/{name}.jsonl, manifest_{name}.csv")


def cmd_features(args):
    lists = cohort(args.data)
    out = os.path.join(args.data, "features", args.model)
    os.makedirs(out, exist_ok=True)
    import pandas as pd
    for key in CENTERS.values():
        ids, splits, embs, absent = [], [], [], []
        for name, rows in lists.items():
            if not name.startswith(key + "_"):
                continue
            emb_dir = os.path.join(args.work, "emb", args.model, name, "embeddings", "train")
            for pid, split, _ in rows:
                f = os.path.join(emb_dir, f"{PREFIX}{pid}.npz")
                if not os.path.exists(f):
                    absent.append(f"{name}:{pid}")
                    continue
                with np.load(f) as d:
                    k = "embedding" if "embedding" in d else list(d.keys())[0]
                    embs.append(np.asarray(d[k], dtype=np.float32).ravel())
                ids.append(pid)
                splits.append(split)
        pd.DataFrame({"patient_id": ids, "split": splits, "embedding": embs}).to_parquet(
            os.path.join(out, f"{key}.parquet"), index=False)
        note = f", missing {len(absent)} (e.g. {absent[:3]})" if absent else ""
        print(f"{args.model} {key}: {len(ids)} embeddings -> {out}/{key}.parquet{note}")


def cmd_radar_inputs(args):
    out = os.path.join(args.work, "radar_inputs")
    os.makedirs(out, exist_ok=True)
    n = 0
    for name, rows in cohort(args.data).items():
        if not name.endswith("_test"):
            continue
        tag = name.split("_")[0].replace("center", "Center")
        for pid, _, path in rows:
            link = os.path.join(out, f"{tag}__{pid}.nii.gz")
            if not os.path.exists(link):
                os.symlink(path, link)
            n += 1
    print(f"{n} test volumes linked in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("series", cmd_series), ("rate-inputs", cmd_rate_inputs), ("features", cmd_features),
                     ("radar-inputs", cmd_radar_inputs)):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True, help="the downloaded TriALS-Report folder")
        p.add_argument("--work", required=True, help="working folder for intermediate files")
        if name == "features":
            p.add_argument("--model", required=True, help="name of the features folder, e.g. pillar0")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
