"""Load labels and features."""

import os

import numpy as np
import pandas as pd

CENTER_DIRS = {"center1": "Center 1", "center2": "Center 2"}


def available_models(data_root):
    feat = os.path.join(data_root, "features")
    if not os.path.isdir(feat):
        raise FileNotFoundError(f"no features/ folder in {data_root}")
    return sorted(d for d in os.listdir(feat) if os.path.isdir(os.path.join(feat, d)))


def question_map(data_root):
    d = pd.read_csv(os.path.join(data_root, "label_dictionary.csv"))
    return dict(zip(d["column"], d["question"]))


def load_split(data_root, model, center, split):
    """Return X (N, D), {question: labels}, patient_ids for one center/split."""
    labels = pd.read_csv(os.path.join(data_root, CENTER_DIRS[center], "labels.csv"),
                         dtype={"patient_id": str})
    feats = pd.read_parquet(os.path.join(data_root, "features", model, f"{center}.parquet"))
    feats["patient_id"] = feats["patient_id"].astype(str)

    labels = labels[labels["split"] == split]
    feats = feats[feats["split"] == split]
    merged = labels.merge(feats[["patient_id", "embedding"]], on="patient_id", how="inner")

    dropped = sorted(set(labels["patient_id"]) - set(merged["patient_id"]))
    if dropped:
        print(f"  {model} {center}/{split}: no features for {len(dropped)} patients, dropped: {dropped[:10]}")
    if merged.empty:
        raise ValueError(f"{model} {center}/{split}: no patients with labels and features")

    qmap = question_map(data_root)
    X = np.stack([np.asarray(e, dtype=np.float32) for e in merged["embedding"]])
    Y = {qmap[c]: merged[c].fillna(0).astype(int).to_numpy() for c in qmap if c in merged.columns}
    return X, Y, merged["patient_id"].tolist()
