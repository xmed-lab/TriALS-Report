#!/usr/bin/env python3
"""Linear probe on released features. Train on Center 1 train, test internal and external.

  python run_benchmark.py --data ./TriALS-Report
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from trials_benchmark import metrics
from trials_benchmark.data import available_models, load_split
from trials_benchmark.probe import LinearProbe, ProbeConfig

TRAIN = ("center1", "train")
VAL = ("center1", "val")
TESTS = {"internal": ("center1", "test"), "external": ("center2", "test")}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=ProbeConfig.num_epochs)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-val", action="store_true",
                    help="skip validation and keep the last epoch (the paper's protocol)")
    return ap.parse_args()


def check_splits(train_ids, tests):
    """Test patients of the same center must not appear in train.

    Patient ids are only unique within a center, so ids are compared per center.
    """
    train_ids = set(train_ids)
    for name, (center, _), ids in tests:
        if center != TRAIN[0]:
            print(f"  {name}: {len(ids)} patients (different center)")
            continue
        overlap = train_ids & set(ids)
        if overlap:
            raise SystemExit(f"{name}: {len(overlap)} patients also in train: {sorted(overlap)[:10]}")
        print(f"  {name}: {len(ids)} patients, none in train")


def predictions_frame(ids, questions, probs, labels):
    n, q = probs.shape
    return pd.DataFrame({
        "accession": np.repeat(ids, q),
        "question": np.tile(questions, n),
        "probability": probs.ravel(),
        "prediction": (probs > 0.5).astype(int).ravel(),
        "true_label": np.stack([labels[qq] for qq in questions], axis=1).ravel(),
    })


def main():
    args = parse_args()
    models = args.models or available_models(args.data)
    os.makedirs(args.out, exist_ok=True)
    organ_rows = []

    for model in models:
        print(f"\n{model}")
        X_tr, Y_tr, ids_tr = load_split(args.data, model, *TRAIN)
        tests = {n: load_split(args.data, model, *cs) for n, cs in TESTS.items()}
        questions = sorted(Y_tr)
        print(f"  train: {len(ids_tr)} patients, dim {X_tr.shape[1]}, {len(questions)} questions")

        val = None
        if not args.no_val:
            try:
                X_va, Y_va, ids_va = load_split(args.data, model, *VAL)
                val = (X_va, Y_va)
                print(f"  val:   {len(ids_va)} patients, used to pick the epoch")
            except (FileNotFoundError, ValueError) as e:
                print(f"  val:   unavailable ({e}); keeping the last epoch")
        else:
            print("  val:   skipped (--no-val), keeping the last epoch")

        check_splits(ids_tr, [(n, TESTS[n], t[2]) for n, t in tests.items()]
                     + ([("val", VAL, ids_va)] if val is not None else []))

        for seed in args.seeds:
            t0 = time.time()
            probe = LinearProbe(questions, ProbeConfig(num_epochs=args.epochs), args.device)
            probe.fit(X_tr, Y_tr, seed=seed, val=val)
            for name, (X_te, Y_te, ids) in tests.items():
                out = os.path.join(args.out, model, name, f"seed{seed}")
                os.makedirs(out, exist_ok=True)
                pred = predictions_frame(ids, questions, probe.predict_proba(X_te), Y_te)
                pred.to_csv(os.path.join(out, "predictions.csv"), index=False)

                boot = metrics.bootstrap(pred, n_boot=args.n_boot, seed=42)
                json.dump(boot, open(os.path.join(out, "bootstrap.json"), "w"), indent=2)
                per_q, overall = metrics.threshold_metrics(pred)
                per_q.to_csv(os.path.join(out, "threshold_per_question.csv"), index=False)
                overall.to_csv(os.path.join(out, "threshold_overall.csv"), index=False)

                for organ, r in [("AVG", boot["macro_avg_15"])] + sorted(boot["per_organ"].items()):
                    organ_rows.append(dict(model=model, test_set=name, seed=seed,
                                           epoch=probe.best_epoch, val_auc=probe.best_val_auc, organ=organ,
                                           AUC=r["AUC"], AUC_lo=r["AUC_lo"], AUC_hi=r["AUC_hi"],
                                           F1=r["F1"], F1_lo=r["F1_lo"], F1_hi=r["F1_hi"]))
                a = boot["macro_avg_15"]
                val_note = "" if probe.best_val_auc is None else f" val AUC {probe.best_val_auc:.3f}"
                print(f"  seed {seed} {name:8s} AUC {a['AUC']:.3f} ({a['AUC_lo']:.3f}-{a['AUC_hi']:.3f})"
                      f"  F1 {a['F1']:.3f} ({a['F1_lo']:.3f}-{a['F1_hi']:.3f})"
                      f"  [epoch {probe.best_epoch}{val_note}, {time.time() - t0:.0f}s]")

    organs = pd.DataFrame(organ_rows)
    organs.to_csv(os.path.join(args.out, "summary_organs.csv"), index=False)
    avg = organs[organs.organ == "AVG"].groupby(["model", "test_set"]).agg(
        AUC_mean=("AUC", "mean"), AUC_sd=("AUC", "std"),
        F1_mean=("F1", "mean"), F1_sd=("F1", "std"), seeds=("seed", "count")).reset_index()
    avg.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\n" + avg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
