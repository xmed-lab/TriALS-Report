"""Organ-level AUC/F1 with patient bootstrap, and threshold metrics.

Disease taxonomy, question mapping and bootstrap follow the paper.
"""

import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

# Organs excluded from the 15-organ average
EXCLUDE_FROM_AVG = ("Multi_Organs",)




DISEASE_MAPPING = {
    "hepatomegaly": "hepatomegaly",
    "metastatic disease": "metastatic disease",
    "inguinal hernias": "inguinal hernias",
    "hepatic steatosis": "hepatic steatosis",
    "hepatic masses": "hepatic masses",
    "simple renal cysts": "simple renal cysts",
    "hiatal hernia": "hiatal hernia",
    "prostatomegaly": "prostatomegaly",
    "submucosal edema": "submucosal edema",
    "enteric diverticula": "enteric diverticula",
    "splenomegaly": "splenomegaly",
    "lymphadenopathy": "lymphadenopathy",
    "atelectasis": "atelectasis",
    "hepatic cysts": "hepatic cysts",
    "cirrhosis": "cirrhosis",
    "gallbladder stones": "gallbladder stones",
    "congenital liver cysts": "congenital liver cysts",
    "ascites": "ascites",
    "hepatic changes post-resection": "hepatic changes post-resection",
    "hepatic hemangiomas": "hepatic hemangiomas",
    "renal stones": "renal stones",
    "gallbladder surgically absent": "a surgically absent gallbladder",
    "renal hypodensities": "renal hypodensities",
    "adrenal masses": "adrenal masses",
    "hysterectomy": "changes status post hysterectomy",
    "inflammatory bowel disease": "inflammatory bowel disease",
    "hepatocellular carcinoma": "hepatocellular carcinoma",
    "adrenal adenomas": "adrenal adenomas",
    "leiomyomas": "leiomyomas",
    "intra-uterine devices": "intra-uterine devices",
    "hepatic metastases": "hepatic metastases",
    "intrahepatic biliary ductal dilation": "intrahepatic biliary ductal dilation",
    "accessory spleens": "accessory spleens",
    "aortic atherosclerosis": "aortic atherosclerosis",
    "extrahepatic biliary ductal dilation": "extrahepatic biliary ductal dilation",
    "lung masses": "lung masses",
    "osteosclerotic lesions": "osteosclerotic lesions",
    "diverticulitis": "diverticulitis",
    "gastric varices": "esophageal or gastric varices",
    "bowel obstruction": "bowel obstruction",
    "sleeve gastrectomy": "sleeve gastrectomy",
    "portal venous occlusion": "portal venous occlusion",
    "gastric conduit": "surgical gastric conduit",
    "pleural effusion": "pleural effusion",
    "pancreatic tumors": "pancreatic tumors",
    "ovarian tumors": "ovarian tumors",
    "hydronephrosis": "hydronephrosis",
    "colonic carcinomas": "colonic carcinomas",
    "gallbladder wall thickening": "gallbladder wall thickening",
    "pancreatic atrophy": "pancreatic atrophy",
    "cardiomegaly": "cardiomegaly",
    "fractures": "fractures",
    "pancreatic duct dilatation": "main pancreatic duct dilatation",
}

DISEASE_TO_TARGET_ORGAN = {
    # Liver
    "hepatomegaly": "Liver", "hepatic steatosis": "Liver", "hepatic masses": "Liver",
    "cirrhosis": "Liver", "hepatic cysts": "Liver", "congenital liver cysts": "Liver",
    "hepatic changes post-resection": "Liver", "hepatic hemangiomas": "Liver",
    "hepatocellular carcinoma": "Liver", "hepatic metastases": "Liver",
    # Gallbladder vs Biliary tree
    "gallbladder stones": "Gallbladder", "gallbladder wall thickening": "Gallbladder",
    "a surgically absent gallbladder": "Gallbladder",
    "intrahepatic biliary ductal dilation": "Biliary_Tree",
    "extrahepatic biliary ductal dilation": "Biliary_Tree",
    # Pancreas
    "pancreatic tumors": "Pancreas", "pancreatic atrophy": "Pancreas",
    "main pancreatic duct dilatation": "Pancreas",
    # Spleen
    "splenomegaly": "Spleen", "accessory spleens": "Spleen",
    # Adrenal
    "adrenal masses": "Adrenal_gland", "adrenal adenomas": "Adrenal_gland",
    # Genitourinary
    "simple renal cysts": "Genitourinary", "renal stones": "Genitourinary",
    "renal hypodensities": "Genitourinary", "hydronephrosis": "Genitourinary",
    # Pelvis
    "prostatomegaly": "Male_Pelvis",
    "changes status post hysterectomy": "Female_pelvis", "leiomyomas": "Female_pelvis",
    "intra-uterine devices": "Female_pelvis", "ovarian tumors": "Female_pelvis",
    # GI
    "hiatal hernia": "Gastrointestinal", "submucosal edema": "Gastrointestinal",
    "enteric diverticula": "Gastrointestinal", "bowel obstruction": "Gastrointestinal",
    "inflammatory bowel disease": "Gastrointestinal", "sleeve gastrectomy": "Gastrointestinal",
    "surgical gastric conduit": "Gastrointestinal", "diverticulitis": "Gastrointestinal",
    "colonic carcinomas": "Gastrointestinal",
    # Peritoneum
    "ascites": "Peritoneum",
    # Great vessels
    "aortic atherosclerosis": "Great_Vessel", "portal venous occlusion": "Great_Vessel",
    "cardiomegaly": "Great_Vessel", "esophageal or gastric varices": "Great_Vessel",
    # Visible thoracic
    "atelectasis": "Visible_Thoracic", "lung masses": "Visible_Thoracic",
    "pleural effusion": "Visible_Thoracic",
    # Retroperitoneum
    "lymphadenopathy": "Retroperitoneum",
    # MSK
    "fractures": "Musculoskeletal", "osteosclerotic lesions": "Musculoskeletal",
    # Multi-organ / systemic
    "metastatic disease": "Multi_Organs", "inguinal hernias": "Multi_Organs",
}



def norm_q(s):
    s = str(s).strip().lower()
    s = re.sub(r"[?\.]+$", "", s)
    s = re.sub(
        r"^(are there any|are there|is there any|is there|does the patient have|is)\s+",
        "", s
    ).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def map_question_to_disease(q_norm):
    matches = []
    for key, canonical in DISEASE_MAPPING.items():
        k = key.lower().strip()
        if k and k in q_norm:
            matches.append((len(k), key, canonical))
    if not matches:
        return None
    matches.sort(reverse=True, key=lambda x: x[0])
    return matches[0][2]




def load_and_map(exam_csv_path, allowed_accessions=None):
    """Map question -> disease -> organ. Takes a DataFrame or a CSV path."""
    df = exam_csv_path.copy() if isinstance(exam_csv_path, pd.DataFrame) else pd.read_csv(exam_csv_path)
    required = {"accession", "question", "probability", "prediction", "true_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions missing columns {missing}")
    if allowed_accessions is not None:
        df = df[df["accession"].isin(allowed_accessions)].copy()
    df["q_norm"]       = df["question"].apply(norm_q)
    df["disease"]      = df["q_norm"].apply(map_question_to_disease)
    df = df[df["disease"].notna()].copy()
    df["target_organ"] = df["disease"].map(DISEASE_TO_TARGET_ORGAN)
    df = df[df["target_organ"].notna()].copy()
    return df



def _stratified_resample(g, rng):
    """Resample positives and negatives separately to preserve prevalence."""
    idx_pos = g.index[g["true_label"] == 1].to_numpy()
    idx_neg = g.index[g["true_label"] == 0].to_numpy()
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return None
    s_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
    s_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
    return g.loc[np.concatenate([s_pos, s_neg])]


def _disease_metrics(sample):
    f1 = f1_score(sample["true_label"], sample["prediction"], zero_division=0)
    if sample["true_label"].nunique() > 1:
        auc = roc_auc_score(sample["true_label"], sample["probability"])
    else:
        auc = np.nan
    return f1, auc


def bootstrap_one_csv(exam_csv_path, n_boot=1000, alpha=0.05, seed=42,
                       exclude_from_avg=EXCLUDE_FROM_AVG,
                       allowed_accessions=None):
    """
    Run patient-level stratified bootstrap, return per-organ and macro AVG
    point estimates with 95% percentile CIs.

    If allowed_accessions is provided, only those patients are included
    (used to enforce matched cohort across phases).
    """
    rng = np.random.default_rng(seed)
    df = load_and_map(exam_csv_path, allowed_accessions=allowed_accessions)

    groups = {key: g.reset_index(drop=True)
              for key, g in df.groupby(["target_organ", "disease"])}
    organs = sorted({k[0] for k in groups})
    n_diseases_per_organ = {o: sum(1 for k in groups if k[0] == o) for o in organs}
    organs_in_avg = [o for o in organs if o not in exclude_from_avg]

    boot_per_organ = {o: {"F1": [], "AUC": []} for o in organs}
    boot_avg = {"F1": [], "AUC": []}

    for _ in range(n_boot):
        # disease-level metrics under this resample, grouped by organ
        organ_dis_f1  = {o: [] for o in organs}
        organ_dis_auc = {o: [] for o in organs}
        for (organ, _disease), g in groups.items():
            samp = _stratified_resample(g, rng)
            if samp is None:
                continue
            f1, auc = _disease_metrics(samp)
            organ_dis_f1[organ].append(f1)
            organ_dis_auc[organ].append(auc)

        # organ-level means
        organ_f1, organ_auc = {}, {}
        for o in organs:
            if organ_dis_f1[o]:
                organ_f1[o]  = float(np.nanmean(organ_dis_f1[o]))
                organ_auc[o] = float(np.nanmean(organ_dis_auc[o]))
                boot_per_organ[o]["F1"].append(organ_f1[o])
                boot_per_organ[o]["AUC"].append(organ_auc[o])

        # 15-organ AVG (drop Multi_Organs)
        avg_f1  = [organ_f1[o]  for o in organs_in_avg if o in organ_f1]
        avg_auc = [organ_auc[o] for o in organs_in_avg if o in organ_auc]
        if avg_f1:  boot_avg["F1"].append(np.nanmean(avg_f1))
        if avg_auc: boot_avg["AUC"].append(np.nanmean(avg_auc))

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def summarize(arr):
        a = np.array(arr, dtype=float)
        return {
            "mean": float(np.nanmean(a)) if a.size else float("nan"),
            "lo":   float(np.nanpercentile(a, lo_q)) if a.size else float("nan"),
            "hi":   float(np.nanpercentile(a, hi_q)) if a.size else float("nan"),
        }

    per_organ_out = {}
    for o in organs:
        f1s = summarize(boot_per_organ[o]["F1"])
        aus = summarize(boot_per_organ[o]["AUC"])
        per_organ_out[o] = {
            "F1": f1s["mean"], "F1_lo": f1s["lo"], "F1_hi": f1s["hi"],
            "AUC": aus["mean"], "AUC_lo": aus["lo"], "AUC_hi": aus["hi"],
            "n_diseases": n_diseases_per_organ[o],
        }

    avg_f1  = summarize(boot_avg["F1"])
    avg_auc = summarize(boot_avg["AUC"])
    macro_out = {
        "F1": avg_f1["mean"], "F1_lo": avg_f1["lo"], "F1_hi": avg_f1["hi"],
        "AUC": avg_auc["mean"], "AUC_lo": avg_auc["lo"], "AUC_hi": avg_auc["hi"],
        "organs_included": organs_in_avg,
        "n_organs": len(organs_in_avg),
    }

    return {
        "per_organ": per_organ_out,
        "macro_avg_15": macro_out,
        "n_patients": int(df["accession"].nunique()),
        "n_rows": int(len(df)),
        "n_boot": n_boot,
        "alpha": alpha,
        "seed": seed,
    }


bootstrap = bootstrap_one_csv

NAN = float("nan")


def _wilson(k, n, z=1.96):
    if n == 0:
        return NAN, NAN
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _ratio(a, b):
    return a / b if b else NAN


def _rates(tp, fn, fp, tn):
    sens, spec, prec = _ratio(tp, tp + fn), _ratio(tn, tn + fp), _ratio(tp, tp + fp)
    if math.isnan(sens) or math.isnan(prec):
        f1 = NAN
    elif tp == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * sens / (prec + sens)
    return sens, spec, prec, f1


def _nanmean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return (sum(xs) / len(xs), len(xs)) if xs else (NAN, 0)


def threshold_metrics(pred, min_pos=5):
    """Return (per_question DataFrame, overall DataFrame) from a predictions DataFrame."""
    counts = defaultdict(lambda: [0, 0, 0, 0])
    for q, y, p in zip(pred["question"], pred["true_label"].astype(int), pred["prediction"].astype(int)):
        counts[q][(0 if p else 1) if y else (2 if p else 3)] += 1
    per = []
    for q in sorted(counts):
        tp, fn, fp, tn = counts[q]
        sens, spec, prec, f1 = _rates(tp, fn, fp, tn)
        (s_lo, s_hi), (sp_lo, sp_hi), (p_lo, p_hi) = _wilson(tp, tp + fn), _wilson(tn, tn + fp), _wilson(tp, tp + fp)
        per.append(dict(question=q, n=tp + fn + fp + tn, positives=tp + fn, TP=tp, FN=fn, FP=fp, TN=tn,
                        sensitivity=sens, sens_ci_lo=s_lo, sens_ci_hi=s_hi,
                        specificity=spec, spec_ci_lo=sp_lo, spec_ci_hi=sp_hi,
                        precision=prec, prec_ci_lo=p_lo, prec_ci_hi=p_hi, f1=f1))
    TP, FN, FP, TN = (sum(v[i] for v in counts.values()) for i in range(4))
    s, sp, pr, f = _rates(TP, FN, FP, TN)
    overall = [dict(aggregation="micro", questions_used=sum(1 for v in counts.values() if v[0] + v[1]),
                    sensitivity=s, specificity=sp, precision=pr, f1=f)]
    for name, subset in (("macro", per), (f"macro_pos>={min_pos}", [r for r in per if r["positives"] >= min_pos])):
        ms, n_used = _nanmean([r["sensitivity"] for r in subset])
        overall.append(dict(aggregation=name, questions_used=n_used, sensitivity=ms,
                            specificity=_nanmean([r["specificity"] for r in subset])[0],
                            precision=_nanmean([r["precision"] for r in subset])[0],
                            f1=_nanmean([r["f1"] for r in subset])[0]))
    return pd.DataFrame(per), pd.DataFrame(overall)
