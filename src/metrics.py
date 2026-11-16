"""Metrik evaluasi klasifikasi multikelas (keseluruhan + per kelas) dan bootstrap CI."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)


def _safe(fn, *a, **k):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(fn(*a, **k))
    except ValueError:
        return float("nan")


def expected_calibration_error(y_true, prob, n_bins: int = 15) -> float:
    conf = prob.max(1)
    pred = prob.argmax(1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def multiclass_brier(y_true, prob) -> float:
    onehot = np.eye(prob.shape[1])[y_true]
    return float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))


def compute_metrics(y_true, prob, class_names: list[str], key_prefixes: dict | None = None) -> dict:
    """Semua metrik dari probabilitas softmax.

    key_prefixes: nama pendek per kelas untuk kunci datar, mis. {"Abnormal_TB": "tb"}
                  -> menghasilkan 'tb_recall', 'tb_f1', dst.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    k = len(class_names)
    labels = list(range(k))
    pred = prob.argmax(1)
    cm = confusion_matrix(y_true, pred, labels=labels)

    m = {
        "n": int(len(y_true)),
        "accuracy": _safe(accuracy_score, y_true, pred),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, pred),
        "macro_precision": _safe(precision_score, y_true, pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": _safe(recall_score, y_true, pred, labels=labels, average="macro", zero_division=0),
        "macro_f1": _safe(f1_score, y_true, pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": _safe(f1_score, y_true, pred, labels=labels, average="weighted", zero_division=0),
        "mcc": _safe(matthews_corrcoef, y_true, pred),
        "cohen_kappa": _safe(cohen_kappa_score, y_true, pred),
        "macro_auc": _safe(roc_auc_score, y_true, prob, multi_class="ovr", average="macro", labels=labels),
        "weighted_auc": _safe(roc_auc_score, y_true, prob, multi_class="ovr", average="weighted", labels=labels),
        "log_loss": _safe(log_loss, y_true, np.clip(prob, 1e-12, 1), labels=labels),
        "brier": multiclass_brier(y_true, prob),
        "ece": expected_calibration_error(y_true, prob),
        "confusion_matrix": cm.tolist(),
    }
    spec_all = []
    per_class = {}
    for i, c in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        yb = (y_true == i).astype(int)
        d = {
            "support": int(cm[i].sum()),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,          # = sensitivitas
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
            "auc": _safe(roc_auc_score, yb, prob[:, i]),
            "ap": _safe(average_precision_score, yb, prob[:, i]),          # PR-AUC
        }
        d["f1"] = (2 * d["precision"] * d["recall"] / (d["precision"] + d["recall"])
                   if d["precision"] + d["recall"] else 0.0)
        per_class[c] = d
        spec_all.append(d["specificity"])
    m["macro_specificity"] = float(np.mean(spec_all))
    m["macro_sensitivity"] = m["macro_recall"]
    m["per_class"] = per_class

    # kunci datar agar mudah dianalisis (mis. tb_recall, normal_f1)
    key_prefixes = key_prefixes or {c: c.lower() for c in class_names}
    for c, short in key_prefixes.items():
        for kk, v in per_class[c].items():
            m[f"{short}_{kk}"] = v
    return m


def _fast_metric(name: str, y, prob, pred, k: int, short2idx: dict) -> float:
    """Metrik cepat untuk bootstrap (berbasis confusion matrix numpy)."""
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    sup = cm.sum(1)
    predc = cm.sum(0)
    rec = np.divide(tp, sup, out=np.zeros(k), where=sup > 0)
    prec = np.divide(tp, predc, out=np.zeros(k), where=predc > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    tn = cm.sum() - sup - predc + tp
    spec = np.divide(tn, tn + (predc - tp), out=np.zeros(k), where=(tn + predc - tp) > 0)
    if name == "accuracy":
        return tp.sum() / cm.sum()
    if name in ("balanced_accuracy", "macro_recall", "macro_sensitivity"):
        return rec.mean()
    if name == "macro_f1":
        return f1.mean()
    if name == "macro_precision":
        return prec.mean()
    if name == "macro_specificity":
        return spec.mean()
    if name == "macro_auc":
        return _safe(roc_auc_score, y, prob, multi_class="ovr", average="macro", labels=list(range(k)))
    if name == "mcc":
        return _safe(matthews_corrcoef, y, pred)
    short, _, met = name.partition("_")
    if short in short2idx:
        i = short2idx[short]
        return {"recall": rec[i], "precision": prec[i], "f1": f1[i], "specificity": spec[i],
                "auc": _safe(roc_auc_score, (y == i).astype(int), prob[:, i]),
                "ap": _safe(average_precision_score, (y == i).astype(int), prob[:, i])}.get(met, float("nan"))
    return float("nan")


def bootstrap_ci(y_true, prob, class_names, metrics: list[str], n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0, key_prefixes: dict | None = None) -> dict:
    """CI persentil bootstrap (resampling citra test) untuk metrik terpilih."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    pred = prob.argmax(1)
    k = len(class_names)
    key_prefixes = key_prefixes or {c: c.lower() for c in class_names}
    short2idx = {key_prefixes[c]: i for i, c in enumerate(class_names)}
    n = len(y_true)
    vals = {mname: [] for mname in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if len(np.unique(yb)) < k:
            continue
        for mname in metrics:
            vals[mname].append(_fast_metric(mname, yb, prob[idx], pred[idx], k, short2idx))
    out = {}
    for mname, v in vals.items():
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        out[mname] = ([float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))]
                      if len(v) else [float("nan")] * 2)
    return out


def flatten_metrics(m: dict, prefix: str = "") -> dict:
    """Ambil metrik skalar saja (untuk baris tabel)."""
    return {f"{prefix}{k}": v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
