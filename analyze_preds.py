"""
Post-hoc statistical analysis of test-set predictions saved by test.py.

Reads two preds.npz files (CE and SupCon, same backbone, same test set) and
reports, per class (one-vs-rest):
  - AUC, precision, recall with bootstrap 95% CIs
  - DeLong's test for the AUC difference between CE and SupCon

No retraining or re-inference required — this only reads the predictions/
labels/logits/ids that test.py already saved via --save_preds.

Usage:
    python analyze_preds.py --backbone resnet18
    python analyze_preds.py --ce_preds path/to/ce_preds.npz --supcon_preds path/to/supcon_preds.npz
"""

import argparse
import json

import numpy as np


# ── DeLong's test for paired AUC comparison ──────────────────────────────────
# Standard fast (O(N log N)) implementation, per Sun & Xu (2014). Compares two
# classifiers' AUCs on the SAME ordered set of binary labels.

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(scores, label_1_count):
    """scores: (n_models, n_samples), positives first `label_1_count` columns."""
    m = label_1_count
    n = scores.shape[1] - m
    positive = scores[:, :m]
    negative = scores[:, m:]
    k = scores.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(scores[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(labels, scores_a, scores_b):
    """Returns (auc_a, auc_b, p_value) for the two-sided DeLong test."""
    order = np.argsort(-labels, kind="stable")  # positives (1) first
    labels_sorted = labels[order]
    scores = np.vstack([scores_a[order], scores_b[order]])
    label_1_count = int(labels_sorted.sum())
    if label_1_count == 0 or label_1_count == len(labels_sorted):
        return float("nan"), float("nan"), float("nan")

    aucs, cov = _fast_delong(scores, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], float("nan")
    z = diff / np.sqrt(var)
    # two-sided p-value from the standard normal
    from math import erf
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))
    return aucs[0], aucs[1], p


# ── Bootstrap CIs ─────────────────────────────────────────────────────────────

def bootstrap_ci(labels, scores, preds, n_boot=2000, seed=0):
    """95% CI (percentile method) for AUC, precision, and recall of one class."""
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs, precs, recs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y, s, p = labels[idx], scores[idx], preds[idx]
        if y.sum() == 0 or y.sum() == n:
            continue  # skip resamples with only one class present
        aucs.append(roc_auc_score(y, s))
        precs.append(precision_score(y, p, zero_division=0))
        recs.append(recall_score(y, p, zero_division=0))

    def ci(vals):
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return {"auc_ci": ci(aucs), "precision_ci": ci(precs), "recall_ci": ci(recs)}


# ── Main analysis ─────────────────────────────────────────────────────────────

def load_run(path):
    data = np.load(path)
    run = {
        "labels": data["labels"],
        "preds": data["predictions"],
        "logits": data["logits"],
    }
    run["ids"] = data["ids"] if "ids" in data.files else None
    return run


def align_by_id(run_a, run_b):
    """Reorder run_b to match run_a's id order; error if the id sets differ."""
    if set(run_a["ids"]) != set(run_b["ids"]):
        raise ValueError(
            "CE and SupCon preds.npz files don't cover the same test-set ids — "
            "can't run a paired comparison (DeLong's test) between them."
        )
    order = {id_: i for i, id_ in enumerate(run_b["ids"])}
    reindex = np.array([order[id_] for id_ in run_a["ids"]])
    return {k: (v[reindex] if v is not None else None) for k, v in run_b.items()}


def analyze(ce_path, supcon_path, n_boot, seed):
    ce = load_run(ce_path)
    supcon = load_run(supcon_path)

    # DeLong's test needs both runs evaluated on the exact same ordered subjects.
    # Bootstrap CIs don't — each run's CI only resamples within its own data.
    can_pair = ce["ids"] is not None and supcon["ids"] is not None
    if can_pair:
        supcon = align_by_id(ce, supcon)
    else:
        print("[Warning] ids missing from one or both preds.npz files — skipping "
              "DeLong's paired test. Bootstrap CIs below are still valid (they "
              "don't require pairing). Rerun test.py to get ids and enable DeLong's test.\n")

    num_classes = ce["logits"].shape[1]
    class_names = {0: "Alzheimer's Disease", 1: "Mild Cognitive Impairment", 2: "Normal Cognition"}

    results = {}
    for c in range(num_classes):
        name = class_names.get(c, f"Class {c}")
        y_ce = (ce["labels"] == c).astype(int)
        y_supcon = (supcon["labels"] == c).astype(int)
        ce_scores = ce["logits"][:, c]
        supcon_scores = supcon["logits"][:, c]
        ce_preds_c = (ce["preds"] == c).astype(int)
        supcon_preds_c = (supcon["preds"] == c).astype(int)

        ce_ci = bootstrap_ci(y_ce, ce_scores, ce_preds_c, n_boot, seed)
        supcon_ci = bootstrap_ci(y_supcon, supcon_scores, supcon_preds_c, n_boot, seed)

        if can_pair:
            auc_ce, auc_supcon, p_value = delong_roc_test(y_ce, ce_scores, supcon_scores)
        else:
            from sklearn.metrics import roc_auc_score
            auc_ce = roc_auc_score(y_ce, ce_scores)
            auc_supcon = roc_auc_score(y_supcon, supcon_scores)
            p_value = None

        results[name] = {
            "ce": {"auc": auc_ce, **ce_ci},
            "supcon": {"auc": auc_supcon, **supcon_ci},
            "delong_p_value": p_value,
        }

    return results


def print_report(results):
    for name, r in results.items():
        print(f"\n=== {name} ===")
        ce, sc = r["ce"], r["supcon"]
        print(f"  CE:     AUC = {ce['auc']:.4f}  95% CI {ce['auc_ci']}   "
              f"recall 95% CI {ce['recall_ci']}")
        print(f"  SupCon: AUC = {sc['auc']:.4f}  95% CI {sc['auc_ci']}   "
              f"recall 95% CI {sc['recall_ci']}")
        p = r["delong_p_value"]
        p_str = f"{p:.4f}" if p is not None else "N/A (no ids — rerun test.py to pair runs)"
        print(f"  DeLong's test (CE vs SupCon AUC): p = {p_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap CIs + DeLong's test on test.py preds.npz output")
    parser.add_argument("--backbone", type=str, default=None,
                         help="Shortcut: use results/<backbone>/{ce,contrastive}/preds.npz")
    parser.add_argument("--ce_preds", type=str, default=None)
    parser.add_argument("--supcon_preds", type=str, default=None)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None, help="Optional path to save results as JSON")
    args = parser.parse_args()

    ce_path = args.ce_preds or f"results/{args.backbone}/ce/preds.npz"
    supcon_path = args.supcon_preds or f"results/{args.backbone}/contrastive/preds.npz"
    if not args.ce_preds and not args.supcon_preds and not args.backbone:
        parser.error("Pass --backbone, or both --ce_preds and --supcon_preds")

    results = analyze(ce_path, supcon_path, args.n_bootstrap, args.seed)
    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")
