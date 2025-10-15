"""
Run examples (project root):
  python single_classification_trees/train_model_tree.py --preset uni_fast
  python single_classification_trees/train_model_tree.py --preset uni_std
  python single_classification_trees/train_model_tree.py --preset bi_fast
  python single_classification_trees/train_model_tree.py --preset bi_std
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple, Dict
from datetime import datetime
import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ----------------------------- Data processing ----------------------------- #
def _iter_txt_files_negative(root: Path) -> Iterable[Tuple[str, str, int, str]]:
    label_dirs = [
        ("truthful_from_Web", "truthful"),
        ("deceptive_from_MTurk", "deceptive"),
    ]
    base_root = root / "negative_polarity"
    for dname, label in label_dirs:
        base = base_root / dname
        if not base.exists():
            continue
        for fdir in sorted(base.glob("fold*"), key=lambda p: p.name.lower()):
            digits = "".join(c for c in fdir.name if c.isdigit())
            if not digits:
                continue
            fold = int(digits)
            files = sorted((p for p in fdir.rglob("*.txt")), key=lambda p: str(p).lower())
            for txt in files:
                try:
                    text = txt.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    text = ""
                rel = str(txt.relative_to(root)).replace("\\", "/")
                yield (text, label, fold, rel)

def load_opspam_negative_only() -> pd.DataFrame:
    root = Path("op_spam_v1.4")
    if not root.exists():
        raise FileNotFoundError("Dataset folder './op_spam_v1.4' not found.")
    rows = [(text, label, fold, rel) for text, label, fold, rel in _iter_txt_files_negative(root)]
    df = pd.DataFrame(rows, columns=["text", "label", "fold", "path"])
    if df.empty:
        raise RuntimeError("No NEGATIVE data found under './op_spam_v1.4'.")
    df["text"] = df["text"].fillna("").astype(str)
    # Global stable order (fold, label, path) to match other scripts
    df = df.sort_values(["fold", "label", "path"]).reset_index(drop=True)
    return df

# ----------------------------- Save ----------------------------- #
def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

def save_metrics_summary(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rows = [("accuracy", acc), ("precision", prec), ("recall", rec), ("f1", f1)]
    pd.DataFrame(rows, columns=["metric", " value"]).to_csv(out_dir / "metrics.csv", index=False)
    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}

def save_metrics_full(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path):
    target_names = ["truthful(0)", "deceptive(1)"]
    rpt = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    for k, v in rpt.items():
        if isinstance(v, dict) and "support" in v:
            v["support"] = int(v["support"])  # force integer support
    pd.DataFrame(rpt).T.to_csv(out_dir / "metrics_full.csv")

def save_predictions(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path):
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out_dir / "predictions_fold5.csv", index=False)

def extract_top_terms(pipe: Pipeline, top_k: int = 50) -> pd.DataFrame:
    tfidf: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: DecisionTreeClassifier = pipe.named_steps["clf"]
    imp = getattr(clf, "feature_importances_", None)
    if imp is None or len(imp) == 0:
        return pd.DataFrame(columns=["feature", "importance"])
    terms = tfidf.get_feature_names_out() if hasattr(tfidf, "get_feature_names_out") else tfidf.get_feature_names()
    imp = np.asarray(imp)
    idx = np.argsort(imp)[::-1][: min(top_k, len(imp))]
    rows = [{"feature": str(terms[i]), "importance": float(imp[i])} for i in idx if imp[i] > 0]
    return pd.DataFrame(rows)

def save_params_csv(best_params: Dict, out_dir: Path):
    rows = [{"param": k, "value": v} for k, v in best_params.items()]
    pd.DataFrame(rows).to_csv(out_dir / "dt_params.csv", index=False)

# ----------------------------- Main ----------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Single Classification Tree (NEGATIVE only, RF-style, truthful=0).")
    parser.add_argument("--preset", type=str, required=True,
                        choices=["uni_fast", "uni_std", "bi_fast", "bi_std"],
                        help="Choose one of {uni_fast, uni_std, bi_fast, bi_std}.")
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    PRESET = {
        "uni_fast": ("uni", "fast"),
        "uni_std":  ("uni", "std"),
        "bi_fast":  ("uni+bi", "fast"),
        "bi_std":   ("uni+bi", "std"),
    }
    ngrams, mode = PRESET[args.preset]
    gram_tag = "uni" if ngrams == "uni" else "bi"

    out_dir = Path("single_classification_trees", f"results_{gram_tag}_{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_opspam_negative_only()

    df_train = df[df["fold"].isin([1, 2, 3, 4])].reset_index(drop=True)
    df_test = df[df["fold"] == 5].sort_values(["label", "path"]).reset_index(drop=True)

    y_map = {"truthful": 0, "deceptive": 1}
    y_train = df_train["label"].map(y_map).to_numpy()
    y_test = df_test["label"].map(y_map).to_numpy()
    X_train = df_train["text"].astype(str)
    X_test = df_test["text"].astype(str)

    ngram_range = (1, 1) if ngrams == "uni" else (1, 2)
    tfidf = TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=ngram_range, min_df=2, max_df=0.95, sublinear_tf=True
    )

    pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", DecisionTreeClassifier(random_state=args.seed)),
    ])

    # Hyperparam grid from mode
    if mode == "fast":
        param_grid = {
            "clf__criterion": ["gini"],
            "clf__max_depth": [None, 20],
            "clf__min_samples_leaf": [1, 5],
            "clf__ccp_alpha": [0.0, 1e-4],
        }
    else:
        param_grid = {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [None, 10, 20, 50],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 5, 10],
            "clf__ccp_alpha": [0.0, 1e-4, 5e-4, 1e-3],
        }

    skf = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
    grid = GridSearchCV(
        estimator=pipe, param_grid=param_grid, scoring="f1_macro",
        cv=skf, n_jobs=args.n_jobs, verbose=1, refit=True, return_train_score=False
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    train_time = time.time() - t0

    best_pipe: Pipeline = grid.best_estimator_
    best_params = grid.best_params_
    best_cv = float(grid.best_score_)

    y_pred = best_pipe.predict(X_test)
    save_confusion_matrix(y_test, y_pred, out_dir)
    metrics = save_metrics_summary(y_test, y_pred, out_dir)
    save_metrics_full(y_test, y_pred, out_dir)
    save_predictions(y_test, y_pred, out_dir)
    extract_top_terms(best_pipe, top_k=50).to_csv(out_dir / "top_features.csv", index=False)
    save_params_csv(best_params, out_dir)

    # keep json + run_summary
    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_params": best_params,
            "best_cv_f1_macro": round(best_cv, 6),
            "preset": args.preset,
            "ngrams": ngrams,
            "mode": mode,
            "seed": args.seed,
            "cv_splits": args.cv_splits,
            "train_time_sec": round(train_time, 3),
            "data_root": str(Path("op_spam_v1.4").resolve()),
            "polarity": "negative_only",
            "label_mapping": {"truthful": 0, "deceptive": 1}
        }, f, indent=2)

    pd.DataFrame([{
        "model": "DecisionTree",
        "preset": args.preset,
        "ngrams": ngrams,
        "mode": mode,
        "cv_f1_macro": round(best_cv, 6),
        "test_accuracy": round(metrics["accuracy"], 6),
        "test_precision_macro": round(metrics["precision_macro"], 6),
        "test_recall_macro": round(metrics["recall_macro"], 6),
        "test_f1_macro": round(metrics["f1_macro"], 6),
        "best_params": json.dumps(best_params),
        "train_time_sec": round(train_time, 3),
        "out_dir": str(out_dir.resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "polarity": "negative_only",
        "label_mapping": "truthful=0, deceptive=1"
    }]).to_csv(out_dir / "run_summary.csv", index=False)

    print("\n=== Training complete (NEGATIVE only; truthful=0, deceptive=1) ===")
    print(json.dumps({
        "preset": args.preset,
        "best_params": best_params,
        "best_cv_f1_macro": round(best_cv, 6),
        "test_metrics": metrics,
        "out_dir": str(out_dir.resolve())
    }, indent=2))

if __name__ == "__main__":
    main()
