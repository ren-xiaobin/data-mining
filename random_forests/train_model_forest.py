"""
--------------------------------------------------------------
How to run (choose ONE of the four presets)
--------------------------------------------------------------
# Unigram (fast)
python random_forests/train_model_forest.py --preset uni_fast

# Unigram (standard)
python random_forests/train_model_forest.py --preset uni_std

# Bigram (fast)
python random_forests/train_model_forest.py --preset bi_fast

# Bigram (standard)
python random_forests/train_model_forest.py --preset bi_std
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


# ----------------------------- Data processing ----------------------------- #
def _read_text(p: Path) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            pass
    return p.read_text(encoding="utf-8", errors="ignore")


def _collect_from_class_dir(folder: Path, label: int):
    texts, labels, folds = [], [], []
    for p in folder.rglob("*.txt"):
        m = re.search(r"fold(\d+)", str(p).replace("\\", "/"))
        if not m:
            continue
        texts.append(_read_text(p))
        labels.append(label)
        folds.append(int(m.group(1)))
    return texts, labels, folds


def _find_class_dirs(neg_root: Path) -> tuple[Path, Path]:
    """
    Auto-detect class folders under negative_polarity:
      - deceptive_from_*  -> label 1
      - truthful_from_*   -> label 0
    """
    if not neg_root.exists():
        raise FileNotFoundError(f"negative_polarity not found: {neg_root}")

    dece = sorted([p for p in neg_root.iterdir() if p.is_dir() and p.name.startswith("deceptive_from_")])
    tru = sorted([p for p in neg_root.iterdir() if p.is_dir() and p.name.startswith("truthful_from_")])

    if not dece:
        raise FileNotFoundError(f"No 'deceptive_from_*' under {neg_root}")
    if not tru:
        raise FileNotFoundError(f"No 'truthful_from_*' under {neg_root}")

    return dece[0], tru[0]


def load_negative(data_root: Path):
    neg_root = data_root if data_root.name == "negative_polarity" else (data_root / "negative_polarity")
    dec_dir, tru_dir = _find_class_dirs(neg_root)

    X1, y1, f1 = _collect_from_class_dir(dec_dir, label=1)  # deceptive
    X0, y0, f0 = _collect_from_class_dir(tru_dir, label=0)  # truthful

    X = np.array(X1 + X0, dtype=object)
    y = np.array(y1 + y0, dtype=int)
    folds = np.array(f1 + f0, dtype=int)

    tr_idx = folds != 5
    te_idx = folds == 5

    print(
        f"[Data] train={tr_idx.sum()} docs | test={te_idx.sum()} docs "
        f"(truthful_test={(y[te_idx]==0).sum()}, deceptive_test={(y[te_idx]==1).sum()})"
    )
    return X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]


# ------------------------------- Training ------------------------------- #
def train_and_eval(args):
    project_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root) if args.data_root else (project_root / "op_spam_v1.4" / "negative_polarity")

    preset = args.preset.lower()
    if preset not in {"uni_fast", "uni_std", "bi_fast", "bi_std"}:
        raise ValueError("Unknown preset. Choose from: uni_fast, uni_std, bi_fast, bi_std")

    ngram_range = (1, 1) if preset.startswith("uni") else (1, 2)
    is_fast = preset.endswith("fast")

    out_dir = project_root / "random_forests" / f"results_{preset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, X_te, y_te = load_negative(data_root)

    pipe = Pipeline(
        steps=[
            ("vec", CountVectorizer(lowercase=True, stop_words="english", ngram_range=ngram_range)),
            ("sel", SelectPercentile(score_func=chi2)),
            ("rf", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=4)),
        ]
    )

    grid_fast = {
        "vec__min_df": [2],
        "sel__percentile": [75, 100],
        "rf__n_estimators": [300],
        "rf__max_depth": [None, 30],
        "rf__max_features": ["sqrt"],
        "rf__min_samples_leaf": [1, 2],
    }
    grid_std = {
        "vec__min_df": [1, 2, 3],
        "sel__percentile": [100, 90, 75, 50],
        "rf__n_estimators": [300, 500, 800],
        "rf__max_depth": [None, 20, 40],
        "rf__max_features": ["sqrt", "log2"],
        "rf__min_samples_leaf": [1, 2, 4],
    }
    param_grid = grid_fast if is_fast else grid_std

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=4,
        refit=True,
        verbose=1,
    )

    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    print("Best params:", search.best_params_, "| CV score:", search.best_score_)

    y_pred = best.predict(X_te)

    pd.DataFrame({"y_true": y_te, "y_pred": y_pred}).to_csv(
        out_dir / "predictions_fold5.csv", index=False, encoding="utf-8"
    )

    # ----- metrics.csv (two columns: metric,value; weighted averages) -----
    acc = metrics.accuracy_score(y_te, y_pred)
    prec = metrics.precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec = metrics.recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1w = metrics.f1_score(y_te, y_pred, average="weighted", zero_division=0)

    pd.DataFrame(
        [("accuracy", acc), ("precision", prec), ("recall", rec), ("f1", f1w)],
        columns=["metric", "value"],
    ).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8")

    # ----- metrics_full.csv -----
    rep = metrics.classification_report(
        y_te,
        y_pred,
        output_dict=True,
        target_names=["truthful(0)", "deceptive(1)"],
        zero_division=0,
        digits=4,
    )

    rows = []
    for name in ["truthful(0)", "deceptive(1)"]:
        r = rep[name]
        rows.append([name, r["precision"], r["recall"], r["f1-score"], int(round(r["support"]))])

    acc_scalar = float(rep["accuracy"])
    rows.append(["accuracy", acc_scalar, acc_scalar, acc_scalar, ""])

    total_support = int(len(y_te))
    for agg in ["macro avg", "weighted avg"]:
        a = rep[agg]
        rows.append([agg, a["precision"], a["recall"], a["f1-score"], total_support])

    df_full = pd.DataFrame(rows, columns=["", "precision", "recall", "f1-score", "support"])
    for col in ["precision", "recall", "f1-score"]:
        df_full[col] = pd.to_numeric(df_full[col], errors="coerce").round(4)
    df_full["support"] = df_full["support"].astype("string")
    df_full.to_csv(out_dir / "metrics_full.csv", index=False, encoding="utf-8")

    # ----- confusion matrix -----
    cm = metrics.confusion_matrix(y_te, y_pred, labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        out_dir / "confusion_matrix.csv", encoding="utf-8"
    )

    # ----- top features by RF importance AFTER selection -----
    vec: CountVectorizer = best.named_steps["vec"]
    sel: SelectPercentile = best.named_steps["sel"]
    rf: RandomForestClassifier = best.named_steps["rf"]

    feature_names = np.array(vec.get_feature_names_out())
    kept_idx = sel.get_support(indices=True)
    kept_names = feature_names[kept_idx]

    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    top_k = min(30, len(importances))
    pd.DataFrame(
        {"feature": kept_names[order][:top_k], "importance": importances[order][:top_k]}
    ).to_csv(out_dir / "top_features.csv", index=False, encoding="utf-8")

    print(f"[OK] Files saved to: {out_dir.resolve()}")


# ----------------------------------- CLI ----------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=["uni_fast", "uni_std", "bi_fast", "bi_std"],
        default="uni_fast",
        help="Pick one of the four simple presets."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to '.../op_spam_v1.4' or directly to '.../negative_polarity'."
    )
    args = parser.parse_args()
    train_and_eval(args)


if __name__ == "__main__":
    main()
