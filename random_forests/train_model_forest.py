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


# Data processing
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


# Training
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

    acc = metrics.accuracy_score(y_te, y_pred)
    prec = metrics.precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec = metrics.recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1w = metrics.f1_score(y_te, y_pred, average="weighted", zero_division=0)

    pd.DataFrame(
        [("accuracy", acc), ("precision", prec), ("recall", rec), ("f1", f1w)],
        columns=["metric", "value"],
    ).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8")

    rf: RandomForestClassifier = best.named_steps["rf"]
    sel: SelectPercentile = best.named_steps["sel"]

    n_feat = int(getattr(rf, "n_features_in_", len(getattr(sel, "get_support", lambda **_: [])(indices=True))))

    max_feat_param = rf.max_features
    if isinstance(max_feat_param, str):
        if max_feat_param == "sqrt":
            mtry = int(np.sqrt(n_feat))
        elif max_feat_param == "log2":
            mtry = int(np.log2(n_feat))
        else:
            mtry = n_feat
    elif isinstance(max_feat_param, float):
        if 0.0 < max_feat_param <= 1.0:
            mtry = int(np.ceil(max_feat_param * n_feat))
        else:
            mtry = int(max_feat_param)
    elif isinstance(max_feat_param, (int, np.integer)) and max_feat_param > 0:
        mtry = int(max_feat_param)
    elif max_feat_param is None:
        mtry = n_feat
    else:
        mtry = n_feat

    mtry = max(1, min(n_feat, mtry))

    sel_percentile = getattr(sel, "percentile", None)

    pd.DataFrame(
        [
            ("n_trees", rf.n_estimators),
            ("max_features_rule", str(rf.max_features)),
            ("n_features_after_selection", n_feat),
            ("mtry_count", mtry),
            ("select_percentile", sel_percentile),
        ],
        columns=["param", "value"],
    ).to_csv(out_dir / "rf_params.csv", index=False, encoding="utf-8")

    print(f"[RF] n_trees={rf.n_estimators} | max_features_rule={rf.max_features} "
          f"| n_features_after_selection={n_feat} | mtry_count={mtry} | select_percentile={sel_percentile}")

    # metrics_full.csv
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

    # confusion matrix
    cm = metrics.confusion_matrix(y_te, y_pred, labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        out_dir / "confusion_matrix.csv", encoding="utf-8"
    )

    # top terms per class (two columns like: top_deceptive_terms, top_truthful_terms)
    vec: CountVectorizer = best.named_steps["vec"]
    sel: SelectPercentile = best.named_steps["sel"]

    feature_names = np.array(vec.get_feature_names_out())
    kept_names = feature_names[sel.get_support(indices=True)]

    Xtr_vec = vec.transform(X_tr)
    Xtr_sel = sel.transform(Xtr_vec)  # (n_train, n_kept)
    Xtr_bin = (Xtr_sel > 0).astype(int)

    idx_tru = np.where(y_tr == 0)[0]
    idx_dec = np.where(y_tr == 1)[0]
    p_truth = np.asarray(Xtr_bin[idx_tru].mean(axis=0)).ravel()
    p_decep = np.asarray(Xtr_bin[idx_dec].mean(axis=0)).ravel()

    dec_rank = np.argsort(-(p_decep - p_truth))
    tru_rank = np.argsort(-(p_truth - p_decep))

    top_k = 30
    dec_terms = kept_names[dec_rank][:top_k]
    tru_terms = kept_names[tru_rank][:top_k]

    L = max(len(dec_terms), len(tru_terms))
    dec_terms = list(dec_terms) + [""] * (L - len(dec_terms))
    tru_terms = list(tru_terms) + [""] * (L - len(tru_terms))
    pd.DataFrame(
        {"top_deceptive_terms": dec_terms, "top_truthful_terms": tru_terms}
    ).to_csv(out_dir / "top_features.csv", index=False, encoding="utf-8")

    print(f"[OK] Files saved to: {out_dir.resolve()}")

# CLI
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
