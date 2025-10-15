import os, glob
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

DATA_ROOT = "../op_spam_v1.4/negative_polarity"
OUT_DIR   = "results_lr_l1_bigram"
NGRAMS    = (1, 2)
SEED      = 42
os.makedirs(OUT_DIR, exist_ok=True)

def read_text(fp):
    for enc in ("utf-8","latin-1"):
        try: return Path(fp).read_text(encoding=enc)
        except Exception: pass
    return Path(fp).read_text(errors="ignore")

# Collect data
paths, y, folds = [], [], []
for label_dir, lab in [("deceptive_from_MTurk",1), ("truthful_from_Web",0)]:
    for fold in range(1,6):
        for fp in sorted(glob.glob(os.path.join(DATA_ROOT, label_dir, f"fold{fold}", "*.txt"))):
            paths.append(fp); y.append(lab); folds.append(fold)
texts = [read_text(p) for p in paths]
y = np.array(y); folds = np.array(folds)

X_train = [t for t,f in zip(texts, folds) if f != 5]
y_train = y[folds!=5]
X_test  = [t for t,f in zip(texts, folds) if f == 5]
y_test  = y[folds==5]


pipe = Pipeline([
    ("vec", CountVectorizer(lowercase=True, stop_words="english", ngram_range=NGRAMS)),
    ("sel", "passthrough"),
    ("clf", LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=SEED)),
])

# Grid (2 configs: feature selection or not)
param_grid = [
    {"vec__min_df":[1,2,3], "sel":["passthrough"], "clf__C":[0.01,0.1,1,10]},
    {"vec__min_df":[1,2,3], "sel":[SelectPercentile(score_func=chi2)],
     "sel__percentile":[100,90,75,50,25], "clf__C":[0.01,0.1,1,10]},
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True, verbose=1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Best params:", grid.best_params_, "CV F1:", grid.best_score_)

# Test set
y_pred = best.predict(X_test)
proba = best.predict_proba(X_test)[:, list(best.named_steps["clf"].classes_).index(1)]

report = metrics.classification_report(y_test, y_pred, target_names=["truthful(0)","deceptive(1)"], output_dict=True)
cm = metrics.confusion_matrix(y_test, y_pred, labels=[0,1])

metrics_simple = pd.DataFrame({
    "metric": ["accuracy","precision","recall","f1"],
    "value": [metrics.accuracy_score(y_test, y_pred),
              metrics.precision_score(y_test, y_pred),
              metrics.recall_score(y_test, y_pred),
              metrics.f1_score(y_test, y_pred)]
})

# Save metrics
pd.DataFrame(report).T.to_csv(os.path.join(OUT_DIR, "metrics_full.csv"))
pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))
metrics_simple.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
pd.DataFrame({"y_true":y_test, "y_pred":y_pred, "y_prob":proba}).to_csv(os.path.join(OUT_DIR, "predictions_fold5.csv"), index=False)

# Save top terms
vec = best.named_steps["vec"]; clf = best.named_steps["clf"]
import numpy as np
feature_names = np.array(vec.get_feature_names_out()); coef = clf.coef_[0]
k = 25
top_pos_idx = np.argsort(-coef)[:k]; top_neg_idx = np.argsort(coef)[:k]
pd.DataFrame({
    "top_deceptive_terms": feature_names[top_pos_idx],
    "coef_for_deceptive":  coef[top_pos_idx],
    "top_truthful_terms":  feature_names[top_neg_idx],
    "coef_for_truthful":   coef[top_neg_idx]
}).to_csv(os.path.join(OUT_DIR, "top_terms_lr.csv"), index=False)

print("Done. Files saved to:", OUT_DIR)
