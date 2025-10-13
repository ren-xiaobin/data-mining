import os, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

DATA_ROOT = "../op_spam_v1.4/negative_polarity"
OUT_DIR = "results_mnb_unigram"; os.makedirs(OUT_DIR, exist_ok=True)

def read_text(fp):
    for enc in ("utf-8","latin-1"):
        try: return Path(fp).read_text(encoding=enc)
        except Exception: pass
    return Path(fp).read_text(errors="ignore")


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
    ("vec", CountVectorizer(lowercase=True, stop_words="english", ngram_range=(1,1))), 
    ("sel", SelectKBest(score_func=chi2, k='all')),
    ("clf", MultinomialNB())
])

param_grid = {
    "clf__alpha": [0.1, 0.5, 1.0, 2.0],  
    "clf__fit_prior": [True, False],
    "vec__min_df": [1,2],
    "sel__k": ['all', 2000, 4000, 8000],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True, verbose=1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
print("Best params:", grid.best_params_, "CV F1:", grid.best_score_)

y_pred = best.predict(X_test)
print(metrics.classification_report(y_test, y_pred, target_names=["truthful(0)","deceptive(1)"]))
print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))

# Save metrics
pd.DataFrame({"y_true":y_test, "y_pred":y_pred}).to_csv(os.path.join(OUT_DIR,"predictions_fold5.csv"), index=False)
pd.DataFrame({
    "metric":["accuracy","precision","recall","f1"],
    "value":[metrics.accuracy_score(y_test, y_pred),
             metrics.precision_score(y_test, y_pred),
             metrics.recall_score(y_test, y_pred),
             metrics.f1_score(y_test, y_pred)]
}).to_csv(os.path.join(OUT_DIR,"metrics.csv"), index=False)

# Save top terms
vec = best.named_steps["vec"]; clf = best.named_steps["clf"]
import numpy as np
fn = np.array(vec.get_feature_names_out())
log_ratio = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
top_fake = fn[np.argsort(-log_ratio)[:20]]
top_true = fn[np.argsort(log_ratio)[:20]]
pd.DataFrame({"top_deceptive_terms":top_fake, "top_truthful_terms":top_true}).to_csv(
    os.path.join(OUT_DIR,"top_terms.csv"), index=False)
