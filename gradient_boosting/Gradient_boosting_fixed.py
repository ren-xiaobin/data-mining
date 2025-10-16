import zipfile, re, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#read from zip data file, this zip file has to be in the same directory as this script
zf = zipfile.ZipFile('op_spam_v1.4.zip')
names = [n for n in zf.namelist() if '/negative_polarity/' in n and n.endswith('.txt') and 'readme' not in n.lower()]

label = lambda n: ('deceptive' if '/deceptive_' in n.lower() else 'truthful')
fold  = lambda n: int(re.search(r'/fold(\d)/', n.lower()).group(1))

df = pd.DataFrame([{'path':n,'text':zf.read(n).decode('utf-8','replace'),
                    'label':label(n),'fold':fold(n)} for n in names])
df = df[['text','label','fold']]

train_df = df[df.fold.isin([1,2,3,4])].reset_index(drop=True)
test_df  = df[df.fold==5].reset_index(drop=True)

y_map   = {'truthful':0,'deceptive':1}
y_train = train_df.label.map(y_map).to_numpy()
y_test  = test_df.label.map(y_map).to_numpy()


uni = CountVectorizer(lowercase=True)
bi  = CountVectorizer(lowercase=True, ngram_range=(1,2))


X_train_uni = uni.fit_transform(train_df.text); X_test_uni = uni.transform(test_df.text)
X_train_bi  = bi.fit_transform(train_df.text);  X_test_bi  = bi.transform(test_df.text)

print(len(train_df), len(test_df), '|',
      X_train_uni.shape, X_test_uni.shape, '|',
      X_train_bi.shape,  X_test_bi.shape)

#train and evaluate a gradient boosting model with fixed hyperparameters
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score, confusion_matrix, classification_report)
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
import numpy as np, joblib, json, pandas as pd


class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None): return X.toarray() if sparse.issparse(X) else X


UNI_PARAMS   = {'sel__k': 4000, 'gb__n_estimators': 800, 'gb__max_depth': 2, 'gb__learning_rate': 0.05}
UNIBI_PARAMS = {'sel__k': 'all', 'gb__n_estimators': 800, 'gb__max_depth': 3, 'gb__learning_rate': 0.05}

def fit_eval_gb_fixed(Xtr, Xte, ytr, yte, params):
    pipe = Pipeline([
        ('sel',   SelectKBest(chi2, k=params['sel__k'])),
        ('dense', DenseTransformer()),
        ('gb',    GradientBoostingClassifier(
                    n_estimators=params['gb__n_estimators'],
                    learning_rate=params['gb__learning_rate'],
                    max_depth=params['gb__max_depth'],
                    subsample=0.7, max_features=0.7, random_state=0))
    ])
    model = pipe.fit(Xtr, ytr)
    pred  = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(yte, pred)
    prec, rec, f1, supp = precision_recall_fscore_support(yte, pred, labels=[0,1], zero_division=0)
    f1_macro = f1_score(yte, pred, average='macro')
    report = classification_report(yte, pred, labels=[0,1], target_names=['truthful','deceptive'],
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=[0,1])

    return {
        'model': model, 'pred': pred, 'proba': proba, 'best_params': params,
        'acc': float(acc), 'f1_macro': float(f1_macro),
        'truth_prec': float(prec[0]), 'truth_rec': float(rec[0]), 'truth_f1': float(f1[0]), 'truth_support': int(supp[0]),
        'dec_prec': float(prec[1]), 'dec_rec': float(rec[1]), 'dec_f1': float(f1[1]), 'dec_support': int(supp[1]),
        'report': report, 'cm': cm
    }

def save_run(tag, res, vec):
    # model
    joblib.dump(res['model'], f'gb_{tag}.pkl')

    # metrics
    with open(f'gb_{tag}_metrics.json','w') as f:
        json.dump({
            'acc': res['acc'], 'f1_macro': res['f1_macro'], 'best_params': res['best_params'],
            'truth': {'precision': res['truth_prec'], 'recall': res['truth_rec'], 'f1': res['truth_f1'], 'support': res['truth_support']},
            'deceptive': {'precision': res['dec_prec'], 'recall': res['dec_rec'], 'f1': res['dec_f1'], 'support': res['dec_support']}
        }, f, indent=2)
    with open(f'gb_{tag}_metrics_full.json','w') as f:
        json.dump(res['report'], f, indent=2)

    # confusion + predictions
    pd.DataFrame(res['cm'], index=['true_truthful','true_deceptive'],
                 columns=['pred_truthful','pred_deceptive']).to_csv(f'gb_{tag}_confusion.csv')
    pd.DataFrame({'y_true': y_test, 'y_pred': res['pred'], 'p_deceptive': res['proba']}).to_csv(
        f'gb_{tag}_predictions.csv', index=False)

    # ---------- Directional top terms (truthful vs deceptive) ----------
    sel = res['model'].named_steps['sel']
    gb  = res['model'].named_steps['gb']
    kept_idx = sel.get_support(indices=True)

    # terms kept by chi2 selection
    vocab = vec.get_feature_names_out()
    terms = vocab[kept_idx]
    imps  = gb.feature_importances_

    # Build selected test matrix to estimate direction on fold 5
    X_test_vec = vec.transform(test_df.text)
    X_test_sel = sel.transform(X_test_vec)
    if not sparse.issparse(X_test_sel):
        from scipy import sparse as sp
        X_test_sel = sp.csr_matrix(X_test_sel)

    # Binary presence matrix
    Xbin = X_test_sel.copy()
    Xbin.data[:] = 1

    P = res['proba']  # predicted P(deceptive) on fold 5, shape (N,)
    N = Xbin.shape[0]

    # Vectorized means: mean(p | present) - mean(p | absent)
    present_count = np.asarray(Xbin.sum(axis=0)).ravel()
    present_sum   = np.asarray(Xbin.T.dot(P)).ravel()
    total_sum     = float(P.sum())
    absent_count  = N - present_count
    absent_sum    = total_sum - present_sum

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_present = np.divide(present_sum, present_count, out=np.zeros_like(present_sum, dtype=float), where=present_count>0)
        mean_absent  = np.divide(absent_sum,  absent_count,  out=np.zeros_like(absent_sum,  dtype=float), where=absent_count>0)
        delta = mean_present - mean_absent  # >0 => feature presence raises P(deceptive)

    # Class-specific scores
    score_decep = imps * np.maximum(delta, 0.0)
    score_truth = imps * np.maximum(-delta, 0.0)

    topk = 50  # change if you want fewer/more
    decep_idx = np.argsort(score_decep)[-topk:][::-1]
    truth_idx = np.argsort(score_truth)[-topk:][::-1]

    with open(f'gb_{tag}_top_terms_deceptive.txt','w') as f:
        for i in decep_idx:
            if score_decep[i] <= 0: continue
            f.write(f"{terms[i]}\timp={imps[i]:.6f}\tdelta={delta[i]:.6f}\tscore={score_decep[i]:.6f}\n")

    with open(f'gb_{tag}_top_terms_truthful.txt','w') as f:
        for i in truth_idx:
            if score_truth[i] <= 0: continue
            # note: report positive delta-for-truth as -delta (magnitude toward truth)
            f.write(f"{terms[i]}\timp={imps[i]:.6f}\tdelta={-delta[i]:.6f}\tscore={score_truth[i]:.6f}\n")

# UNI (fixed)
res_uni = fit_eval_gb_fixed(X_train_uni, X_test_uni, y_train, y_test, UNI_PARAMS)
print(f"GB UNI   | acc={res_uni['acc']:.3f}  macroF1={res_uni['f1_macro']:.3f}  {res_uni['best_params']}\n"
      f"  truthful (0):  P={res_uni['truth_prec']:.3f} R={res_uni['truth_rec']:.3f} F1={res_uni['truth_f1']:.3f}  n={res_uni['truth_support']}\n"
      f"  deceptive (1): P={res_uni['dec_prec']:.3f} R={res_uni['dec_rec']:.3f} F1={res_uni['dec_f1']:.3f}  n={res_uni['dec_support']}\n")
save_run('uni', res_uni, uni)

# UNI+BI (fixed)
res_bi = fit_eval_gb_fixed(X_train_bi, X_test_bi, y_train, y_test, UNIBI_PARAMS)
print(f"GB UNI+BI| acc={res_bi['acc']:.3f}  macroF1={res_bi['f1_macro']:.3f}  {res_bi['best_params']}\n"
      f"  truthful (0):  P={res_bi['truth_prec']:.3f} R={res_bi['truth_rec']:.3f} F1={res_bi['truth_f1']:.3f}  n={res_bi['truth_support']}\n"
      f"  deceptive (1): P={res_bi['dec_prec']:.3f} R={res_bi['dec_rec']:.3f} F1={res_bi['dec_f1']:.3f}  n={res_bi['dec_support']}\n")
save_run('uni_plus_bi', res_bi, bi)

print("Saved: models (*.pkl), metrics (*.json), confusion (*.csv), predictions (*.csv), top terms (*.txt)")
