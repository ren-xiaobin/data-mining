import pandas as pd, numpy as np
from math import comb

paths = {
    # Linear Model predictions
    "MNB_uni": "../naive_bayes/results_mnb_unigram/predictions_fold5.csv",
    "MNB_bi":  "../naive_bayes/results_mnb_bigram/predictions_fold5.csv",      
    "LR_uni":  "../logistic_regression/results_lr_l1_unigram/predictions_fold5.csv",
    "LR_bi":   "../logistic_regression/results_lr_l1_bigram/predictions_fold5.csv", 

    # Uncomment these two lines to test TF-IDF for logistic regression
    # "LR_uni_tf_idf": "../logistic_regression/results_lr_l1_compare/tfidf_unigram/predictions_fold5.csv",
    # "LR_bi_tf_idf": "../logistic_regression/results_lr_l1_compare/tfidf_bigram/predictions_fold5.csv",

    # Random forest and Gradient Boosting
    "RF_uni":  "../random_forests/results_uni_std/predictions_fold5.csv",
    "RF_bi":   "../random_forests/results_bi_std/predictions_fold5.csv",
    "GB_uni":  "../gradient_boosting/results_gb_unigram/gb_uni_predictions.csv",   
    "GB_bi":   "../gradient_boosting/results_gb_bigram/gb_uni_plus_bi_predictions.csv",
}

def load_preds(path):
    df = pd.read_csv(path)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    return y_true, y_pred

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return (y_true == y_pred).mean()

def mcnemar_exact(y_true, y_a, y_b):
    y_true = np.asarray(y_true); a = np.asarray(y_a); b = np.asarray(y_b)
    b01 = np.sum((a==y_true) & (b!=y_true))  # A correct B wrong
    b10 = np.sum((a!=y_true) & (b==y_true))  # A wrong B correct
    n = b01 + b10
    if n == 0: 
        p = 1.0
    else:
        k = min(b01, b10)
        p = 2 * sum(comb(n, i) for i in range(0, k+1)) * (0.5 ** n)
        p = min(1.0, p)
    direction = "A>B" if b01 > b10 else ("B>A" if b10 > b01 else "tie")
    return dict(b01=b01, b10=b10, n=n, p_value=p, direction=direction)

# Read all models
models = {}
for name, p in list(paths.items()):
    try:
        y_true, y_pred = load_preds(p)
        models[name] = dict(path=p, y_true=y_true, y_pred=y_pred, acc=accuracy(y_true, y_pred))
    except Exception as e:
        print(f"[WARN] Skip {name}: {e}")

# Select the best linear model under same n-gram
def best_linear(ngram="uni"):
    cands = [f"MNB_{ngram}", f"LR_{ngram}"]
    cands = [m for m in cands if m in models]
    assert cands, f"No linear models found for {ngram}"
    best = max(cands, key=lambda m: models[m]["acc"])
    return best

# Batch compare
def compare_pairs(pairs, title):
    rows = []
    for A, B, tag in pairs:
        y_true = models[A]["y_true"]
        assert np.all(y_true == models[B]["y_true"]), "y_true misaligned"
        res = mcnemar_exact(y_true, models[A]["y_pred"], models[B]["y_pred"])
        dacc = models[A]["acc"] - models[B]["acc"]
        rows.append({
            "pair": f"{A} vs {B}",
            "tag": tag,
            "acc_A": models[A]["acc"],
            "acc_B": models[B]["acc"],
            "Delta_acc(A-B)": dacc,
            "b01(A(Y),B(N))": res["b01"], # A correct B wrong
            "b10(A(N),B(Y))": res["b10"], # A wrong  B correct
            "n": res["n"],
            "p_value": res["p_value"],
            "direction": res["direction"]
        })
    tbl = pd.DataFrame(rows).sort_values("p_value")
    print(f"\n=== {title} ===")
    print(tbl.to_string(index=False))
    return tbl

# Q1: MNB vs LR
pairs_q1 = []
if "MNB_uni" in models and "LR_uni" in models:
    pairs_q1.append(("MNB_uni","LR_uni","linear (uni)"))
if "MNB_bi" in models and "LR_bi" in models:
    pairs_q1.append(("MNB_bi","LR_bi","linear (bi)"))
tbl_q1 = compare_pairs(pairs_q1, "Q1: MNB vs L1-LR")

# Q2: RF/GB vs MNB (Best linear)
pairs_q2 = []
for ng in ["uni","bi"]:
    try:
        best_lin = best_linear(ng)
    except AssertionError:
        continue
    for ens in [f"RF_{ng}", f"GB_{ng}"]:
        if ens in models:
            pairs_q2.append((ens, best_lin, f"{ng}: ensemble vs best-linear({best_lin})"))
tbl_q2 = compare_pairs(pairs_q2, "Q2: Ensembles vs Best Linear")

# Q3: bigram better or not
pairs_q3 = []
for alg in ["MNB","LR","RF","GB"]:
    a = f"{alg}_bi"; b = f"{alg}_uni"
    if a in models and b in models:
        pairs_q3.append((a, b, f"{alg}: bi vs uni"))
tbl_q3 = compare_pairs(pairs_q3, "Q3: Bigram effect within algorithm")

# Holm-Bonferroni correction p-value
def holm_bonferroni(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    for rank, idx in enumerate(order, start=1):
        adj[idx] = min(1.0, (m - rank + 1) * pvals[idx])
    return adj

for name, tbl in [("Q1", tbl_q1), ("Q2", tbl_q2), ("Q3", tbl_q3)]:
    if tbl is None or tbl.empty: 
        continue
    ps = tbl["p_value"].to_numpy(float)
    tbl["p_adj_Holm"] = holm_bonferroni(ps)
    print(f"\n>>> {name} with Holm-Bonferroni correction:")
    print(tbl[["pair","tag","p_value","p_adj_Holm","Delta_acc(A-B)","direction"]].to_string(index=False))
