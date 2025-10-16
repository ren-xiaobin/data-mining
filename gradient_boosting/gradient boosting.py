#export data, define model, run model, print results
#run gradient boosting_fixed.py to save models and results to files

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


#train and evaluate a gradient boosting model with hyperparameter tuning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score




to_dense = FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)

def run_gb(Xtr, Xte, ytr, yte):
    pipe = Pipeline([
        ('sel', SelectKBest(chi2)),
        ('dense', to_dense),
        ('gb', GradientBoostingClassifier(random_state=0, subsample=0.7, max_features=0.7))
    ])
    grid = {
        'sel__k': [2000, 4000, 'all'],
        'gb__n_estimators': [200, 400, 800],
        'gb__learning_rate': [0.05, 0.08,0.1],
        'gb__max_depth': [2, 3, 4],
    }
    rs = RandomizedSearchCV(pipe, grid, n_iter=12, scoring='f1', cv=5, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    pred = rs.predict(Xte)

    acc = accuracy_score(yte, pred)
    prec, rec, f1, supp = precision_recall_fscore_support(yte, pred, labels=[0,1], zero_division=0)
    f1_macro = f1_score(yte, pred, average='macro')

    out = {
        'acc': acc, 'f1_macro': f1_macro, 'params': rs.best_params_,
        'truth_prec': prec[0], 'truth_rec': rec[0], 'truth_f1': f1[0], 'truth_support': supp[0],
        'dec_prec': prec[1], 'dec_rec': rec[1], 'dec_f1': f1[1], 'dec_support': supp[1],
    }
    return out

for name, (Xtr, Xte) in [('UNI', (X_train_uni, X_test_uni)), ('UNI+BI', (X_train_bi, X_test_bi))]:
    res = run_gb(Xtr, Xte, y_train, y_test)
    print(
        f"GB {name} | acc={res['acc']:.3f}  macroF1={res['f1_macro']:.3f}  {res['params']}\n"
        f"  truthful (0):  P={res['truth_prec']:.3f} R={res['truth_rec']:.3f} F1={res['truth_f1']:.3f}  n={res['truth_support']}\n"
        f"  deceptive (1): P={res['dec_prec']:.3f} R={res['dec_rec']:.3f} F1={res['dec_f1']:.3f}  n={res['dec_support']}\n"
    )
