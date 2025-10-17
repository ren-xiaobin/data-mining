# Classification for the Detection of Opinion Spam

The project is organized into the following folders:

- `naive_bayes/`: Contains scripts for training and evaluating Multinomial Naive Bayes models.
- `logistic_regression/`: Contains scripts for training and evaluating Logistic Regression models, including a TF-IDF version.
- `single_classification_trees/`: Contains scripts for training and evaluating Single Classification Trees.
- `random_forests/`: Contains scripts for training and evaluating Random Forest models.
- `gradient_boosting/`: Contains scripts for training and evaluating Gradient Boosting models.
- `significance_test/`: Contains a script for performing significance tests on the results of different classifiers.
- `op_spam_v1.4/` The dataset used in whole experiment.

The only dependency required is [scikit-learn](https://scikit-learn.org/stable/), and all code is implemented using Python 3.10.15.


## Multinomial Naive Bayes

The unigram version is implemented in `naive_bayes_unigram.py`, and the bigram version in `naive_bayes_bigram.py`.
To run the unigram model, simply execute:

```bash
$ cd naive_bayes/

$ python naive_bayes_unigram.py
```

All experimental results will be saved in the `results_mnb_unigram` directory.
Similarly, running `naive_bayes_bigram.py` will store all results in the `results_mnb_bigram` directory.

Each result folder contains the confusion matrix, a simplified metrics report, a full metrics report, the prediction pairs used for statistical significance testing, and the top indicative terms.

## Logistic regression
The unigram version is implemented in `logistic_regression_unigram.py`, and the bigram version in `logistic_regression_bigram.py`.
To run the unigram model, simply execute:
```bash
$ cd logistic_regression/

$ python logistic_regression_unigram.py
```
All experimental results will be saved in the `results_lr_l1_unigram` directory. Similarly, running `logistic_regression_bigram.py` will store all results in the `results_lr_l1_bigram` directory.

We also implemented a TF-IDF-based version in `lr_l1_compare_vecs.py`.
Running this script generates results for both the TF-IDF and the standard Count vectorizer versions, which are saved in the `results_lr_l1_compare` directory.


## Single classification trees

We implemented a simple CLI for this model. Run the following command to show help:
```bash
$ python single_classification_trees/train_model_tree.py -h
```

Usage:
```bash
$ python single_classification_trees/train_model_tree.py \
    --preset <uni_std|bi_std> \
```
The `uni_std` parameter means the unigram version, and `bi_std` means bigram model.

After a run, results are written to:
- `single_classification_trees/results_uni_std/` (for --preset uni_std)
- `single_classification_trees/results_bi_std/` (for --preset bi_std)

Each results folder contains:
1. `best_params.json`: fitted CV configurations.
2. `dt_params.csv`: fitted chosen hyperparameters. 
3. `metrics.csv`: accuracy, macro precision/recall/F1.
4. `metrics_full.csv`: full test report (per-class precision/recall/F1/support, accuracy, macro/weighted avg).
5. `confusion_matrix.csv`: matrix indexed by rows `true_0`, `true_1`, columns `pred_0`, `pred_1.`
6. `predictions_fold5.csv`:  predictions pairs for significance test.
7. `top_features.csv`: top indicative terms.
8. `run_summary.csv`: one-row summary (CV macro-F1, test metrics, best params JSON, grid size, training time, output dir).

## Random forests

The usage method is the same as single classification trees:
```bash
$ python random_forests/train_model_forests.py \
    --preset <uni_std|bi_std> \
```
The command line parameter are same as above. After a run, results are written to:
- `random_forests/results_uni_std/` (for --preset uni_std)
- `random_forests/results_bi_std/` (for --preset bi_std)

Each results folder contains:
1. `rf_params.csv`: fitted hyperparameters 
2. `metrics.csv`: accuracy, macro precision/recall/F1.
3. `metrics_full.csv`: full test report (per-class precision/recall/F1/support, accuracy, macro/weighted avg).
4. `confusion_matrix.csv`: matrix indexed by rows `true_0`,`true_1`, columns `pred_0`, `pred_1`.
5. `predictions_fold5.csv`: prediction pairs for significance test.
6. `top_features.csv`: top indicative terms.

## Gradient boosting

Usage:
```bash
$ python gradient_boosting/gradient_boosting_fixed.py 
```

**Note**: the dataset `op_spam_v1.4.zip` must be in the same directory as `gradient_boosting_fixed.py` file.


After a run, results are written to the same directory as `gradient_boosting_fixed.py:`
1. `gb_uni.pkl, gb_uni_plus_bi.pkl`: training pipelines.
2. `gb_*_metrics.json`: accuracy, macro-F1, per-class precision/recall/F1.
3. `gb_*_metrics_full.json`: full test report (per-class precision/recall/F1/support, accuracy, macro/weighted avg).
4. `gb_*_confusion.csv`: confusion matrix (rows: `true_truthful`, `true_deceptive`; cols: `pred_truthful`, `pred_deceptive`).
5. `gb_*_predictions.csv`: fold-5 predictions with `y_true`, `y_pred`, `p_deceptive`.
6. `gb_*_top_terms_deceptive.txt`: top deceptive-leaning features (term, impurity importance).
7. `gb_*_top_terms_truthful.txt`: top truthful-leaning features (term, impurity importance).
