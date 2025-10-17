README 文件最后再写怎么运行代码

每个文件夹包括一种算法

文件夹里面 .py 是训练/测试 5 个算法 10 个模型的代码文件

文件夹里面 results 开头的文件夹保存运行结果, 比如 metrics.csv

op_spam_v1.4 是数据集

记得修改代码里面的路径，让代码能够找到数据集，成功跑起来

### single_classification_trees

Show help:
```bash
$ python single_classification_trees/train_model_tree.py -h
```

Usage:
```bash
$ python single_classification_trees/train_model_tree.py \
    --preset <uni_std|bi_std> \
```
After a run, results are written to:
- single_classification_trees/results_uni_std/ (for --preset uni_std)
- single_classification_trees/results_bi_std/ (for --preset bi_std)

Each results folder contains:
1. best_params.json: best CV config + metadata
2. dt_params.csv: best parameters as two columns param,value
3. metrics.csv: accuracy, macro precision/recall/F1.
4. metrics_full.csv: full test report (per-class precision/recall/F1/support, accuracy, macro/weighted avg)
5. confusion_matrix.csv: matrix indexed by true_0,true_1, columns pred_0,pred_1
6. predictions_fold5.csv: test predictions
7. top_features.csv: top terms
8. run_summary.csv: one-row summary (CV macro-F1, test metrics, best params JSON, grid size, train time, output dir).

### random_forests

Usage:
```bash
$ python random_forests/train_model_forests.py \
    --preset <uni_std|bi_std> \
```

After a run, results are written to:
- random_forests/results_uni_std/ (for --preset uni_std)
- random_forests/results_bi_std/ (for --preset bi_std)

Each results folder contains:
1. rf_params.csv: best parameters as two columns param,value
2. metrics.csv: accuracy, macro precision/recall/F1.
3. metrics_full.csv: full test report (per-class precision/recall/F1/support, accuracy, macro/weighted avg)
4. confusion_matrix.csv: matrix indexed by true_0,true_1, columns pred_0,pred_1
5. predictions_fold5.csv: test predictions
6. top_features.csv: top terms

### gradient_boosting


Usage:
```bash
$ python gradient_boosting/gradient_boosting_fixed.py \
  
```


Note:
op_spam_v1.4.zip must be in the same directory as gradient_boosting_fixed.py



After a run, results are written to the same directory as gradient_boosting_fixed.py:
1. gb_uni.pkl, gb_uni_plus_bi.pkl: trained pipelines
2. gb_*_metrics.json: accuracy, macro-F1, per-class precision/recall/F1
3. gb_*_metrics_full.json: full test report (per-class precision/recall/F1/support, accuracy, macro/weighted avg)
4. gb_*_confusion.csv: confusion matrix (rows: true_truthful,true_deceptive; cols: pred_truthful,pred_deceptive)
5. gb_*_predictions.csv: fold-5 predictions with y_true, y_pred, p_deceptive
6. gb_*_top_terms_deceptive.txt: top deceptive-leaning features (term, impurity importance)
7. gb_*_top_terms_truthful.txt: top truthful-leaning features (term, impurity importance).
