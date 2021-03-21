#!/usr/bin/python3
import os
import sys
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset_label = sys.argv[1].upper()

# BDT model to load
model_file = "/home/sunde/University/master_project/data/BDT_models/model_"\
             + dataset_label + ".model"

# Datasets from BDT
if dataset_label not in sys.argv[2]:
    sys.exit("Chosen BDT model doesn't match dataset.")

output_LO = "LO_predictions_" + dataset_label + ".csv.gz"
output_LO_NLO = "LO+NLO_predictions_" + dataset_label + ".csv.gz"
model_file = "/home/sunde/University/master_project/data/BDT_models/model_"\
             + dataset_label + ".model"


input_file = sys.argv[2]
dataset = pd.read_csv(input_file, dtype=float, header=0, low_memory=False)
dataset = dataset.sample(frac=1.).reset_index(drop=True)
print(dataset.head(20))

X, y = dataset.iloc[:, 1:], dataset.iloc[:, 0]
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

zeros = len(y_train[y_train == 0])
ones = len(y_train[y_train == 1])
print("# LO: ", zeros, "\n# LO+NLO: ", ones)
print("#LO / #LO+NLO: ", zeros/ones)

# XGBOOST
param = {
    'max_depth': 4,
    'eta': 0.144,
    'verbose': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
}

dtrain = xgb.DMatrix(X_train.to_numpy(),
                     label=y_train.to_numpy(), feature_names=feature_names)
dtest = xgb.DMatrix(X_test.to_numpy(), label=y_test.to_numpy(),
                    feature_names=feature_names)
evallist = [(dtrain, 'train'), (dtest, 'eval')]


bst = xgb.Booster({'nthread': 6})
ans = 'y'
if os.path.isfile(model_file):
    ans = input("'%s' exists already. Overwrite? (y,N): " % model_file).lower()
    if ans != 'y':
        bst.load_model(model_file)

if ans == 'y':
    num_round = 300
    bst = xgb.train(param, dtrain, num_round,
                    evallist, early_stopping_rounds=15)
    bst.save_model(model_file)

predictions = bst.predict(dtest)

# xgb.plot_tree(bst, num_trees=1)
# xgb.plot_importance(bst)

# Sort output by true label (LO or LO+NLO)
true_label = dtest.get_label()
LO = predictions[true_label == 0.]
LO_NLO = predictions[true_label == 1.]

pd.DataFrame(LO, columns=['prediction']).to_csv(
    "/home/sunde/University/master_project/data/processed_data/output_data/"
    + output_LO,
    index=False)
pd.DataFrame(LO_NLO, columns=['prediction']).to_csv(
    "/home/sunde/University/master_project/data/processed_data/output_data/"
    + output_LO_NLO,
    index=False)

print("Done")
plt.show()
