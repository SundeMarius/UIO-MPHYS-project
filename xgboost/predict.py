#!/usr/bin/python3
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset_label = sys.argv[1].upper()

sub_strings = dataset_label.split("_")
m_sl = sub_strings[0]
m_n = sub_strings[1]
features = sub_strings[2]

# BDT model to load
model_file = "/home/marius/University/master_project/data/BDT_models/model_"\
             + dataset_label + ".model"

# Datasets from BDT
if dataset_label not in sys.argv[2]:
    sys.exit("Chosen BDT model doesn't match dataset.")

output_LO = "LO_predictions_" + dataset_label + ".csv.gz"
output_LO_NLO = "LO+NLO_predictions_" + dataset_label + ".csv.gz"

input_file = sys.argv[2]
dataset = pd.read_csv(input_file, dtype=float, header=0, low_memory=False)


# Balance dataset
n_events_LO, n_events_NLO = dataset.target.value_counts(sort=False)
numb_events_to_remove = n_events_LO - n_events_NLO
if numb_events_to_remove < 0:
    dataset = dataset[:numb_events_to_remove]
else:
    dataset = dataset[numb_events_to_remove:]

LO, LO_NLO = dataset[dataset['target'] == 0.], dataset[dataset['target'] == 1.]

X_LO, y_LO = LO.iloc[:, 1:], LO.iloc[:, 0]
X_NLO, y_NLO = LO_NLO.iloc[:, 1:], LO_NLO.iloc[:, 0]

train_frac = 0.8
N = int(train_frac*X_LO.shape[0])
X_train, X_test = X_LO[:N].append(X_NLO[:N]), X_LO[N:].append(X_NLO[N:])
y_train, y_test = y_LO[:N].append(y_NLO[:N]), y_LO[N:].append(y_NLO[N:])

zeros = sum(y_train == 0)
ones = sum(y_train == 1)
print("# LO: ", zeros, "\n# LO+NLO: ", ones)
print("#LO / #LO+NLO: ", zeros/ones)

# XGBOOST
param = {
    'max_depth': 4,
    'eta': 0.1,
    'gamma': 3,
    'verbosity': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist'
}

feature_names = X_test.columns
dtrain = xgb.DMatrix(X_train.to_numpy(),
                     label=y_train.to_numpy(), feature_names=feature_names)
dtest = xgb.DMatrix(X_test.to_numpy(), label=y_test.to_numpy(),
                    feature_names=feature_names)
evallist = [(dtrain, 'train'), (dtest, 'eval')]


bst = xgb.Booster({'nthread': 6})
ans = 'y'

if os.path.isfile(model_file):
    ans = input("'%s' exists already. Overwrite? (y,N): " % model_file).lower()

if ans == 'y':
    num_estimators = 10_000
    bst = xgb.train(param, dtrain, num_estimators,
                    evallist, early_stopping_rounds=20)
    bst.save_model(model_file)
else:
    bst.load_model(model_file)


y_pred = bst.predict(dtest)
print(y_pred)
predictions = np.round(y_pred)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


plt.rcParams["figure.figsize"] = (10,8)
# Settings for seaborn plots
sns.set_context("paper", font_scale=1.7, rc={"lines.linewidth": 2})
custom_style = {
    'grid.color': '0.8',
    'shadow': True,
    'font.size': 18,
    'axes.facecolor': 'white',
    'axes.linewidth': 1.5,
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'legend.handlelength': 1.0,
    'style': 'whitegrid',
}
sns.set_style(custom_style)

f_importance = np.array(list(bst.get_fscore().values()))
f_importance = f_importance/np.sum(f_importance)
sns.barplot(x=100.*f_importance, y=feature_names)
plt.xlabel("Feature importance [%]")
plt.ylabel("Feature")
plt.grid()
plt.tight_layout()
plt.savefig("../thesis/results/BDT_feature_importance_" + dataset_label)


# xgb.plot_tree(bst, num_trees=1)
# Sort output by true label (LO or LO+NLO)
LO = y_pred[y_test == 0]
LO_NLO = y_pred[y_test == 1]

pd.DataFrame(LO, columns=['prediction']).to_csv(
    "/home/marius/University/master_project/data/processed_data/output_data/"
    + output_LO,
    index=False)
pd.DataFrame(LO_NLO, columns=['prediction']).to_csv(
    "/home/marius/University/master_project/data/processed_data/output_data/"
    + output_LO_NLO,
    index=False)

print("Done")
