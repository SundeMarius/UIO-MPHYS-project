#!/usr/bin/python
import sys
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

input_file = sys.argv[1]

dataset = pd.read_csv(input_file, dtype=float, header=0, low_memory=False)
dataset = dataset.sample(frac=1).reset_index(drop=True)

print(dataset.head(20))

X, y = dataset.iloc[:, 1:], dataset.iloc[:, 0]
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

zeros = len(y_train[y_train == 0])
ones = len(y_train[y_train == 1])
print(zeros, ones)
print(zeros/ones)

# XGBOOST
param = {
    'max_depth': 5,
    'eta': 0.1,
    'gamma': 1,
    'verbosity': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
}

dtrain = xgb.DMatrix(X_train.to_numpy(), label=y_train.to_numpy(), feature_names=feature_names)
dtest = xgb.DMatrix(X_test.to_numpy(), label=y_test.to_numpy(), feature_names=feature_names)

evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 150
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=20)


predictions = bst.predict(dtest)

# xgb.plot_tree(bst, num_trees=1)
xgb.plot_importance(bst)
plt.show()


# Sort output by true label (LO or LO+NLO)
true_label = dtest.get_label()
LO = predictions[true_label == 0.]
LO_NLO = predictions[true_label == 1.]


pd.DataFrame(LO, columns=['prediction']).to_csv(
    "/home/mariusss/University/master_project/data/output_data/LO_predictions.csv",
    index=False
)
pd.DataFrame(LO_NLO, columns=['prediction']).to_csv(
    "/home/mariusss/University/master_project/data/output_data/LO+NLO_predictions.csv",
    index=False
)
print("Done")
