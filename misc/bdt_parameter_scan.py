#!/usr/bin/python3
import os
import sys
import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

dataset_label = sys.argv[1].upper()
input_file = sys.argv[2]

if not dataset_label in input_file:
    sys.exit("Chosen dataset label doesn't match the dataset.")


# Training parameters
learning_rate = np.round(np.linspace(1.e-2, 5.e-1, 12), 2)
max_depth = list(range(3, 10))
n1, n2 = len(learning_rate), len(max_depth)
num_rounds = 200

logger = "BDT_tuning_logger_" + dataset_label + ".txt"
output_fn = "test_auc_scores_" + dataset_label + ".txt"
data = np.zeros((n1, n2))

if os.path.isfile(logger) and os.path.isfile(output_fn):

    ans = input("%s already exists. Overwrite? (y,N): " % logger).lower()

    if ans == 'y':

        with open(logger, 'w') as f:

            from datetime import datetime
            now = datetime.now()
            dt_string = now.ctime()

            line = "BDT tuning logger -- program started: %s\n" % dt_string
            f.write(line)
            f.write(len(line)*'-' + '\n')


            dataset = pd.read_csv(input_file, dtype=float, header=0, low_memory=False)
            dataset = dataset.sample(frac=1).reset_index(drop=True)

            X, y = dataset.iloc[:, 1:], dataset.iloc[:, 0]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # XGBOOST
            dtrain = xgb.DMatrix(
                X_train.to_numpy(),
                label=y_train.to_numpy(),
                feature_names=X.columns
            )
            dtest = xgb.DMatrix(
                X_test.to_numpy(),
                label=y_test.to_numpy(),
                feature_names=X.columns
            )

            evaluation = [(dtrain, 'train'), (dtest, 'test')]

            number_of_points = n1*n2

            c = 1
            for i, eta in enumerate(learning_rate):

                for j, depth in enumerate(max_depth):

                    print("lr, depth: %.2f, %d" % (eta, depth))
                    results = {}
                    param = {
                        'max_depth': depth,
                        'eta': eta,
                        'verbosity': 1,
                        'objective': 'binary:logistic',
                        'eval_metric': 'auc',
                        'tree_method': 'hist',
                    }

                    bst = xgb.train(param, dtrain,
                                    num_rounds,
                                    evals=evaluation,
                                    evals_result=results,
                                    early_stopping_rounds=15)

                    auc_train = float(results['train']['auc'][-1])
                    auc_test = float(results['test']['auc'][-1])

                    data[i][j] = auc_test

                    line = f"auc_train: {auc_train:.3e}, auc_test: {auc_test:.3e}\n"
                    f.write(line)

                    print("%d/%d done" % (c, number_of_points))
                    c += 1

                f.write(len(line)*'-'+'\n')

            now = datetime.now()
            dt_string = now.ctime()
            f.write("BDT tuning logger -- program ended: %s" % dt_string)

            np.savetxt(output_fn, data, fmt='%.3e')
    else:
        data = np.loadtxt(output_fn)

xlabels = [str(c) for c in max_depth]
ylabels = [str(c) for c in learning_rate]

sns.set_context("notebook")
sns.heatmap(data, xticklabels=xlabels, yticklabels=ylabels)

plt.xlabel("Max depth")
plt.ylabel("Learning rate")
plt.show()
