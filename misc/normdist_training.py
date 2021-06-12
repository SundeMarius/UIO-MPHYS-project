#!/usr/bin/python3
import os
import sys
sys.path.insert(0, "..")
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import lib.statistics_tools as st
from numpy.random import default_rng
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

plt.rcParams["figure.figsize"] = (13,13)
# Settings for seaborn plots
sns.set_context("paper", font_scale=2.9, rc={"lines.linewidth": 2})
custom_style = {
    'grid.color': '0.8',
    'shadow': True,
    'font.size': 16,
    'axes.facecolor': 'white',
    'axes.linewidth': 1.5,
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'legend.handlelength': 1.0,
    'style': 'whitegrid',
}
sns.set_style(custom_style)

fig1 = "/home/marius/University/master_project/thesis/results/trained_gauss_ex1_manybins.pdf"
test = "Example 1"
# Create a default random number generator
rng = default_rng()

bins = np.linspace(0, 1, 5000)
x = bins[:-1] + np.diff(bins)/2.

mu1, sg1, sample1 = n1 = -0.1, 1., 10_000_000
mu2, sg2, sample2 = n2 = 0.1, 1., 10_000_000
# Analytical D_KL(n2 || n1)
analytical_DKL = ((sg2/sg1)**2 + (mu2 - mu1)**2 / sg1**2 - 1. + np.log(sg1**2/sg2**2))/(2.*np.log(2.))

target_feature1 = np.stack((np.zeros(sample1), rng.normal(*n1)), axis=-1)
target_feature2 = np.stack((np.ones(sample2), rng.normal(*n2)), axis=-1)
dataset = np.concatenate((target_feature1, target_feature2))

# Balance dataset
X_LO, y_LO = target_feature1[:, 1], target_feature1[:, 0]
X_NLO, y_NLO = target_feature2[:, 1], target_feature2[:, 0]

train_frac = 0.8
N = int(train_frac*X_LO.shape[0])
X_train, X_test = np.concatenate((X_LO[:N], X_NLO[:N])), np.concatenate((X_LO[N:], X_NLO[N:]))
y_train, y_test = np.concatenate((y_LO[:N], y_NLO[:N])), np.concatenate((y_LO[N:], y_NLO[N:]))

# XGBOOST
param = {
    'max_depth': 4,
    'eta': 0.05,
    'verbosity': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist'
}

X_train = np.asmatrix(X_train).T
X_test = np.asmatrix(X_test).T

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
evallist = [(dtrain, 'train'), (dtest, 'eval')]

bst = xgb.Booster({'nthread': 6})
ans = 'y'
model_file = "normaldists_training.model"

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
predictions = np.round(y_pred)

# evaluate accuracy score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

pred0 = y_pred[y_test == 0]
pred1 = y_pred[y_test == 1]

hist0 = np.histogram(pred0, bins)
hist1 = np.histogram(pred1, bins)


log_diff, log_diff_uncert = st.calculate_log_difference(hist0[0], hist1[0], base=2)

# Calculate KL-divergence
kl_div, kl_div_cum, kl_xvals = st.KL_div(hist1, hist0, base_two=True)


indices = (np.digitize(pred1, bins) - 1).tolist()
p = hist1[0]/(hist1[0].sum())
q = hist0[0]/(hist0[0].sum())

p = p[indices]
q = q[indices]

r = np.divide(p, q, out=np.ones_like(p), where=q>1e-8)
lpq = np.log2(r, out=np.zeros_like(r), where=r>1e-8)
avg_DKL = np.average(lpq)

print("Average D_KL from p:  %.1e bits" % avg_DKL)
print("Analytic D_KL from p: %.1e bits" % analytical_DKL)

#Uncertainty in D_KL
N = len(lpq)
sample_std = np.std(lpq)/np.sqrt(N)
print("std in DKL-estimate   : %.1e" % sample_std)
print("uncert in DKL-estimate: %.1f percent" % (sample_std/avg_DKL*1e2))


# Prepare a plot object
fig, axes = plt.subplots(
    2, 1, sharex=True,
    figsize=(12, 12),
    gridspec_kw={'height_ratios': [2, 1]}
)
X_LABEL = "BDT output (s)"
Y_LABEL = "Probability density"
TITLE = "Classification of Samples from Two Normal Distributions \n %s" % test

#fig.suptitle(TITLE)
axes[0].set_ylabel(Y_LABEL)
axes[1].set_xlabel(X_LABEL)
axes[1].set_ylabel(r"$\log_2(\frac{p_s}{q_s})$")

axes[0].annotate(r"$D_{KL}$$\left(\mathrm{p}_s||\mathrm{q}_s\right)$: %.1e bits" % (kl_div),
                 xy=(0.05, 0.92), xycoords='axes fraction')
axes[0].annotate(r"$D_{KL}$$\left(\mathrm{p}_x||\mathrm{q}_x\right)$: %.1e bits" % (avg_DKL),
                 xy=(0.047, 0.85), xycoords='axes fraction')
# axes[0].annotate(test, xy=(0.047, 0.81), xycoords='axes fraction')

sns.histplot(data=pred0, bins=bins, ax=axes[0], stat='density',
             label=r'$q_s$', element='step', fill=False)
sns.histplot(data=pred1, bins=bins, ax=axes[0], stat='density',
             label=r'$p_s$', element='step', fill=False)

# Plot the log difference between datasets
axes[1].errorbar(x=x, y=log_diff, yerr=log_diff_uncert,
    fmt='o', ecolor='r', mec='b', ms=2.4)
axes[1].axhline(0., ls='--')

x_idx = np.argwhere(log_diff != 0.)
x_min, x_max = x[x_idx][0]-1e-3, x[x_idx][-1]+1e-3
y_min, y_max = min(log_diff[x_idx])-0.35, max(log_diff[x_idx])+0.35

axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)

axes[0].set_ylim(0, axes[0].axis()[3]*1.25)

axes[0].legend(loc="center right") #frameon=False, bbox_to_anchor=(1.14, 0.5))
axes[0].grid()
axes[1].grid()

plt.savefig(fig1)
