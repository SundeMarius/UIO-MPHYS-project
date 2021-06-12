#!/usr/bin/python3

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lib.statistics_tools as st

plt.rcParams["figure.figsize"] = (8,8)
# Settings for seaborn plots
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2})
custom_style = {
    'grid.color': '0.8',
    'shadow': True,
    'font.size': 14,
    'axes.facecolor': 'white',
    'axes.linewidth': 1.5,
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'legend.handlelength': 1.0,
    'style': 'whitegrid',
}
sns.set_style(custom_style)

if len(sys.argv) != 2:
    sys.exit("Please provide exactly one argument: dataset-label")


dataset_label = sys.argv[1].upper()
sub_strings = dataset_label.split("_")
m_sl = int(sub_strings[0][2:])
m_n = int(sub_strings[1][1:])
features = sub_strings[2]

# BDT model to load
model_file = "/home/marius/University/master_project/data/BDT_models/model_"\
             + dataset_label + ".model"

path = "/home/marius/University/master_project/data/processed_data/output_data/"
LO_file = path + "LO_predictions_" + dataset_label + ".csv.gz"
LO_NLO_file = path + "LO+NLO_predictions_" + dataset_label + ".csv.gz"
df_LO = pd.read_csv(LO_file)
df_LO_NLO = pd.read_csv(LO_NLO_file)

# Event dataset for D_KL calculation
event_file = "../data/processed_data/input_data/dataset_"\
    + dataset_label + ".csv.gz"
event_dataset = pd.read_csv(event_file)
event_dataset = event_dataset.sample(frac=.10).reset_index(drop=True)
event_dataset = event_dataset[event_dataset['target'] == 1.]
event_dataset = event_dataset.iloc[:, 1:]


# Prepare a plot object
fig, axes = plt.subplots(
    2, 1, sharex=True,
    figsize=(12, 12),
    gridspec_kw={'height_ratios': [2, 1]}
)
X_LABEL = "BDT output (s)"
Y_LABEL = "Probability density"
TITLE = "Dislepton Production Event Classification, %s" % features

fig.suptitle(TITLE)
axes[0].set_ylabel(Y_LABEL)
axes[1].set_xlabel(X_LABEL)
axes[1].set_ylabel(r"$\log_2(\frac{p_s}{q_s})$")


# Create fixed binning
binning = np.linspace(0, 1, 200)
x = binning[:-1] + np.diff(binning)/2.

# Bin the data
index = "prediction"
LO_hist = np.histogram(df_LO[index], bins=binning)
LO_NLO_hist = np.histogram(df_LO_NLO[index], bins=binning)

log_diff, log_diff_uncert = st.calculate_log_difference(LO_hist[0], LO_NLO_hist[0], base=2)


# Calculate KL-divergence
kl_div, kl_div_cum, kl_xvals = st.KL_div(LO_NLO_hist, LO_hist, base_two=True)


# Calculate D_KL(Kinematics)
bst = xgb.Booster({'nthread': 6})  # init model
bst.load_model(model_file)  # load data

dataset = xgb.DMatrix(event_dataset)
predictions = bst.predict(dataset)

indices = (np.digitize(predictions, binning)-1).tolist()
p = LO_NLO_hist[0]/(LO_NLO_hist[0].sum())
q = LO_hist[0]/(LO_hist[0].sum())

p = p[indices]
q = q[indices]

# Estimate D_KL
r = np.divide(p, q, out=np.ones_like(p), where=q>1e-5)
lpq = np.log2(r, out=np.zeros_like(r), where=r>1e-5)
avg_DKL = np.average(lpq)
print("DKL-estimate   : %.4e bits" % avg_DKL)

#Uncertainty in D_KL
N = len(lpq)
sample_std = np.std(lpq)/np.sqrt(N)
print("std in DKL-estimate   : %.4e bits" % sample_std)

# Plot cum. KL-div
# second_yaxis = axes[0].twinx()
# sns.lineplot(x=kl_xvals[:-1], y=kl_div_cum, color='g',
#             alpha=0.5, lw=1, ax=second_yaxis, label=r'cum. $D_{KL}$')
# second_yaxis.legend(loc='lower right')
# second_yaxis.set_ylabel(r"$D_{KL}$")

# Plot histograms along chosen column from datasets (x=index)
sns.histplot(data=df_LO, x=index, bins=binning, ax=axes[0],
             label=r'$q_s$', stat='density', element='step', fill=False)
sns.histplot(data=df_LO_NLO, x=index, bins=binning, ax=axes[0],
             label=r'$p_s$', stat='density', element='step', fill=False)

# Plot the ratio between the two classes
axes[1].errorbar(x=x, y=log_diff, yerr=log_diff_uncert,
    fmt='o', ecolor='r', mec='b', ms=2.4)
axes[1].axhline(0., ls='--')

x_idx = np.argwhere(log_diff != 0.)
x_min, x_max = x[x_idx][0]-1e-3, x[x_idx][-1]+1e-3
y_min, y_max = min(log_diff[x_idx])-0.35, max(log_diff[x_idx])+0.35

axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)

axes[0].set_ylim(-5e-2, axes[0].axis()[3]*1.25)
axes[0].legend(loc="center right", frameon=True) #bbox_to_anchor=(1.13, 0.5))
axes[0].annotate(st.KL_div_legend_title(kl_div, 'bits'),
                 xy=(0.05, 0.95), xycoords='axes fraction')
axes[0].annotate(r"$D_{KL}$$\left(\mathrm{p}_x||\mathrm{q}_x\right)$: %.1e bits" % (avg_DKL),
                 xy=(0.048, 0.88), xycoords='axes fraction')
axes[0].annotate(f"# LO events: {df_LO.shape[0]:,}",
                 xy=(0.50, 0.95), xycoords='axes fraction')
axes[0].annotate(f"# LO+NLO events: {df_LO_NLO.shape[0]:,}",
                 xy=(0.50, 0.88), xycoords='axes fraction')
axes[0].annotate(r"m($\tilde{e}$, $\tilde{\chi}_1^0$): (%i, %i) GeV" % (m_sl, m_n),
                 xy=(0.50, 0.81), xycoords='axes fraction')

axes[0].grid()
axes[1].grid()

path = "/home/marius/University/master_project/thesis/results/"
plt.savefig(path+"BDToutput_hist_"+dataset_label+".pdf")
