#!/usr/bin/python
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
import lib.statistics_tools as st

# Settings for seaborn plots
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})
custom_style = {
    'grid.color': '0.8',
    'shadow': True,
    'font.size': 10,
    'axes.facecolor': 'white',
    'axes.linewidth': 1.5,
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'legend.handlelength': 1.0,
    'style': 'whitegrid',
}
sns.set_style(custom_style)

if len(sys.argv) != 4:

    print("Please provide exactly two arguments: dataset-label,"
          "path to LO-csv-dataset, path to LO+NLO-csv-dataset")
    sys.exit(1)


# Kinematic variable to plot
index = "prediction"
dataset_label = sys.argv[1].upper()

# BDT model to load
model_file = "/home/sunde/University/master_project/data/BDT_models/model_"\
             + dataset_label + ".model"

# Datasets from BDT
if dataset_label not in sys.argv[2] or dataset_label not in sys.argv[3]:
    sys.exit("Chosen BDT model doesn't match datasets.")

df_LO = pd.read_csv(sys.argv[2])
df_LO_NLO = pd.read_csv(sys.argv[3])

# Event dataset for D_KL calculation
event_file = "../data/processed_data/input_data/dataset_"\
    + dataset_label + ".csv.gz"
event_dataset = pd.read_csv(event_file)
event_dataset = event_dataset.sample(frac=.02).reset_index(drop=True)
event_dataset = event_dataset[event_dataset['target'] == 1.]
event_dataset = event_dataset.iloc[:, 1:]


# Prepare a plot object
fig, axes = plt.subplots(
    2, 1, sharex=True,
    figsize=(15, 15),
    gridspec_kw={'height_ratios': [3, 1]}
)
X_LABEL = "BDT output"
Y_LABEL = "Density"
TITLE = r"$pp\rightarrow$ -11 1000022 11 1000022 j "\
        r"electroweak, $\sqrt{s} = 13$TeV"
TITLE = "Dislepton production, " + r"$\sqrt{s} = 13$TeV"

fig.suptitle(TITLE, fontweight='bold')
axes[0].set_ylabel(Y_LABEL)
axes[1].set_xlabel(X_LABEL)
axes[1].set_ylabel(r"relative diff. [$\%$]")


# Create fixed binning
binning = np.linspace(0, 1, 200)
x = binning[:-1] + np.diff(binning)/2.

# Bin the data
binned_df_LO = pd.cut(df_LO[index], bins=binning).value_counts(sort=False)
binned_df_LO_NLO = pd.cut(
    df_LO_NLO[index], bins=binning).value_counts(sort=False)
rel_diff, rel_diff_uncert = st.calculate_relative_difference(
    binned_df_LO, binned_df_LO_NLO)

# Calculate KL-divergence
LO_hist = [binned_df_LO.to_numpy(), binning]
LO_NLO_hist = [binned_df_LO_NLO.to_numpy(), binning]
kl_div, kl_div_cum, kl_xvals = st.KL_div(LO_NLO_hist, LO_hist, base_two=True)


# Calculate D_KL(Kinematics)
bst = xgb.Booster({'nthread': 6})  # init model
bst.load_model(model_file)  # load data

dataset = xgb.DMatrix(event_dataset)
predictions = bst.predict(dataset)

indices = np.digitize(predictions, binning).tolist()
p = LO_NLO_hist[0]/(LO_NLO_hist[0].sum())
q = LO_hist[0]/(LO_hist[0].sum())

p = p[indices]
q = q[indices]
# zero_ind_p = np.where(p == 0.)[0]
# zero_ind_q = np.where(q == 0.)[0]

# pred_zero = predictions[zero_ind_p]
# print(pred_zero)

# print(p[p == 0.])
# print(q[q == 0.])

# print(np.isclose(p, np.zeros(len(p))))
# print(np.isclose(q, np.zeros(len(q))))

# print(p, q, np.log(p/q))
lpq = 0.
N = len(q)
for i in range(N):
    if np.isclose(q[i], 0.):
        continue
    lpq += np.log(p[i]/q[i])

avg_DKL = lpq/N
print("Average D_KL from p: %.3e bits" % (avg_DKL/np.log(2.)))
exit(1)
# Plot cum. KL-div
second_yaxis = axes[0].twinx()
sns.lineplot(x=kl_xvals[:-1], y=kl_div_cum, color='g',
             alpha=0.5, lw=1, ax=second_yaxis, label=r'cum. $D_{KL}$')
second_yaxis.legend(loc='lower right')
second_yaxis.set_ylabel(r"$D_{KL}$")

# Plot histograms along chosen column from datasets (x=index)
sns.histplot(data=df_LO, x=index, bins=binning, ax=axes[0],
             label='LO', stat='density', element='step', fill=False)
sns.histplot(data=df_LO_NLO, x=index, bins=binning, ax=axes[0],
             label='LO+NLO', stat='density', element='step', fill=False)

# Plot the relative difference between datasets
axes[1].errorbar(x=x, y=100.*rel_diff, yerr=100. *
                 rel_diff_uncert, fmt='o', ecolor='r', mec='b', ms=2.4)
axes[1].axhline(0., ls='--')

axes[0].legend()
axes[0].annotate(st.KL_div_legend_title(kl_div, 'bits'),
                 xy=(0.05, 0.95), xycoords='axes fraction')
axes[0].annotate(f"# LO predictions: {df_LO.shape[0]:,}", xy=(
    0.05, 0.90), xycoords='axes fraction')
axes[0].annotate(f"# LO+NLO predictions: {df_LO_NLO.shape[0]:,}",
                 xy=(0.05, 0.85), xycoords='axes fraction')

axes[0].grid()
axes[1].grid()

# Show plot
plt.show()
