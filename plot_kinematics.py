#!/usr/bin/python3
import sys
import numpy as np
import pandas as pd
import seaborn as sns
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


if len(sys.argv) != 2:
    sys.exit("Please provide exactly one argument: dataset-label")

dataset_label = sys.argv[1].upper()
sub_strings = dataset_label.split("_")
m_sl = int(sub_strings[0][2:])
m_n  = int(sub_strings[1][1:])
features = sub_strings[2]

event_file = "../data/processed_data/input_data/dataset_"\
    + dataset_label + ".csv.gz"

df = pd.read_csv(event_file)
df_LO = df[df['target'] == 0.]
df_LO_NLO = df[df['target'] == 1.]


# Prepare a plot object
fig, axes = plt.subplots(
    2, 1, sharex=True,
    figsize=(15, 15),
    gridspec_kw={'height_ratios': [3, 1]}
)

X_LABEL = r"$p^{miss}_T$ [GeV]"
Y_LABEL = r"Density $[\mathrm{GeV^{-1}}]$"
TITLE = "Dislepton production, " + r"$\sqrt{s} = 13$TeV"

fig.suptitle(TITLE)
axes[0].set_ylabel(Y_LABEL)
axes[1].set_xlabel(X_LABEL)
axes[1].set_ylabel(r"relative diff. [$\%$]")


# Create fixed binning
binning = np.linspace(110, 450, 200)
x = binning[:-1] + np.diff(binning)/2.


# Bin the data
INDEX = 'met'
binned_df_LO = pd.cut(df_LO[INDEX], bins=binning).value_counts()
binned_df_LO_NLO = pd.cut(df_LO_NLO[INDEX], bins=binning).value_counts()

rel_diff, rel_diff_uncert = st.calculate_relative_difference(
    binned_df_LO, binned_df_LO_NLO)

# Calculate KL-divergence
LO_hist = binned_df_LO, binning
LO_NLO_hist = binned_df_LO_NLO, binning

kl_div, kl_div_cum, kl_xvals = st.KL_div(LO_NLO_hist, LO_hist, base_two=True)

# Plot histograms along chosen column from datasets (x=INDEX)
sns.histplot(ax=axes[0], data=df_LO, x=INDEX, label='LO',
             bins=binning, stat='density', element='step', fill=False)
sns.histplot(ax=axes[0], data=df_LO_NLO, x=INDEX, label='LO+NLO',
             bins=binning, stat='density', element='step', fill=False)

# Plot the relative difference between datasets (in %)
axes[1].errorbar(x=x, y=100.*rel_diff, yerr=100. *
                 rel_diff_uncert, fmt='o', ecolor='r', mec='b', ms=2.4)
axes[1].axhline(0., ls='--')

axes[0].legend(title=st.KL_div_legend_title(kl_div, 'bits'))

axes[0].grid()
axes[1].grid()

# Show plot
plt.show()
