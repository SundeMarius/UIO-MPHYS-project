#!/usr/bin/python
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
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'savefig.dpi': 300,
        'shadow': True,
        'font.size': 11,
        'axes.facecolor': 'lightgrey',
        'axes.linewidth': 1.5,
        'mathtext.fontset': 'stix',
        'font.family': 'serif',
        'font.serif': 'Times',
        'legend.handlelength': 1.0,
}
sns.set_style(custom_style)

if len(sys.argv) != 3:

    print("Please provide exactly two arguments: "
          "path to LO-csv-dataset, path to LO+NLO-csv-dataset).")
    sys.exit(1)


# Kinematic variable to plot
INDEX = "missing_pt_GeV"

df_LO = pd.read_csv(sys.argv[1])
df_LO_NLO = pd.read_csv(sys.argv[2])


# Prepare plot object
fig, axes = plt.subplots(
    2, 1, sharex=True,
    figsize=(15, 15),
    gridspec_kw={'height_ratios': [3, 1]}
)

X_LABEL = r"$p^{miss}_T$ [GeV]"
Y_LABEL = r"Density $[\mathrm{GeV^{-1}}]$"
TITLE = r"$pp\rightarrow$ -11 1000022 11 1000022 j"\
        r" electroweak, $\sqrt{s} = 13$TeV"
fig.suptitle(TITLE)

axes[0].set_ylabel(Y_LABEL)
axes[1].set_xlabel(X_LABEL)
axes[1].set_ylabel(r"relative diff. [$\%$]")


# Create fixed binning
binning = np.linspace(1, 600, 300)
x = binning[:-1] + np.diff(binning)/2.


# Bin the data
binned_df_LO = pd.cut(df_LO[INDEX], bins=binning).value_counts()
binned_df_LO_NLO = pd.cut(df_LO_NLO[INDEX], bins=binning).value_counts()

rel_diff, rel_diff_uncert = st.calculate_relative_difference(binned_df_LO, binned_df_LO_NLO)

# Calculate KL-divergence
LO_hist = binned_df_LO, binning
NLO_hist = binned_df_LO_NLO, binning
kl_div = st.KL_div(NLO_hist, LO_hist, base_two=True)


# Plot histograms along chosen column from datasets (x=INDEX)
sns.histplot(ax=axes[0], data=df_LO, x=INDEX, label='LO',
             bins=binning, stat='density', element='step', fill=False)
sns.histplot(ax=axes[0], data=df_LO_NLO, x=INDEX, label='LO+NLO',
             bins=binning, stat='density', element='step', fill=False)

# Plot the relative difference between datasets (in %)
axes[1].errorbar(x=x, y=100.*rel_diff, yerr=100.*rel_diff_uncert, fmt='o')
axes[1].axhline(0., ls='--')

axes[0].legend(title=st.KL_div_legend_title(kl_div, 'bits'))


# Show plot
plt.show()
