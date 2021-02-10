#!/usr/bin/python
import sys

if len(sys.argv) != 3:
    print("Please provide exactly two arguments (path to LO-csv-dataset, path to LO+NLO-csv-dataset).")
    exit(1)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# To set the default plot style 
import lib.plot_config

# Kinematic variable to plot
index = 'missing_pt_GeV'

# Prepare plot to visualise result
x_label = r"$p^{miss}_T$ [GeV]"
y_label = r"Density $[\mathrm{GeV^{-1}}]$"
title = r"$pp\rightarrow$ -11 1000022 11 1000022 j electroweak, $\sqrt{s} = 13$TeV"
labels = ["LO","LO+NLO"]

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15,15), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(title)

df_LO = pd.read_csv(sys.argv[1])
df_LO_NLO = pd.read_csv(sys.argv[2])

# Create histograms, and fixed binning
binning = np.linspace(110., 550., 101)

# Bin the data -- normalize to make valid comparison
binned_df_LO = pd.cut(df_LO[index], bins=binning).value_counts(normalize=True, sort=True)
binned_df_LO_NLO = pd.cut(df_LO_NLO[index], bins=binning).value_counts(normalize=True, sort=True)


# Plot step histograms along chosen column from datasets (x=index)
sns.histplot(ax=axes[0], data=df_LO, x=index, stat='density', bins=binning, element='step', fill=False)
sns.histplot(ax=axes[0], data=df_LO_NLO, x=index, stat='density', bins=binning, element='step', fill=False)

# Plot the relative difference between datasets (in %)
relative_diff = 100.*(binned_df_LO_NLO - binned_df_LO).div(binned_df_LO)
x = binning[:-1] + np.diff(binning)/2.
sns.scatterplot(ax=axes[1], x=x, y=relative_diff)
sns.lineplot(ax=axes[1], x=x, y=relative_diff, color='red', lw=0.8)
axes[1].axhline(0., ls='--')

axes[0].set_ylabel(y_label)
axes[0].set_xlim(110,440)
axes[0].legend(labels)
axes[1].set_xlabel(x_label)
axes[1].set_ylabel(r"relative diff. [$\%$]")
axes[1].set_ylim(-15,15)
plt.show()
exit(1)

# Create histogram (use fixed manual set binning)
counts, bins = np.histogram(data, bins=binning)

histograms.append([counts, bins])

# Q-distribution representing the model
LO_hist = histograms[0]
# P-distribution representing the data
NLO_hist = histograms[1]
# Calculate KL-divergence between LO and NLO distributions
kl_div = st.KL_div(NLO_hist, LO_hist, base_two=True)
plot.add_label(r"KL-div(LO$\rightarrow$NLO): %1.2e bits"%kl_div)

# Plot data
plot.plot_all()
