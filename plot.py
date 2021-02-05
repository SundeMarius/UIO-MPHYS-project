
# Prepare plot object to visualise result
x_label = r"$p_T^{miss}$ [GeV]"
y_label = r"density $[\mathrm{GeV^{-1}}]$"
title = r"$pp\rightarrow$ -11 1000022 11 1000022 j electroweak, $\sqrt{s} = 13$TeV"
plot = util.Plot(x_label, y_label, title)

labels = ["LO","NLO"]


# Create list to store histograms, and fixed binning
histograms = []
binning = np.linspace(0., 1.e3, 201)

#TODO add code to read datasets

# Create histogram (use fixed manual set binning)
counts, bins = np.histogram(data, bins=binning)

histograms.append([counts, bins])

# Add histogram to plot
plot.add_histogram(counts, bins, label=lab)

# Q-distribution representing the model
LO_hist = histograms[0]
# P-distribution representing the data
NLO_hist = histograms[1]
# Calculate KL-divergence between LO and NLO distributions
kl_div = st.KL_div(NLO_hist, LO_hist, base_two=True)
plot.add_label(r"KL-div(LO$\rightarrow$NLO): %1.2e bits"%kl_div)

# Plot data
plot.plot_all()

