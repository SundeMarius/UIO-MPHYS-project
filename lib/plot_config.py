import seaborn as sns
import matplotlib.pyplot as plt

# Default settings for plots
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
sns.set_style("whitegrid")

plt.rcParams["mathtext.fontset"] = "stix"
custom_style = {
        'grid.color': '0.8',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'savefig.dpi':300,
        'font.size': 14,
        'axes.facecolor':'lightgrey',
        'axes.linewidth': 1.5,
        'mathtext.fontset': 'stix',
        'font.family': 'serif',
        'font.serif': 'Times',
}
sns.set_style(custom_style)

