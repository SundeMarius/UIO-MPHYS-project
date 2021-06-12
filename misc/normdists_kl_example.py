import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

fig_name = "/home/marius/University/master_project/thesis/results/kl_divergence_ex4.pdf"

plt.rcParams["figure.figsize"] = (13,13)
# Settings for seaborn plots
sns.set_context("paper", font_scale=2.9, rc={"lines.linewidth": 2})
custom_style = {
    'grid.color': '0.8',
    'shadow': True,
    'font.size': 19,
    'axes.facecolor': 'white',
    'axes.linewidth': 1.5,
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'legend.handlelength': 1.0,
    'style': 'whitegrid',
}
sns.set_style(custom_style)

x = np.linspace(-7, 7, 250)

mu1, sg1 = n1 = 0, 1.0
mu2, sg2 = n2 = 0, 1.5

# Analytical D_KL(n2 || n1)
analytical_DKL = ((sg2/sg1)**2 + (mu2 - mu1)**2 / sg1**2 - 1. + np.log(sg1**2/sg2**2))/(2.*np.log(2.))

n_obj1 = norm(*n1)
y1 = n_obj1.pdf(x)
n_obj2 = norm(*n2)
y2 = n_obj2.pdf(x)

# KL-divergence integrand
rel_entropy = y2*np.log2(y2/y1)

X_LABEL = "X"
Y_LABEL = "Probability density"
TITLE = r"$D_{KL}(p || q) =$ %.1e bits" % analytical_DKL

plt.suptitle(TITLE)
plt.ylabel(Y_LABEL)
plt.xlabel(X_LABEL)

sns.lineplot(x=x, y=y1, label=r'$q = \mathcal{N}(%.1f, %.1f)$' % (mu1, sg1))
sns.lineplot(x=x, y=y2, label=r'$p = \mathcal{N}(%.1f, %.1f)$' % (mu2, sg2))
plt.fill_between(x=x, y1=rel_entropy, label='Rel. entropy', alpha=0.4)
plt.legend(loc='upper right')
plt.grid()
#plt.show()
plt.savefig(fig_name)
