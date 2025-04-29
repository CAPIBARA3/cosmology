# import libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

colors = list(mcolors.BASE_COLORS.values())

# import modules
from readingCSV import read_data
data = read_data()

from stats_plot import stats
stats = stats()


# read data from dataset.csv file
H0, sigmaH0 = data.get_hubble()
Ode, sigmaOde = data.get_lambda()

plt.figure()

# scatter plot
for i, name, year in zip(range(len(H0)), data.get_dataset(), data.get_year()):
    plt.errorbar(H0[i], Ode[i], xerr=sigmaH0[i], yerr=sigmaOde[i], label=name+f' ({year})', fmt='o', capsize=5, color=colors[i])

# without discriminating over detection method
#H0, Ode, sigmaH0, sigmaOde = [H0], list(Ode), list(sigmaH0), list(sigmaOde)
mean = np.mean(np.array(H0), np.array(Ode))
cov = np.cov(np.array(H0), np.array(Ode))
#mean, cov = stats.mean_cov(H0, sigmaH0, Ode, sigmaOde)
stats.plot_confidence_region(plt.gca(), mean, cov, n_std=1, edgecolor='red', facecolor='red', fill=True, alpha=0.6, label='1σ Confidence Region')
stats.plot_confidence_region(plt.gca(), mean, cov, n_std=2, edgecolor='red', facecolor='red', fill=True, alpha=0.2, label='2σ Confidence Region')


# discriminating over detection methode (early universe vs distance ladder)
# method = data.get_detection()
# for method_type in ['Direct','Indirect']:
#     for i in range(len(set(method))):
#         H0_data, H0_err = [], []
#         Ode_data, Ode_err = [], []
#         if method[i] == method_type:
#             H0_data.append(H0[i])
#             H0_err.append(sigmaH0[i])
#             Ode_data.append(Ode[i])
#             Ode_err.append(sigmaOde[i])
#         mean, cov = stats.mean_cov(H0_data, Ode_data, H0_err, Ode_err)
#         stats.plot_confidence_ellipse(plt.gca(), mean, cov, n_std=1, edgecolor='red', facecolor='red', fill=True, alpha=0.6, label='1σ Confidence Region')
#         stats.plot_confidence_ellipse(plt.gca(), mean, cov, n_std=2, edgecolor='red', facecolor='red', fill=True, alpha=0.2, label='2σ Confidence Region')


# set limits and labels
plt.xlim(64, 76)
plt.ylim(0.675, 0.74)
plt.xlabel("$H_0$ (km/s/Mpc)")
plt.ylabel("$Omega_\Lambda$")
plt.title("Confidence Regions for $H_0$ and $Omega_\Lambda$ relation")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.grid()
plt.show()

# plt.savefig('Ode-H0.png')