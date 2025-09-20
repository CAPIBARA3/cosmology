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
    plt.errorbar(H0[i], Ode[i], xerr=sigmaH0[i], yerr=sigmaOde[i],
                 label=name+f' ({year})', fmt='o', capsize=5, color=colors[i])

# OPTION 1: Use your stats.mean_cov method (recommended as it handles errors)
mean, cov = stats.mean_cov(H0, Ode, sigmaH0, sigmaOde)
stats.plot_confidence_region(plt.gca(), mean, cov, n_std=1,
                            edgecolor='red', facecolor='red', fill=True,
                            alpha=0.6, label='1σ Confidence Region')
stats.plot_confidence_region(plt.gca(), mean, cov, n_std=2,
                            edgecolor='red', facecolor='red', fill=True,
                            alpha=0.2, label='2σ Confidence Region')

# OPTION 2: Alternative using numpy only (if your stats methods have issues)
# H0_array = np.array(H0)
# Ode_array = np.array(Ode)
# mean = np.array([np.mean(H0_array), np.mean(Ode_array)])
# data_matrix = np.vstack((H0_array, Ode_array))
# cov = np.cov(data_matrix)
# # You'll need to use a different plotting method since plot_confidence_region expects different inputs
# # For numpy approach, you might need to create ellipses manually or use a different function

# discriminating over detection method (early universe vs distance ladder)
# method = data.get_detection()
# for method_type in ['Direct','Indirect']:
#     H0_data, H0_err = [], []
#     Ode_data, Ode_err = [], []
#     for i in range(len(method)):
#         if method[i] == method_type:
#             H0_data.append(H0[i])
#             H0_err.append(sigmaH0[i])
#             Ode_data.append(Ode[i])
#             Ode_err.append(sigmaOde[i])
#     if H0_data:  # Only proceed if we have data for this method
#         mean_method, cov_method = stats.mean_cov(H0_data, Ode_data, H0_err, Ode_err)
#         stats.plot_confidence_region(plt.gca(), mean_method, cov_method, n_std=1,
#                                     edgecolor='blue', facecolor='blue', fill=True,
#                                     alpha=0.4, label=f'1σ {method_type}')
#         stats.plot_confidence_region(plt.gca(), mean_method, cov_method, n_std=2,
#                                     edgecolor='blue', facecolor='blue', fill=True,
#                                     alpha=0.1, label=f'2σ {method_type}')

# set limits and labels
plt.xlim(64, 76)
plt.ylim(0.675, 0.74)
plt.xlabel("$H_0$ (km/s/Mpc)")
plt.ylabel(r"$\Omega_\Lambda$")
plt.title(r"Confidence Regions for $H_0$ and $\Omega_\Lambda$ relation")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.grid()
plt.tight_layout()
plt.savefig('Ode-H0.png', bbox_inches='tight')
plt.show()