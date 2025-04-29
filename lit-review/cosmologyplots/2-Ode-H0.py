import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors


def mean_cov_without_errors(x, y):
    return np.mean(x, y), np.cov(x, y)

def mean_cov(x, y, x_sig, y_sig):
    x_weights = 1 / (x_sig**2) # weights are inverse of variance
    y_weights = 1 / (y_sig**2)

    # Normalize weights
    x_weights = x_weights / np.sum(x_weights)
    y_weights = y_weights / np.sum(y_weights)

    # Calculate weighted mean
    mean_x = np.sum(x_weights * x)
    mean_y = np.sum(y_weights * y)
    mean = np.array([mean_x, mean_y])

    # Calculate weighted covariance
    cov = np.zeros((2, 2))
    for i in range(len(x)):
        cov[0, 0] += x_weights[i] * (x[i] - mean_x) ** 2
        cov[1, 1] += y_weights[i] * (y[i] - mean_y) ** 2
        cov[0, 1] += x_weights[i] * y_weights[i] * (x[i] - mean_x) * (y[i] - mean_y)

    # Since covariance matrix is symmetric
    cov[1, 0] = cov[0, 1]

    return mean, cov

def plot_confidence_ellipse(ax, mean, cov, n_std=1.0, **kwargs):
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Calculate the angle of the ellipse
    angle = np.arctan2(*eigenvectors[:, 0][::-1])
    # Width and height of the ellipse
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    # Create the ellipse
    ellipse = Ellipse(mean, width, height, angle=np.degrees(angle), **kwargs)
    ax.add_patch(ellipse)

def plot_confidence_region(ax, x_data, y_data, label, std_dev=[1, 2], **kwargs):
    if isinstance(x_data, tuple):
        x, xerr = np.asarray(x_data[0]), np.asarray(x_data[1])
        y, yerr = np.asarray(y_data[0]), np.asarray(y_data[1])
        mean, cov = mean_cov(x, y, xerr, yerr)
    else:
        mean, cov = mean_cov_without_errors(x_data, y_data)

    for i in std_dev:
        if i != 1:
            plot_confidence_ellipse(ax, mean, cov, n_std=i, fill=True, alpha=0.5 / i, **kwargs)
        else:
            plot_confidence_ellipse(ax, mean, cov, n_std=i, fill=True, alpha=0.5/i, label=f'{label}', **kwargs)


def data():
    H0 = ([73.45, 67.4, 70.4, 67.6, 67.7, 69.51], [1.66, 0.5, 1.5, 1.2, 0.4, 0.92])
    Ode = ([0.7, 0.685, 0.71, 0.69, 0.685, 0.7073], [0.02, 0.007, 0.015, 0.01, 0.007, 0.0073])

    name = ['Local Hubble Measurement', 'CMB (Planck)', 'BAO (SDSS)', 'BAO (BOSS)', 'CMB (ACT)', 'DESI+CMB (OdeegaCDM)']

    return H0, Ode

def early_data():
    # data from CMB and BAO

    H0 = ([67.4, 70.4, 67.6, 67.7, 69.51], [0.5, 1.5, 1.2, 0.4, 0.92])
    Ode = ([0.685, 0.71, 0.69, 0.685, 0.7073], [0.007, 0.015, 0.01, 0.007, 0.0073])

    name = ['CMB (Planck)', 'BAO (SDSS)', 'BAO (BOSS)', 'CMB (ACT)', 'DESI+CMB (OdeegaCDM)']
    year = [2020, 2005, 2016, 2025, 2025]

    return H0, Ode

def local_data():
    # Data from distance ladder methods
    H0 = ([73.45], [1.66])
    Ode = ([0.7], [0.02])

    name = ['Local Hubble Measurement']
    year = [2018]

    return H0, Ode

names = ['Local Hubble Measurement', 'CMB (Planck)', 'BAO (SDSS)', 'BAO (BOSS)', 'CMB (ACT)', 'DESI+CMB (OdeegaCDM)']

# Create the figure
colors = list(mcolors.TABLEAU_COLORS.values())

plt.figure(figsize=(8, 6))
H0, Ode = data()
for i, name in enumerate(names):
    plt.errorbar(H0[0][i], Ode[0][i], xerr=H0[1][i], yerr=Ode[1][i], fmt='o', label=name, capsize=5, color=colors[i])

# for i, name in enumerate(names):
#     plt.annotate(name, H0[0][i], Ode[0][i], textcoords="offset points", xytext=(0,10), ha='center')

for i, data in enumerate([data(), early_data(), local_data()]):
    sample_id = ['general', 'CMB-BAO', 'local']
    plot_confidence_region(plt.gca(), data[0], data[1], sample_id[i], std_dev=[1, 2], edgecolor=colors[i], facecolor=colors[i])


# for i, (H0_data, Ode_data, H0_err_data, Ode_err_data) in enumerate(zip((BG_H0, DL_H0), (BG_Ode, DL_Ode), (BG_H0_err, DL_H0_err), (BG_Ode_err, DL_Ode_err))):
#     mean, cov = mean_cov(H0_data, Ode_data, H0_err_data, Ode_err_data)
#     plot_confidence_ellipse(plt.gca(), mean, cov, n_std=1, edgecolor=colors[i], facecolor=colors[i], fill=True, alpha=0.6, label=f"$1 \sigma$ {names[i]}")
#     plot_confidence_ellipse(plt.gca(), mean, cov, n_std=2, edgecolor=colors[i], facecolor=colors[i], fill=True, alpha=0.2, label=f"$2 \sigma$ {names[i]}")

my_H0 = ([69.49], [3.12])
my_Ode = ([2.13], [0.01])
plt.errorbar(my_H0[0], my_Ode[0], xerr=my_H0[1], yerr=my_Ode[1])

# Set limits and labels
plt.xlim(64, 76)
#plt.ylim(0.6, 0.8)
plt.yscale('log')
plt.xlabel('$H_0$ (km/s/Mpc)')
plt.ylabel('$Omega_{de}$')
plt.title('Confidence Regions for $H_0$ and $\Omega_{de}$')
plt.legend() #loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.show()


plt.savefig('Ode-H0.png')