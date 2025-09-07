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
    H0 = (np.array([67.4, 70.39, 73.04, 68.52, 70.26, 69.49]),
          np.array([0.5, 3.25, 1.04, 0.62, 7.09, 3.12]))

    Om = (np.array([0.315, 0.3, 0.3, 0.307, 0.3, 0.3]),
          np.array([0.007, 0.01, 0.01, 0.005, 0.01, 0.01]))

    names = ['Planck', 'CCHP', 'SHOES', 'DESI', 'JiC 2023', 'JiC 2024']

    return H0, Om

def CB_data():
    # data from CMB and BAO
    H0 = (np.array([67.4, 68.52]),
        np.array([0.5, 0.62]))

    Om = (np.array([0.315, 0.307]),
        np.array([0.107,0.105]))

    names = ['Planck', 'DESI']

    return H0, Om

def DL_data():
    # Data from distance ladder methods
    H0 = (
        np.array([70.39, 73.04, 70.26, 69.49]),
        np.array([3.25, 1.04, 7.09, 3.12])
    )

    Om = (
        np.array([0.3, 0.29, 0.3, 0.35]),
        np.array([0.01, 0.01, 0.01, 0.01]))

    names = ['CCHP', 'SHOES', 'JiC 2023', 'JiC 2024']

    return H0, Om

names = ['Planck', 'CCHP', 'SHOES', 'DESI', 'JiC 2023', 'JiC 2024']

# Create the figure
colors = list(mcolors.TABLEAU_COLORS.values())  # Ensure the number of colors matches the number of points

plt.figure(figsize=(8, 6))
H0, Om = data()
for i, name in enumerate(names):
    plt.errorbar(H0[0][i], Om[0][i], xerr=H0[1][i], yerr=Om[1][i], fmt='o', label=name, capsize=5, color=colors[i])

# for i, name in enumerate(names):
#     plt.annotate(name, H0[0][i], Om[0][i], textcoords="offset points", xytext=(0,10), ha='center')

for i, data in enumerate([data(), CB_data(), DL_data()]):
    sample_id = ['general', 'CMB-BAO', 'DL']
    plot_confidence_region(plt.gca(), data[0], data[1], sample_id[i], std_dev=[1, 2], edgecolor=colors[i], facecolor=colors[i])


# for i, (H0_data, Om_data, H0_err_data, Om_err_data) in enumerate(zip((BG_H0, DL_H0), (BG_Om, DL_Om), (BG_H0_err, DL_H0_err), (BG_Om_err, DL_Om_err))):
#     mean, cov = mean_cov(H0_data, Om_data, H0_err_data, Om_err_data)
#     plot_confidence_ellipse(plt.gca(), mean, cov, n_std=1, edgecolor=colors[i], facecolor=colors[i], fill=True, alpha=0.6, label=f"$1 \sigma$ {names[i]}")
#     plot_confidence_ellipse(plt.gca(), mean, cov, n_std=2, edgecolor=colors[i], facecolor=colors[i], fill=True, alpha=0.2, label=f"$2 \sigma$ {names[i]}")

import pandas as pd
class read_data:
    def __init__(self, file_path="./dataset.csv"):
        try:
            self.df = pd.read_csv(file_path)
            self.hubble = self.df['H0']
            self.hubble_sigma = self.df['sigmaH0']
            self.Om = self.df['Om']
            self.Om_sigma = self.df['sigmaOm']
            self.Ode = self.df['Ode']
            self.Ode_sigma = self.df['sigmaOde']
            self.dataset = self.df['Dataset']
            self.year = self.df['Year']
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            self.df = None
        except Exception as e:
            print(f"An error occurred: {e}")
            self.df = None

    def get_hubble(self):
        return (np.array(self.hubble), np.array(self.hubble_sigma))

    def get_matter(self):
        return (np.array(self.Om), np.array(self.Om_sigma))

    def get_lambda(self):
        return (np.array(self.Ode), np.array(self.Ode_sigma))

    def get_detection(self):
        return np.array(self.detections)

    def get_dataset(self):
        return self.dataset

    def get_year(self):
        return self.year


rd = read_data()
H0, H0_err = rd.get_hubble()
Om, Om_err = rd.get_matter()

for i, name, year in zip(range(len(H0)), rd.get_dataset(), rd.get_year()):
    plt.errorbar(H0[i], Om[i], xerr=H0_err[i], yerr=Om_err[i], label=name+f' ({year})', fmt='o', capsize=5, color=colors[i])

# without discriminating over detection types
mean, cov = mean_cov(H0, Om, H0_err, Om_err)
plot_confidence_ellipse(plt.gca(), mean, cov, n_std=1, edgecolor='red', facecolor='red', fill=True, alpha=0.6, label='1σ Confidence Region')
plot_confidence_ellipse(plt.gca(), mean, cov, n_std=2, edgecolor='red', facecolor='red', fill=True, alpha=0.2, label='2σ Confidence Region')

# Set limits and labels
plt.xlim(64, 76)
plt.ylim(0.26, 0.36)
plt.xlabel('$H_0$ (km/s/Mpc)')
plt.ylabel('$\Omega_m$')
plt.title('Confidence Regions for $H_0$ and $\Omega_m$')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.savefig('Om-H0.png')
plt.show()
