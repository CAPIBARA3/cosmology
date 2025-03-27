import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors

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

# Your data
names = ['Planck', 'CCHP', 'SHOES', 'DESI', 'JiC 2023', 'JiC 2024']
H0 = np.array([67.4, 70.39, 73.04, 68.52, 70.26, 69.49])
H0_err = np.array([0.5, 3.25, 1.04, 0.62, 7.09, 3.12])
Om = np.array([0.315, 0.3, 0.3, 0.307, 0.3, 0.3])
Om_err = np.array([0.107, 0.1, 0.1, 0.105, 0.1, 0.1])

BG_H0 = np.array([67.4, 68.52])  # CMB and BAO
BG_H0_err = np.array([0.5, 0.62])
BG_Om = np.array([0.315, 0.307])
BG_Om_err = np.array([0.107, 0.105])

DL_H0 = np.array([70.39, 73.04, 70.26, 69.49])  # Distance ladder methods
DL_H0_err = np.array([3.25, 1.04, 7.09, 3.12])
DL_Om = np.array([0.3, 0.29, 0.3, 0.35])
DL_Om_err = np.array([0.01, 0.01, 0.01, 0.01])

# Create the figure
plt.figure(figsize=(8, 6))
colors = list(mcolors.TABLEAU_COLORS.values())[:len(names)]  # Ensure the number of colors matches the number of points
plt.errorbar(H0, Om, xerr=H0_err, yerr=Om_err, fmt='o', label=' Data Points', capsize=5, color='gray')

for i, (H0_data, Om_data, H0_err_data, Om_err_data) in enumerate(zip((BG_H0, DL_H0), (BG_Om, DL_Om), (BG_H0_err, DL_H0_err), (BG_Om_err, DL_Om_err))):
    mean, cov = mean_cov(H0_data, Om_data, H0_err_data, Om_err_data)
    plot_confidence_ellipse(plt.gca(), mean, cov, n_std=1, edgecolor=colors[i], facecolor=colors[i], fill=True, alpha=0.6, label=f"$1 \sigma$ {names[i]}")
    plot_confidence_ellipse(plt.gca(), mean, cov, n_std=2, edgecolor=colors[i], facecolor=colors[i], fill=True, alpha=0.2, label=f"$2 \sigma$ {names[i]}")

# Annotate points with names
for i, name in enumerate(names):
    plt.annotate(name, (H0[i], Om[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Set limits and labels
plt.xlim(65, 75)
plt.ylim(0.20, 0.40)
plt.xlabel('$H_0$ (km/s/Mpc)')
plt.ylabel('$\Omega_m$')
plt.title('Confidence Regions for $H_0$ and $\Omega_m$')
plt.legend()
plt.grid()
plt.show()