import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.values())
import pandas as pd


def mean_cov(x, y, x_sig, y_sig):
    """Calculate weighted mean and covariance considering errors."""
    x_weights = 1 / (x_sig ** 2)  # weights are inverse of variance
    y_weights = 1 / (y_sig ** 2)

    # Normalize weights
    x_weights /= np.sum(x_weights)
    y_weights /= np.sum(y_weights)

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
    """Plot a confidence ellipse based on the mean and covariance."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.arctan2(*eigenvectors[:, 0][::-1])
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(mean, width, height, angle=np.degrees(angle), **kwargs)
    ax.add_patch(ellipse)

def plot_confidence_region(ax, x_data, y_data, std_dev=[1, 2], **kwargs):
    """Plot confidence regions based on data."""
    x, xerr = np.array(x_data[0]), np.array(x_data[1])
    y, yerr = np.array(y_data[0]), np.array(y_data[1])
    print(x, xerr)
    print(y, yerr)
    mean, cov = mean_cov(x, y, xerr, yerr)

    for i in std_dev:
        plot_confidence_ellipse(ax, mean, cov, n_std=i, fill=True, alpha=0.5 / i, **kwargs)


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


# Main execution
rd = read_data()

H0, H0_err = rd.get_hubble()
Om, Om_err = rd.get_matter()
Ode, Ode_err = rd.get_lambda()

plt.figure(figsize=(8, 6))



for i, name, year in zip(range(len(H0)), rd.get_dataset(), rd.get_year()):

    plt.errorbar(Om[i], Ode[i], xerr=Om_err[i], yerr=Ode_err[i], label=name+f' ({year})', fmt='o', capsize=5, color=colors[i])

# without discriminating over detection types
mean, cov = mean_cov(Om, Ode, Om_err, Ode_err)
plot_confidence_ellipse(plt.gca(), mean, cov, n_std=1, edgecolor='green', facecolor='green', fill=True, alpha=0.6, label='1σ Confidence Region')
plot_confidence_ellipse(plt.gca(), mean, cov, n_std=2, edgecolor='green', facecolor='green', fill=True, alpha=0.2, label='2σ Confidence Region')

# Set limits and labels
# plt.xlim(63, 76)
# plt.ylim(0.225, 0.38)
plt.xlabel('$Omega_m$')
plt.ylabel('$Omega_{de}$')
plt.title('Confidence Regions for Om and Ode')
plt.legend()
plt.grid()
plt.show()

# Combined,DESI Collaboration,2025,DESI+CMB+Pantheon+,67.97,0.57,0.3047,0.0051,0.6953,0.0051,2503.14738
# Combined,DESI Collaboration,2025,DESI+CMB+Union3,68.01,0.68,0.3044,0.0059,0.6956,0.0059,2503.14738
# Combined,DESI Collaboration,2025,DESI+CMB+DESY5,67.34,0.54,0.3098,0.005,0.6902,0.005,2503.14738