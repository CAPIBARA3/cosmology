import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import norm
import matplotlib.colors as mcolors
colors = list(mcolors.BASE_COLORS.values())

class stats():
    def mean_cov(self, x, y, x_sig, y_sig):
        # for item in [x,y,x_sig,y_sig]:
        #     if not isinstance(item, list):
        #         raise KeyError('All arguments must be lists.')
                # todo: convert to list instead of raising error
        """Calculate weighted mean and covariance considering errors."""
        x_weights = [1 / (value ** 2) for value in x_sig] # weights are inverse of variance
        y_weights = [1 / (value ** 2) for value in y_sig]

        # Normalize weights
        x_weights = [value / np.sum(x_weights) for value in x_weights]
        y_weights = [value / np.sum(y_weights) for value in y_weights]

        # Calculate weighted mean
        mean_x, mean_y = 0,0
        for weight, value in zip(x_weights, x):
            mean_x += weight * value
        for weight, value in zip(y_weights, y):
            mean_y += weight * value
        mean = np.array([mean_x, mean_y])

        # Calculate weighted covariance
        cov = np.zeros((2, 2))
        for i in range(len(x)): # TODO: make this function (overall) working for np.arrays and for lists (for both if possible)
            cov[0, 0] += x_weights[i] * (x[i] - mean_x) ** 2
            cov[1, 1] += y_weights[i] * (y[i] - mean_y) ** 2
            cov[0, 1] += x_weights[i] * y_weights[i] * (x[i] - mean_x) * (y[i] - mean_y)

        # Since covariance matrix is symmetric
        cov[1, 0] = cov[0, 1]

        return mean, cov

    def plot_confidence_ellipse(self, ax, mean, cov, n_std=1.0, **kwargs):
        """Plot a confidence ellipse based on the mean and covariance."""
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.arctan2(*eigenvectors[:, 0][::-1])
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ellipse = Ellipse(mean, width, height, angle=np.degrees(angle), **kwargs)
        ax.add_patch(ellipse)

    def plot_confidence_region(self, ax, x_data, y_data, std_dev=[1, 2], **kwargs):
        """Plot confidence regions based on data."""
        x, xerr = np.array(x_data[0]), np.array(x_data[1])
        y, yerr = np.array(y_data[0]), np.array(y_data[1])
        mean, cov = self.mean_cov(x, y, xerr, yerr)

        for i in std_dev:
            self.plot_confidence_ellipse(ax, mean, cov, n_std=i, fill=True, alpha=0.5 / i, **kwargs)

    def pdf(self, ax, data, **kwargs):
        values, errors = data
        for i, (mean, sigma) in enumerate(zip(values, errors)):
            x = np.linspace(mean - sigma*4, mean + sigma*4, 1000) # array for X values
            pdf = norm.pdf(x, loc=mean, scale=sigma) # array for PDF results

            # plot probability density function
            ax.plot(x, pdf, color=colors[i], **kwargs)
            ax.fill_between(x, pdf, color=colors[i], alpha=0.2)