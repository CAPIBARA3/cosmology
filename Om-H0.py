import matplotlib.pyplot as plt
import numpy as np

# Define the data
names = ['Planck', 'CCHP', 'SHOES', 'DESI', 'Alcaide-Núñez 2023', 'Alcaide-Núñez 2024']
H0 = np.array([67.4, 70.39, 73.04, 68.52, 70.26, 69.49])
H0_err = np.array([0.5, 3.25, 1.04, 0.62, 7.09, 3.12])
Om = np.array([0.315, 0.3, 0.3, 0.307, 0.3, 0.3])
Om_err = np.array([0.007, 0.0, 0.0, 0.005, 0.0, 0.0])

# Create a grid of Omega_m and H0 values
Om_grid = np.linspace(0.2, 0.4, 100)
H0_grid = np.linspace(60, 80, 100)
Om_grid, H0_grid = np.meshgrid(Om_grid, H0_grid)

# Calculate the chi-squared values for each point on the grid
chi2_grid = np.zeros_like(Om_grid)
for i in range(len(names)):
    chi2_grid += ((Om_grid - Om[i]) / Om_err[i])**2 + ((H0_grid - H0[i]) / H0_err[i])**2

# Create the contour plot
plt.figure(figsize=(10, 6))
contour = plt.contourf(Om_grid, H0_grid, chi2_grid, levels=20, cmap='viridis')
plt.colorbar(label=r'$\chi^2$')

# Add contours for 1, 2, and 3 sigma confidence levels
contours = [2.30, 6.18, 11.83]  # 1, 2, and 3 sigma confidence levels for 2 parameters
plt.contour(Om_grid, H0_grid, chi2_grid, levels=contours, colors='k', linestyles='--')

# Plot the data points as scatter points
plt.scatter(Om, H0, color='red', label='Data Points', zorder=5)

# Add labels and title
plt.xlabel(r'$\Omega_m$')
plt.ylabel(r'$H_0$')
plt.title(r'$\Omega_m$ - $H_0$ Contour Plot')
plt.legend(loc='upper left')

# Show the plot
plt.show()