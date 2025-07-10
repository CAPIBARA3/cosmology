import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
dirpath = os.path.dirname(os.path.abspath(__file__))

# Given data
values = [72.0, 70.0, 67.4, 68.52, 69.96, 72.6, 70.26, 69.49]
errors = [8.0, 10.0, 0.5, 0.62, 1.05, 2.0, 7.1, 1.6]
methods = ['Cepheid, Ia SNe, TF, SBF, II SNe', 'GW', 'CMB', 'BAO', 'Cepheid, JAGB, TRGB', 'Cepheid, JAGB, TRGB, Ia SNe', 'Cepheid', 'Ia SNe']
author = [r'$\mathbf{2001}$ HST Key Project', r'$\mathbf{2017}$ LIGO Collaboration', r'$\mathbf{2018}$ Planck Collaboration', r'$\mathbf{2024}$ DESI Collaboration', r'$\mathbf{2024}$ CCHP Program', r'$\mathbf{2024}$ SH0ES Collaboration', r'$\mathbf{2023\ Alcaide-Núñez}$', r'$\mathbf{2024\ Alcaide-Núñez}$']

#'2001 HST Key Project
# Define colors for different methods
colors = {'Cepheid, JAGB, TRGB': 'blue',
        'Cepheid, JAGB, TRGB, Ia SNe': 'red',
        'CMB': 'black',
        'BAO': 'orange',
        'GW': 'green',
        'Cepheid, Ia SNe, TF, SBF, II SNe':  'purple',
        'Ia SNe': 'magenta',
        'Cepheid': 'lightblue'
}

# Create a range of Hubble constant values
x_range = np.linspace(65, 80, 1000)

# Plot the probability density function (PDF) for each method
plt.figure(figsize=(12, 6))

for i, method in enumerate(methods):
    # Assuming a normal distribution for the Hubble constant values
    mean = values[i]
    std_dev = errors[i]
    
    # Get the PDF from a normal distribution
    pdf = norm.pdf(x_range, mean, std_dev)
    
    # Plot the PDF with the method-specific color and label
    plt.plot(x_range, pdf, color=colors[method], label=f'{author[i]} ({method})')
    plt.fill_between(x_range, pdf, color=colors[method], alpha=0.3)

# Customize the plot
plt.xlabel("Hubble Constant (km/s/Mpc)")
plt.ylabel("Probability Density")
plt.title("Probability Density Function of Hubble Constant by Method")
plt.legend()
plt.grid(True)
plt.show()