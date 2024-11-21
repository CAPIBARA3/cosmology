import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
dirpath = os.path.dirname(os.path.abspath(__file__))

# Given data
values = [69.96, 72.6, 67.4, 68.52, 70.0, 72.0]
errors = [1.05, 2.0, 0.5, 0.62, 10.0, 8.0]
methods = ['Cepheid, JAGB, TRGB', 'Cepheid, JAGB, TRGB, Ia SNe', 'CMB', 'BAO', 'GW', 'Cepheid, Ia SNe, TF, SBF, II SNe']
author =['CCHP Program', 'SH0ES Collaboration', 'Planck Collaboration', 'DESI Collaboration', 'LIGO Collaboration', 'HST Key Project']

# Define colors for different methods
colors = {'Cepheid, JAGB, TRGB': 'blue',
        'Cepheid, JAGB, TRGB, Ia SNe': 'red',
        'CMB': 'black',
        'BAO': 'orange',
        'GW': 'green',
        'Cepheid, Ia SNe, TF, SBF, II SNe':  'purple'
}

# Create a range of Hubble constant values
x_range = np.linspace(65, 80, 1000)

# Plot the probability density function (PDF) for each method
plt.figure()# figsize=(8, 6))

for i, method in enumerate(methods):
    # Assuming a normal distribution for the Hubble constant values
    mean = values[i]
    std_dev = errors[i]
    
    # Get the PDF from a normal distribution
    pdf = norm.pdf(x_range, mean, std_dev)
    
    # Plot the PDF with the method-specific color and label
    plt.plot(x_range, pdf, color=colors[method], label=f'{method} ({author[i]})')

# Customize the plot
plt.xlabel("Hubble Constant (km/s/Mpc)")
plt.ylabel("Probability Density")
plt.title("Probability Density Function of Hubble Constant by Method")
plt.legend()
plt.grid(True)
plt.show()