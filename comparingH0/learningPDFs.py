import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import os
dirpath = os.path.dirname(os.path.abspath(__file__))
# data_path = dirpath+'/ICE_FASE2JiC/dataset.csv'
data_path = "/Users/joanalnu/Library/Mobile Documents/com~apple~CloudDocs/ICE_FASE2JiC/dataset.csv"

# data input
df = pd.read_csv(data_path)
means = df['Value']
std_dev_lows = df['Lower']
std_dev_highs = df['Upper']
types = df['Type']
colors = list(types.replace({'Cepheids':'red', 'Cepheids-SNIa':'orange', 'CMB with Planck':'black', 'CMB without Planck':'gold',
'GW related':'green', 'HII galaxies':'pink', 'Lensing related; mass model-dependent':'pink',
'Masers':'pink', 'Miras-SNIa':'pink', 'No CMB; with BBN':'pink', 'Optimistic average':'pink',
'Pl(k) + CMB lensing':'pink','SNII':'pink', 'Surface Brightness Fluctuations':'pink', 'TRGB-SNIa':'firebrick',
'Tully-Fisher Relation':'pink', 'Ultra-conservative; no cepheids; no lensing':'pink',
'BAO':'coral', 'SNIa-BAO':'sandybrown', 'other':'pink', 'SNIa':'orange','TRGB':'firebrick'}))

x_data = list()
y_data = list()
color_data = list()
label_data = list()
for mean, std_dev_low, std_dev_high, color, this_type in zip(means, std_dev_lows, std_dev_highs, colors, types):
    # Generate a rango of values around the mean
    x = np.linspace(mean - 4*std_dev_low, mean + 4*std_dev_high, 1000)
    # Compute the PDf of the normal distribution
    pdf = norm.pdf(x, loc=mean, scale=(std_dev_low+std_dev_high)/2)
    # Plot the PDF
    if std_dev_high<2 or std_dev_low<2:
        x_data.append(x)
        y_data.append(pdf)
        color_data.append(color)
        if this_type not in label_data:
            label_data.append(this_type)
        
plt.figure()
for x, y, color, this_type in zip(x_data, y_data, color_data, label_data):
    plt.plot(x, y, color=color, label=this_type)
    plt.fill_between(x, y, color=color, alpha=0.05)

# Customize the plot
plt.title('Probability Density Function of the Hubble Constant')
plt.xlabel('Hubble Constant $km s^{-1} Mpc^{-1}$')
plt.ylabel('Density')
plt.legend()

# Show the plot
plt.show()
plt.savefig(f'{dirpath}/H0pdf.png')