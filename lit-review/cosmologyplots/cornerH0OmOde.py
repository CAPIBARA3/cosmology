# import general libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.values())
import os
dirpath = os.path.dirname(os.path.abspath(__file__))


# import modules
from readingCSV import read_data
rd = read_data()

from stats_plot import stats
stats = stats()


H0, sigmaH0 = rd.get_hubble()
Om, sigmaOm = rd.get_matter()
Ode, sigmaOde = rd.get_lambda()

fig, axs = plt.subplots(3, 3)

# 0 0       1       2
# 0 PDF     Blanck  Blanck
# 1 contour PDF     Blanck
# 2 contour contour PDF

# ploting PDFS
stats.pdf(axs[0,0], (H0, sigmaH0))
stats.pdf(axs[1,1], (Om, sigmaOm))
stats.pdf(axs[2,2], (Ode, sigmaOde))


# contour plots
for i in range(len(H0)):
    stats.plot_confidence_region(axs[1,0], (H0, sigmaH0), (Om, sigmaOm), edgecolor=colors[i], facecolor=colors[i])
    stats.plot_confidence_region(axs[2,0], (H0, sigmaH0), (Ode, sigmaOde), edgecolor=colors[i], facecolor=colors[i])
    stats.plot_confidence_region(axs[2,1], (Om, sigmaOm), (Ode, sigmaOde), edgecolor=colors[i], facecolor=colors[i])

# set labels
axs[0,0].set_ylabel('H0')
axs[1,0].set_ylabel('Omega_m')
axs[2,0].set_ylabel('Omega_de')
axs[2,0].set_xlabel('H0')
axs[2,1].set_xlabel('Omega_m')
axs[2,2].set_xlabel('Omega_de')

# set limits
axs[0,0].set_xlim(65, 75) #H0
axs[0,0].set_ylim(0, 1) # PDF
axs[1,0].set_xlim(65, 75) #H0
axs[1,0].set_ylim(0.25, 0.35) # Om
axs[2,0].set_xlim(65, 75) #H0
axs[2,0].set_ylim(0.65, 0.75) # Ode

#axs[1,1].set_xlim(0.25, 0.35) # Om
axs[1,1].set_ylim(0, 1) # PDF
axs[2,1].set_xlim(0.25, .35) # Om
axs[2,1].set_ylim(0.65, 0.75) # Ode

#axs[2,2].set_xlim(0.65, 0.75) # Ode
axs[2,2].set_ylim(0, 1) # PDF

plt.show()