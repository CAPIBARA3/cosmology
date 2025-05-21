import numpy as np

H0 = 70  # or whatever value in km/s/Mpc
z = np.linspace(0.01, 10, 100)  # a range of redshifts
OmegaM = 0.3
OmegaDE = 0.7
OmegaR = 0.0  # if ignoring radiation, else something like ~1e-4
