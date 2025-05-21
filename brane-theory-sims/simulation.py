import numpy as np
import matplotlib.pyplot as plt

# parameters
H0 = 70.0  # km/s/Mpc
z = np.linspace(0, 14, 1000)
a = 1 / (1 + z)

# Density parameters
Om0 = 0.3
Or0 = 0.001
Ode0 = 0.7

# Hubble in km/s/Mpc to s^-1
H0_SI = H0 / (3.0857e19)  # 1 Mpc in km
G = 6.67430e-11  # m^3 kg^-1 s^-2

# Critical density in SI
p_c0 = 3 * H0_SI**2 / (8 * np.pi * G)

sigma = 1e-9 * p_c0  # for the brane-world correction to the friedmann equation

# Energy densities
rho_m = Om0 * p_c0 * a**-3
rho_r = Or0 * p_c0 * a**-4
rho_total = rho_m + rho_r

# Brane term
brane_term = 1 + (rho_total) / (2 * sigma)

# Hubble parameter squared (SI)
H2 = (8 * np.pi * G / 3) * rho_total * brane_term
H = np.sqrt(H2) * (3.0857e19 / 1000)  # convert s^-1 to km/s/Mpc

# LCDM comparison
OmegaDE = 0.7
OmegaM = 0.3
OmegaR = 0.001
OmegaK = 1.0 - (OmegaM + OmegaDE + OmegaR)

lcdm_density = OmegaM * a**-3 + OmegaR * a**-4 + OmegaDE + OmegaK * a**-2
H_LCDM = H0 * np.sqrt(lcdm_density)

# Plotting
plt.figure()
plt.plot(z, H, label='Brane Cosmology')
plt.plot(z, H_LCDM, label='ΛCDM')
plt.yscale('log')
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.legend()
plt.title('Hubble Parameter vs Redshift')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(z, brane_term, label='Brane Cosmology Correction')
plt.xlabel('Redshift z')
plt.ylabel(r'$1+\frac{\rho}{2\sigma}$')
plt.legend()
plt.title('Brane Cosmology Correction')
plt.grid(True)
plt.show()

