import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
# Constants
c_km_s = 299792.458  # speed of light in km/s
H0 = 70.0  # Hubble parameter in km/s/Mpc
H0_SI = H0 / (3.0857e19)  # H0 in 1/s
G = 6.67430e-11  # gravitational constant in m^3/kg/s^2
rho_crit_0 = 3 * H0_SI**2 / (8 * np.pi * G)  # critical density today in kg/m^3

# Redshift and scale factor
z = np.linspace(0, 14, 1000)
a = 1 / (1 + z)

# Cosmological parameters
Om0 = 0.3
Or0 = 0.001
Ode0 = 0.7
OmegaK = 0.0  # Assume flatness

# Dark Energy Evolution - CPL parameterization
w0 = -1.0
wa = 0.2

def w_z(z):
    return w0 + wa * z / (1 + z)

def rho_de_z(z):
    return Ode0 * rho_crit_0 * (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))

# Brane tension values (scaled)
sigmas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
sigmas = [s * rho_crit_0 for s in sigmas]

# Energy densities
rho_m = Om0 * rho_crit_0 * a**-3
rho_r = Or0 * rho_crit_0 * a**-4
rho_total = rho_m + rho_r
rho_de = rho_de_z(z)

# LCDM Hubble function
lcdm_density = Om0 * a**-3 + Or0 * a**-4 + Ode0 + OmegaK * a**-2
H_LCDM = H0 * np.sqrt(lcdm_density)

# Brane cosmology: loop over sigmas
plt.figure(figsize=(10, 6))
for sigma in sigmas:
    brane_term = 1 + rho_total / (2 * sigma)
    H2 = (8 * np.pi * G / 3) * (rho_total + rho_de) * brane_term
    H_brane = np.sqrt(H2) * (3.0857e19 / 1000)  # convert from 1/s to km/s/Mpc
    plt.plot(z, H_brane, label=f'σ = {sigma / rho_crit_0:.0e} × ρ_c0')

plt.plot(z, H_LCDM, '--k', label='ΛCDM')
plt.yscale('log')
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.title('Hubble Parameter in Brane Cosmology with Evolving DE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Brane correction factor
plt.figure(figsize=(10, 4))
for sigma in sigmas:
    brane_term = 1 + rho_total / (2 * sigma)
    plt.plot(z, brane_term, label=f'σ = {sigma / rho_crit_0:.0e} × ρ_c0')
plt.xlabel('Redshift z')
plt.ylabel(r'$1 + \rho/(2\sigma)$')
plt.title('Brane Correction Factor')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.tight_layout()
plt.show()