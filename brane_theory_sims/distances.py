import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import numpy as np

# === Constants ===
c_km_s = 299792.458  # speed of light in km/s
H0 = 70.0  # Hubble constant in km/s/Mpc
H0_SI = H0 / (3.0857e19)  # Convert H0 to s^-1 (1 Mpc = 3.0857e19 km)
G = 6.67430e-11  # gravitational constant in m^3/kg/s^2
rho_crit_0 = 3 * H0_SI**2 / (8 * np.pi * G)  # critical density today in kg/m^3

# === Redshift and Scale Factor ===
z = np.linspace(0, 14, 1000)
a = 1 / (1 + z)

# === Cosmological Parameters ===
Om0 = 0.3
Or0 = 0.001
Ode0 = 0.7
OmegaK = 0.0  # Assume flatness

# === CPL Dark Energy Evolution ===
w0 = -1.0
wa = 0.2

# === Energy Densities ===
rho_m = Om0 * rho_crit_0 * a**-3
rho_r = Or0 * rho_crit_0 * a**-4
rho_total = rho_m + rho_r

# Dark energy density as a function of redshift (CPL parameterization)
def rho_de_z(z):
    return Ode0 * rho_crit_0 * (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))

rho_de = rho_de_z(z)


class BraneWorld:
    def __init__(self, z, rho_total, rho_de, sigma_frac=1e-8):
        self.z = z
        self.a = 1 / (1 + z)
        self.rho_total = rho_total
        self.rho_de = rho_de
        self.sigma = sigma_frac * rho_crit_0

        # Compute brane correction and Hubble parameter
        self.brane_term = 1 + self.rho_total / (2 * self.sigma)
        self.H2 = (8 * np.pi * G / 3) * (self.rho_total + self.rho_de) * self.brane_term
        self.H_brane = np.sqrt(self.H2) * (3.0857e19 / 1000)  # Convert to km/s/Mpc

    def H(self):
        return self.H_brane

    def comoving_distance(self):
        chi = cumulative_trapezoid(c_km_s / self.H_brane, self.z, initial=0)
        return chi

    def luminosity_distance(self):
        chi = self.comoving_distance()
        return (1 + self.z) * chi

    def proper_distance(self):
        return self.comoving_distance()  # In flat cosmology, same as comoving

    def angular_distance(self):
        chi = self.comoving_distance()
        return chi / (1 + self.z)

class LCDM:
    def __init__(self, z, H0=70.0, OmegaM=0.3, OmegaR=0.001, OmegaDE=0.7):
        self.z = z
        self.a = 1 / (1 + z)
        self.H0 = H0  # km/s/Mpc
        self.OmegaM = OmegaM
        self.OmegaR = OmegaR
        self.OmegaDE = OmegaDE
        self.OmegaK = 1.0 - (OmegaM + OmegaR + OmegaDE)

        # Compute Hubble parameter as a function of z
        self.H_array = self.H_func()

    def H_func(self):
        density = (self.OmegaM * self.a**-3 +
                   self.OmegaR * self.a**-4 +
                   self.OmegaDE +
                   self.OmegaK * self.a**-2)
        return self.H0 * np.sqrt(density)

    def H(self):
        return self.H_array

    def comoving_distance(self):
        chi = cumulative_trapezoid(c_km_s / self.H_array, self.z, initial=0)
        return chi

    def luminosity_distance(self):
        return (1 + self.z) * self.comoving_distance()

    def proper_distance(self):
        return self.comoving_distance()

    def angular_distance(self):
        return self.comoving_distance() / (1 + self.z)

bw = BraneWorld(z, rho_total, rho_de)
lcdm = LCDM(z, H0=H0, OmegaM=Om0, OmegaR=Or0, OmegaDE=Ode0)

# initiate figure
plt.figure(figsize=(10, 5))
labels = ['proper', 'luminosity', 'angular']
import matplotlib.colors as mcolors
colors = list(mcolors.BASE_COLORS.keys())


# brane-world lines
for i, data in enumerate([bw.proper_distance(), bw.luminosity_distance(), bw.angular_distance()]):
    plt.plot(z, data, label=f'brane-theory {labels[i]}', color=colors[i])

# LCDM lines
for i, data in enumerate([lcdm.proper_distance(), lcdm.luminosity_distance(), lcdm.angular_distance()]):
    plt.plot(z, data, label=fr'$\Lambda$CDM {labels[i]}', color=colors[i], linestyle='--')

# figure settings
plt.xlabel('Redshift z')
plt.ylabel('Distances [Mpc]')
plt.title('Distance vs Redshift')
plt.legend()
plt.yscale('log')
plt.grid(True)


plt.tight_layout()
plt.show()