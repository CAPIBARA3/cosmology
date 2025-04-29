from parameters import H0, z, OmegaM, OmegaDE, OmegaR
import numpy as np
from scipy.integrate import quad
import pandas as pd

# Constants
c = 299792.458  # Speed of light in km/s
to_mpc = 3.0857e19  # km to Mpc
to_gyr = 3.1536e16  # s to Gyr

# Ensure z is a numpy array
z = np.array(z)
x = 1 / (1 + z)  # Scale factor a = 1 / (1 + z)
q_0 = 0.5*OmegaM + OmegaR - OmegaDE

# Curvature density
OmegaK = 1.0 - (OmegaM + OmegaDE + OmegaR)

# =========================
# Cosmological Integrands
# =========================

def integrand_comoving(x_val, OmegaK):
    H_squared = OmegaDE + OmegaM * x_val**-3 + OmegaR * x_val**-4 + OmegaK * x_val**-2
    return 1.0 / (x_val**2 * np.sqrt(H_squared))

def comoving_distance_scalar(x_val, OmegaK):
    integral, _ = quad(integrand_comoving, x_val, 1, args=(OmegaK,))
    return (c / H0) * integral  # in km

def comoving_distance(OmegaK):
    return np.array([comoving_distance_scalar(xi, OmegaK) for xi in x])  # in km

def proper_distance(OmegaK):
    com_dist = comoving_distance(OmegaK)  # in km
    chi = (H0 / c) * com_dist  # dimensionless
    sqrt_ok = np.sqrt(np.abs(OmegaK))
    
    if OmegaK < 0:
        return (c / H0) * np.sin(sqrt_ok * chi) / sqrt_ok
    elif OmegaK > 0:
        return (c / H0) * np.sinh(sqrt_ok * chi) / sqrt_ok
    else:
        return com_dist  # flat universe

def angular_distance(OmegaK):
    return (1 / (1 + z)) * proper_distance(OmegaK)

def luminosity_distance(OmegaK):
    return (1 + z) * proper_distance(OmegaK)

def time_integrand(x_val, OmegaK):
    H_squared = OmegaDE + OmegaM * x_val**-3 + OmegaR * x_val**-4 + OmegaK * x_val**-2
    return 1.0 / (x_val * np.sqrt(H_squared))

# Convert H0 from km/s/Mpc to 1/s
H0_SI = H0 / 3.0857e19  # s^-1

def light_travel_time_scalar(x_val, OmegaK):
    integral, _ = quad(time_integrand, x_val, 1, args=(OmegaK,))
    return (1 / H0_SI) * integral  # in seconds

def light_travel_time(OmegaK):
    return np.array([light_travel_time_scalar(xi, OmegaK) for xi in x])  # in seconds

def big_bang_to_z_travel_time_scalar(x_val, OmegaK):
    integral, _ = quad(time_integrand, 0, x_val, args=(OmegaK,))
    return (1 / H0_SI) * integral  # in seconds

def big_bang_to_z_travel_time(OmegaK):
    return np.array([big_bang_to_z_travel_time_scalar(xi, OmegaK) for xi in x])  # in seconds

def universe_time(OmegaK):
    integral, _ = quad(time_integrand, 0, 1, args=(OmegaK,))
    return (1 / H0_SI) * integral  # in seconds


# =========================
# Data Table Construction
# =========================

data = {
    'Scale Factor (a)': [],
    'Curvature (k)': [],
    'Redshift (z)': [],
    'Comoving Distance (Mpc)': [],
    'Proper Distance (Mpc)': [],
    'Angular Distance (Mpc)': [],
    'Luminosity Distance (Mpc)': [],
    'Light Travel Time (Gyr)': [],
    'Big Bang to z Travel Time (Gyr)': [],
    'Universe Time (Gyr)': []
}

def append_data_for_curvature(OmegaK, label):
    # scale factor is dimensionless
    com_dist = comoving_distance(OmegaK) / to_mpc
    prop_dist = proper_distance(OmegaK) / to_mpc
    ang_dist = angular_distance(OmegaK) / to_mpc
    lum_dist = luminosity_distance(OmegaK) / to_mpc
    ltt = light_travel_time(OmegaK) / to_gyr
    bbtt = big_bang_to_z_travel_time(OmegaK) / to_gyr
    ut = universe_time(OmegaK) / to_gyr

    n = len(z)
    data['Scale Factor (a)'].extend(x.tolist())
    data['Curvature (k)'].extend([label] * n)
    data['Redshift (z)'].extend(z.tolist())
    data['Comoving Distance (Mpc)'].extend(com_dist.tolist())
    data['Proper Distance (Mpc)'].extend(prop_dist.tolist())
    data['Angular Distance (Mpc)'].extend(ang_dist.tolist())
    data['Luminosity Distance (Mpc)'].extend(lum_dist.tolist())
    data['Light Travel Time (Gyr)'].extend(ltt.tolist())
    data['Big Bang to z Travel Time (Gyr)'].extend(bbtt.tolist())
    data['Universe Time (Gyr)'].extend([ut] * n)

# Run calculations
append_data_for_curvature(0, 'k = 0')
# append_data_for_curvature(-0.1, 'k < 0')
# append_data_for_curvature(0.1, 'k > 0')

# Create and save table
df = pd.DataFrame(data)
print(df.head())  # quick sanity check
df.to_csv('cosmological_distances.csv', index=False)
