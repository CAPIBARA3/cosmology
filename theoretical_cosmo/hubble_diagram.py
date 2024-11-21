### search for "added afterwards" for changes


# importing libraries
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import xlwings as xw
dirpath = os.path.dirname(os.path.abspath(__file__))
### added afterwards data_path = f'{dirpath}/SNCOSMO/Results-copy'
data_path = f'/Users/j.alcaide/Documents/ICE_FASE2JiC/SNCOSMO/Results-copy' # added afterwards
path_redshift = '/Users/j.alcaide/Desktop/Joves i Ciència/article/Hubble computing/without_outliners/copy_redshift.xlsx'
dist_path = '/Users/j.alcaide/Desktop/Joves i Ciència/article/Hubble computing/without_outliners/copy_I_distances.txt'
def to_rgb(a,b,c):
    return (a/255, b/255, c/255)

# SN
# Read SN data
#df = pd.read_csv(f'{data_path}/COPY_Results_SNCOSMO.csv')
df = pd.read_csv(f'{data_path}/Results_SNCOSMO.csv')
df = df.dropna()

# applying cuts for clear outliners
df = df[df['dm']>0]
df = df[df['dm']<40]
df = df[df['dmerr']>0]
df = df[df['dmerr']<5]

# applying additional cuts
dropping = list()
for index, row in df.iterrows():
    if row[2]>0.1 and row[3]<36:
        dropping.append(index)
    # if row[2]>0.05 and row[2]<0.125:
    #     if row[3] < 36:
    #         dropping.append(index)
df = df.drop(dropping)

# Creating variables
names = np.array(df['name'])
redshifts = np.array(df['redshift'])
dms = np.array(df['dm'])
dm_errs = np.array(df['dmerr'])


# Compute additional magnitudes (velocity & distance)
velocities = redshifts * 299792.458 # km/s
distances = 10**((dms/5)+1) # pc
sigma_distances = distances * np.log(10) / 5 * dm_errs # pc
distances /= 1000000 # to Mpc
sigma_distances /= 1000000 # to Mpc

# remove error in sigma distances (huge, weird errorbars)
new_values = list()
for value in sigma_distances:
    if value > 0.01* (10**6):
        new_values.append(1)
    else:
        new_values.append(value)
sigma_distances = np.array(new_values)



# Compute fits with COSMOLOGY
from astropy.cosmology import FlatLambdaCDM, LambdaCDM

# define models
cosmo_org = FlatLambdaCDM(H0=70, Om0=0.3)
cosmo_manual = FlatLambdaCDM(H0=70, Om0=0.9)
cosmo2 = LambdaCDM(H0=70, Om0=0.3, Ode0=1.7)
cosmo3 = LambdaCDM(H0=50, Om0=0.3, Ode0=0.0)
def model_flatlambdacdm(x_data, H0, Om0):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    distmod = cosmo.distmod(x_data).value
    return distmod

def model_lambdacdm(x_data, H0, Om0, Ode0):
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    distmod = cosmo.distmod(x_data).value
    return distmod

def model_computeHo(x_data, H0):
    cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)
    distmod = cosmo.distmod(x_data).value
    return distmod

# compute best fit models
# model_flatlambdacdmxmxm
guess = [70, 0.3]
bounds = ([67.0, 0.0],[73.0, 1.0])
popt_flat, pcov_flat = curve_fit(model_flatlambdacdm, redshifts, dms, sigma=dm_errs, p0=guess, bounds=bounds)
H0_best_flat, Om0_best_flat = popt_flat
H0_best_flat_err, Om0_best_flat_err = np.sqrt(np.diag(pcov_flat)) # std dev of parameters
print(f'Astropycosmo fit flat:\t H0 = {H0_best_flat} ± {H0_best_flat_err}\n\t\t\t Om0 = {Om0_best_flat} ± {Om0_best_flat_err}\n')

# model_lambdacdm
guess = [70, 0.3, 0.7]
bounds = ([67.0, 0.0, 0.0],[73.0, 1.0, 1.0])
popt_lam, pcov_lam = curve_fit(model_lambdacdm, redshifts, dms, sigma=dm_errs, p0=guess, bounds=bounds)
H0_best_lam, Om0_best_lam, Ode0_best_lam = popt_lam
H0_best_lam_err, Om0_best_lam_err, Ode0_best_lam_err = np.sqrt(np.diag(pcov_lam)) # std dev of parameters
print(f'Astropycosmo fit noflat: H0 = {H0_best_lam} ± {H0_best_lam_err}\n\t\t\t Om0 = {Om0_best_lam} ± {Om0_best_lam_err}\n\t\t\t Ode0 = {Ode0_best_lam} ± {Ode0_best_lam_err}\n')

guess = [70, 0.3, 0.7]
# bounds = ([67.0, 0.2, 0.7],[73.0, 0.4, 0.8]) # restrictive
bounds = ([0.0,0.0,0.0],[1000.0, 1000.0, 1000.0]) # free
# popt_res, pcov_res = curve_fit(model_lambdacdm, redshifts, dms, sigma=dm_errs, p0=guess, bounds=bounds)
# H0_res, Om0_res, Ode0_res = popt_res
# H0_best_res_err, Om0_best_res_err = np.sqrt(np.diag(pcov_res)) # std dev of parameters
# print(f'Astropycosmo fit res: H0 = {H0_best_res} ± {H0_best_res_err}\n\t\t\tOm0 = {Om0_best_res} ± {Om0_best_res_err}\n')

# model_computeHo
guess = 70
# bounds = []
popt_cpt, pcov_cpt = curve_fit(model_computeHo, redshifts, dms, sigma=dm_errs, p0=guess)
H0_best_cpt = float(popt_cpt)
H0_best_cpt = float(H0_best_cpt)
H0_best_cpt_err = np.sqrt(np.diag(pcov_cpt)) # std dev of parameters
print(f'Astropycosmo fit onlyHo: H0 = {H0_best_cpt} ± {H0_best_cpt_err}\n\t\t\t Om0 = 0.3 ± 0.0\n')


# define redshift array (z axis)
z=np.linspace(0.001, 0.17, 1000)

# Compute distmods
distmod_manual = cosmo_manual.distmod(z).value
distmod2 = cosmo2.distmod(z).value
distmod_org = cosmo_org.distmod(z).value
distmod3 = cosmo3.distmod(z).value
distmod_bestfit_flat = model_flatlambdacdm(z, H0_best_flat, Om0_best_flat)
distmod_bestfit_lam = model_lambdacdm(z, H0_best_lam, Om0_best_lam, Ode0_best_lam)
# distmod_bestfit_res = model_lambdacdm(z, H0_res, Om0_res, Ode0_res)
distmod_bestfit_cpt = model_computeHo(z, H0_best_cpt)

# plot cosmological models
plt.figure()
plt.scatter(redshifts, dms)
plt.errorbar(redshifts, dms, yerr=[dm_errs], fmt='none')

# plt.plot(z, distmod2, label='distmod2', color=(187/255,58/255,50/255))
# plt.plot(z, distmod3, label='distmod3', color=(81/255,197/255,58/255))
plt.plot(z, distmod_org, label='original', color='gray', alpha=0.8)
plt.plot(z, distmod_bestfit_cpt, label=f'Best Fit (only $H_0$): $H_0$={H0_best_cpt:.5f} ($\Omega_m$=0.3)')
# plt.plot(z, distmod_manual, label='distmod', color=(239/255, 134/255, 54/255))
plt.plot(z, distmod_bestfit_flat, label=f'Best Fit: $H_0$={H0_best_flat:.5f}, $\Omega_m$={Om0_best_flat:.5f}', color='red', linewidth=1)
plt.plot(z, distmod_bestfit_lam, label=f'Best Fit: $H_0$={H0_best_lam:.5f}, $\Omega_m$={Om0_best_lam:.5f}, $\Omega_\Lambda$={Ode0_best_lam:.5f}', color='orange', linewidth=1, linestyle='dashed')
# plt.plot(z, distmod_bestfit_res, label=f'Best Fit: $H_0$={H0_res:.5f}, $\Omega_m$={Om0_res:.5f}, $\Omega_\Lambda$={Ode0_res:.5f}', color='green', linewidth=1, linestyle='dashed')

plt.title('Cosmological models')
plt.legend()
plt.xlabel('$z$')
plt.ylabel('$\mu$')
plt.savefig('actual_fig.png', dpi=300)

# plot difference between models
# diff = distmod_bestfit_flat -
# distmod_bestfit_lam # change your preferences
# plt.figure()
# plt.plot(z, diff)
# plt.savefig('diff.png')






##################################################################################################################################################################
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
# otras cositas
cosmo_best = LambdaCDM(H0=H0_best_lam, Om0=Om0_best_lam, Ode0=Ode0_best_lam, Tcmb0=2.7)
z_evo = np.linspace(0.001, 1.0, num=100)
h_z = [cosmo_best.H(red).value for red in z_evo]
Om_z = [cosmo_best.Om(red) for red in z_evo]
Ode_z = [cosmo_best.Ode(red) for red in z_evo]
Ogamma_z = [cosmo_best.Ogamma(red) for red in z_evo]
Tcmb_z = [cosmo_best.Tcmb(red).value for red in z_evo]
random_fig, axs = plt.subplots(3, 2)
axs[0,0].plot(z_evo, h_z, label='$H(z)$')
axs[0,1].plot(z_evo, Om_z, label='$\Omega_m$')
axs[1,0].plot(z_evo, Ode_z, label='$\Omega_\Lambda$')
axs[1,1].plot(z_evo, Ogamma_z, label='$\Omega_\gamma$')
axs[2,1].plot(z_evo, Tcmb_z, label='T CMB')
axs[0,0].legend()
axs[0,1].legend()
axs[1,0].legend()
axs[1,1].legend()
axs[2,1].legend()
# axs[2,2].plot(z_evo, h_z, label='$H(z)$')
# axs[2,2].plot(z_evo, Om_z, label='$\Omega_m$')
# axs[2,2].plot(z_evo, Ode_z, label='$\Omega_\Lambda$')
# axs[2,2].plot(z_evo, Ogamma_z, label='$\Omega_\gamma$')
# axs[2,2].plot(z_evo, Tcmb_z, label='T CMB')
plt.tight_layout()
plt.legend()
random_fig.savefig('max_randomness.png',dpi=300)
###################################################################################################################################################################################################################################################
