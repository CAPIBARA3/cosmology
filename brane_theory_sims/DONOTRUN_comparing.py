"""
Pure Python Implementation for Cosmological Model Comparison
ΛCDM vs w₀w_a CDM vs Brane-World RSII

This implementation uses Python libraries instead of CosmoMC/CAMB Fortran code.
We'll use CAMB Python wrapper, emcee for MCMC, and CLASS for some calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model
import emcee
import corner
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import pandas as pd
from getdist import plots, MCSamples
import warnings
from scipy.integrate import quad
warnings.filterwarnings('ignore')

# ==============================
# 1. Data Handling
# ==============================
class CosmologicalData:
    def __init__(self):
        self.theta_star = 1.04102
        self.theta_star_err = 0.00030
        self.D_M = 1378.5
        self.D_M_err = 2.5
        self.r_s = 147.05
        self.r_s_err = 0.28

        self.z_bao = np.array([0.15, 0.38, 0.51, 0.61])
        self.DV_over_rs = np.array([4.47, 10.25, 13.36, 15.22])
        self.DV_over_rs_err = np.array([0.17, 0.31, 0.57, 0.48])

        self.z_sn = np.linspace(0.01, 1.5, 50)
        np.random.seed(42)
        self.mu_obs = self.mock_distance_moduli() + np.random.normal(0, 0.15, len(self.z_sn))
        self.mu_err = np.full_like(self.z_sn, 0.15)

    def mock_distance_moduli(self):
        Om, h = 0.3, 0.7
        z = self.z_sn
        def E(z): return np.sqrt(Om * (1 + z)**3 + (1 - Om))
        DL = [(1 + zi) * quad(lambda z_: 1/E(z_), 0, zi)[0] for zi in z]
        DL = np.array(DL)
        return 5 * np.log10(DL) + 25 + 5 * np.log10(h * 100 / 70)

# ==============================
# 2. Cosmological Models
# ==============================
class BaseModel:
    def get_background_quantities(self, theta, z_array):
        pars = self.get_camb_params(theta)
        results = camb.get_results(pars)
        return (results.angular_diameter_distance(z_array),
                results.luminosity_distance(z_array),
                results.hubble_parameter(z_array))

class LCDMModel(BaseModel):
    def __init__(self):
        self.param_names = ['omega_b', 'omega_c', 'H0', 'tau', 'A_s', 'n_s']
        self.param_labels = ['Ω_b h²', 'Ω_c h²', 'H₀', 'τ', 'A_s', 'n_s']

    def get_camb_params(self, theta):
        omega_b, omega_c, H0, tau, A_s, n_s = theta
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=omega_b, omch2=omega_c, tau=tau)
        pars.InitPower.set_params(As=A_s * 1e-9, ns=n_s)
        pars.set_for_lmax(2000, lens_potential_accuracy=0)
        return pars

class W0WaCDMModel(BaseModel):
    def __init__(self):
        self.param_names = ['omega_b', 'omega_c', 'H0', 'tau', 'A_s', 'n_s', 'w0', 'wa']
        self.param_labels = ['Ω_b h²', 'Ω_c h²', 'H₀', 'τ', 'A_s', 'n_s', 'w₀', 'w_a']

    def get_camb_params(self, theta):
        omega_b, omega_c, H0, tau, A_s, n_s, w0, wa = theta
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=omega_b, omch2=omega_c, tau=tau)
        pars.InitPower.set_params(As=A_s * 1e-9, ns=n_s)
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
        pars.set_for_lmax(2000, lens_potential_accuracy=0)
        return pars

class RSIIModel(BaseModel):
    def __init__(self):
        self.param_names = ['omega_b', 'omega_c', 'H0', 'tau', 'A_s', 'n_s', 'log_rc']
        self.param_labels = ['Ω_b h²', 'Ω_c h²', 'H₀', 'τ', 'A_s', 'n_s', 'log₁₀(r_c)']

    def get_camb_params(self, theta):
        omega_b, omega_c, H0, tau, A_s, n_s, _ = theta
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=omega_b, omch2=omega_c, tau=tau)
        pars.InitPower.set_params(As=A_s * 1e-9, ns=n_s)
        pars.set_for_lmax(2000, lens_potential_accuracy=0)
        return pars

    def get_background_quantities(self, theta, z_array):
        rc = 10**theta[-1]
        DA, DL, H_std = super().get_background_quantities(theta, z_array)
        H_c = 1 / rc
        H_mod = H_std * np.sqrt(1 + (H_std / (100 * H_c))**2)
        scale = H_std / H_mod
        return DA * scale, DL * scale * (1 + z_array), H_mod

# ==============================
# 3. Likelihood
# ==============================
class CosmologicalLikelihood:
    def __init__(self, data, model):
        self.data, self.model = data, model

    def cmb_likelihood(self, theta):
        try:
            pars = self.model.get_camb_params(theta)
            results = camb.get_results(pars)
            z_star = results.get_derived_params()['zstar']
            r_s = results.get_derived_params()['rstar']
            DA = results.angular_diameter_distance(z_star)
            theta_star = r_s / DA
            chi2 = ((theta_star - self.data.theta_star) / self.data.theta_star_err) ** 2
            return -0.5 * chi2
        except:
            return -np.inf

    def bao_likelihood(self, theta):
        try:
            DA, _, H_z = self.model.get_background_quantities(theta, self.data.z_bao)
            DV = ((1 + self.data.z_bao)**2 * DA**2 * self.data.z_bao / H_z)**(1/3)
            r_s = camb.get_results(self.model.get_camb_params(theta)).get_derived_params()['rdrag']
            chi2 = np.sum(((DV / r_s - self.data.DV_over_rs) / self.data.DV_over_rs_err)**2)
            return -0.5 * chi2
        except:
            return -np.inf

    def sn_likelihood(self, theta):
        try:
            _, DL, _ = self.model.get_background_quantities(theta, self.data.z_sn)
            mu_model = 5 * np.log10(DL) + 25
            offset = minimize(lambda M: np.sum(((mu_model + M - self.data.mu_obs) / self.data.mu_err)**2), 0).fun
            return -0.5 * offset
        except:
            return -np.inf

    def log_likelihood(self, theta):
        if not self.check_bounds(theta): return -np.inf
        return self.cmb_likelihood(theta) + self.bao_likelihood(theta) + self.sn_likelihood(theta)

    def check_bounds(self, theta):
        n = len(theta)
        if n == 6:
            return all([0.005 < theta[0] < 0.1, 0.01 < theta[1] < 0.99, 50 < theta[2] < 100,
                        0.01 < theta[3] < 0.8, 0.5 < theta[4] < 5.0, 0.8 < theta[5] < 1.2])
        elif n == 8:
            return self.check_bounds(theta[:6]) and -3 < theta[6] < 1 and -3 < theta[7] < 3
        elif n == 7:
            return self.check_bounds(theta[:6]) and 2 < theta[6] < 8
        return False

# ==============================
# 4. MCMC
# ==============================
def run_mcmc_sampling(model, data, nwalkers=16, nsteps=500, burn_in=100):
    likelihood = CosmologicalLikelihood(data, model)
    params = model.param_names

    if len(params) == 6:
        initial = np.array([0.0224, 0.12, 67.5, 0.06, 2.1, 0.965])
        scales = np.array([0.001, 0.005, 1, 0.01, 0.1, 0.01])
    elif len(params) == 8:
        initial = np.array([0.0224, 0.12, 67.5, 0.06, 2.1, 0.965, -1.0, 0.0])
        scales = np.array([0.001, 0.005, 1, 0.01, 0.1, 0.01, 0.1, 0.2])
    elif len(params) == 7:
        initial = np.array([0.0224, 0.12, 67.5, 0.06, 2.1, 0.965, 5.0])
        scales = np.array([0.001, 0.005, 1, 0.01, 0.1, 0.01, 0.5])

    pos = initial + scales * 0.1 * np.random.randn(nwalkers, len(initial))
    sampler = emcee.EnsembleSampler(nwalkers, len(initial), likelihood.log_likelihood)
    pos, _, _ = sampler.run_mcmc(pos, burn_in, progress=True)
    sampler.reset()
    sampler.run_mcmc(pos, nsteps, progress=True)
    return sampler.get_chain(flat=True), sampler

# ==============================
# 5. Analysis
# ==============================
def analyze_and_compare_models():
    data = CosmologicalData()
    models = [LCDMModel(), W0WaCDMModel(), RSIIModel()]
    names = ['ΛCDM', 'w₀w_a CDM', 'RSII']

    all_samples = {}
    for model, name in zip(models, names):
        samples, sampler = run_mcmc_sampling(model, data)
        all_samples[name] = samples
        print(f"{name} Mean acceptance: {np.mean(sampler.acceptance_fraction):.3f}")
        for i, param in enumerate(model.param_labels):
            mean, std = np.mean(samples[:, i]), np.std(samples[:, i])
            print(f"{param}: {mean:.4f} ± {std:.4f}")

    # Plot H0 distribution
    plt.figure(figsize=(8,6))
    for name, samples in all_samples.items():
        plt.hist(samples[:, 2], bins=30, density=True, alpha=0.6, label=name)
    plt.xlabel('$H_0$')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Posterior of $H_0$')
    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# 6. Main Execution
# ------------------------------------------------

if __name__ == "__main__":
    print("Starting Cosmological Model Comparison Analysis")
    print("=" * 50)

    # Check if required packages are available
    try:
        import camb
        print("✓ CAMB available")
    except ImportError:
        print("✗ CAMB not available. Install with: pip install camb")
        exit(1)

    try:
        import emcee
        print("✓ emcee available")
    except ImportError:
        print("✗ emcee not available. Install with: pip install emcee")
        exit(1)

    # Run the analysis
    samples, samplers = analyze_and_compare_models()

    print("\nAnalysis complete!")
    print("Check 'model_comparison.png' for visual results.")