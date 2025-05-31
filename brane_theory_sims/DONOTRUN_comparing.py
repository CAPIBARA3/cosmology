"""
Pure Python Implementation for Cosmological Model Comparison
ΛCDM vs w₀w_a CDM vs Brane-World RSII

This implementation uses Python libraries instead of CosmoMC/CAMB Fortran code.
We'll use CAMB Python wrapper, emcee for MCMC, and CLASS for some calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
import emcee
import corner
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import pandas as pd
from getdist import plots, MCSamples
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------
# 1. Setup and Data Loading
# ------------------------------------------------

class CosmologicalData:
    """Class to handle observational data"""

    def __init__(self):
        # Mock Planck 2018 data (in reality, load from actual files)
        # CMB distance priors
        self.theta_star = 1.04102  # acoustic angle
        self.theta_star_err = 0.00030

        self.D_M = 1378.5  # angular diameter distance to last scattering
        self.D_M_err = 2.5

        self.r_s = 147.05  # sound horizon at drag epoch
        self.r_s_err = 0.28

        # BAO measurements (mock data based on BOSS/eBOSS)
        self.z_bao = np.array([0.15, 0.38, 0.51, 0.61])
        self.DV_over_rs = np.array([4.47, 10.25, 13.36, 15.22])
        self.DV_over_rs_err = np.array([0.17, 0.31, 0.57, 0.48])

        # Supernovae data (mock Pantheon-like)
        self.z_sn = np.linspace(0.01, 1.5, 50)
        # Generate mock distance moduli with scatter
        np.random.seed(42)
        self.mu_obs = self.mock_distance_moduli() + np.random.normal(0, 0.15, len(self.z_sn))
        self.mu_err = np.full_like(self.z_sn, 0.15)

    def mock_distance_moduli(self):
        """Generate mock supernova distance moduli for ΛCDM"""
        # Simple ΛCDM calculation for mock data
        Om = 0.3
        h = 0.7
        z = self.z_sn

        # Luminosity distance in ΛCDM
        def E(z, Om):
            return np.sqrt(Om*(1+z)**3 + (1-Om))

        def integrand(z, Om):
            return 1/E(z, Om)

        DL = np.zeros_like(z)
        for i, zi in enumerate(z):
            from scipy.integrate import quad
            integral, _ = quad(integrand, 0, zi, args=(Om,))
            DL[i] = (1+zi) * integral

        # Distance modulus
        mu = 5*np.log10(DL) + 25 + 5*np.log10(h*100/70)
        return mu

# ------------------------------------------------
# 2. Cosmological Models
# ------------------------------------------------

class LCDMModel:
    """Standard ΛCDM model"""

    def __init__(self):
        self.param_names = ['omega_b', 'omega_c', 'H0', 'tau', 'A_s', 'n_s']
        self.param_labels = ['Ω_b h²', 'Ω_c h²', 'H₀', 'τ', 'A_s', 'n_s']

    def get_camb_params(self, theta):
        """Convert parameter vector to CAMB parameters"""
        omega_b, omega_c, H0, tau, A_s, n_s = theta

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=omega_b, omch2=omega_c, tau=tau)
        pars.InitPower.set_params(As=A_s*1e-9, ns=n_s)
        pars.set_for_lmax(2500, lens_potential_accuracy=0)

        return pars

    def get_background_quantities(self, theta, z_array):
        """Calculate background quantities at given redshifts"""
        pars = self.get_camb_params(theta)
        results = camb.get_results(pars)

        # Angular diameter distances
        DA = results.angular_diameter_distance(z_array)
        # Luminosity distances
        DL = results.luminosity_distance(z_array)
        # Hubble parameter
        H_z = results.hubble_parameter(z_array)

        return DA, DL, H_z

class W0WaCDMModel:
    """w₀w_a CDM model with dynamical dark energy"""

    def __init__(self):
        self.param_names = ['omega_b', 'omega_c', 'H0', 'tau', 'A_s', 'n_s', 'w0', 'wa']
        self.param_labels = ['Ω_b h²', 'Ω_c h²', 'H₀', 'τ', 'A_s', 'n_s', 'w₀', 'w_a']

    def get_camb_params(self, theta):
        """Convert parameter vector to CAMB parameters"""
        omega_b, omega_c, H0, tau, A_s, n_s, w0, wa = theta

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=omega_b, omch2=omega_c, tau=tau)
        pars.InitPower.set_params(As=A_s*1e-9, ns=n_s)

        # Set dark energy equation of state
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
        pars.set_for_lmax(2500, lens_potential_accuracy=0)

        return pars

    def get_background_quantities(self, theta, z_array):
        """Calculate background quantities at given redshifts"""
        pars = self.get_camb_params(theta)
        results = camb.get_results(pars)

        DA = results.angular_diameter_distance(z_array)
        DL = results.luminosity_distance(z_array)
        H_z = results.hubble_parameter(z_array)

        return DA, DL, H_z

class RSIIModel:
    """Randall-Sundrum II brane-world model"""

    def __init__(self):
        self.param_names = ['omega_b', 'omega_c', 'H0', 'tau', 'A_s', 'n_s', 'log_rc']
        self.param_labels = ['Ω_b h²', 'Ω_c h²', 'H₀', 'τ', 'A_s', 'n_s', 'log₁₀(r_c)']

    def get_camb_params(self, theta):
        """Convert parameter vector to CAMB parameters (base ΛCDM)"""
        omega_b, omega_c, H0, tau, A_s, n_s, log_rc = theta

        # Start with standard ΛCDM parameters
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=omega_b, omch2=omega_c, tau=tau)
        pars.InitPower.set_params(As=A_s*1e-9, ns=n_s)
        pars.set_for_lmax(2500, lens_potential_accuracy=0)

        return pars

    def rsii_hubble_modification(self, H_standard, rc_mpc):
        """Apply RSII modification to Hubble parameter"""
        # Convert crossover scale to 1/time units
        # rc in Mpc, H in km/s/Mpc
        H_c = 1.0 / rc_mpc  # Crossover Hubble scale

        # RSII modification: H² = H_std² * (1 + H_std²/H_c²)
        H_ratio_sq = (H_standard / (100 * H_c))**2  # Normalize units
        H_modified = H_standard * np.sqrt(1.0 + H_ratio_sq)

        return H_modified

    def get_background_quantities(self, theta, z_array):
        """Calculate background quantities with RSII modifications"""
        omega_b, omega_c, H0, tau, A_s, n_s, log_rc = theta
        rc = 10**log_rc  # Crossover scale in Mpc

        # Get standard ΛCDM results first
        pars = self.get_camb_params(theta)
        results = camb.get_results(pars)

        # Standard quantities
        DA_std = results.angular_diameter_distance(z_array)
        H_std = results.hubble_parameter(z_array)

        # Apply RSII modifications
        H_modified = self.rsii_hubble_modification(H_std, rc)

        # Recalculate distances with modified expansion history
        # This is a simplified approach - full implementation would solve
        # the modified Friedmann equation consistently

        # For simplicity, scale the distances by the ratio of Hubble parameters
        scale_factor = H_std / H_modified
        DA_modified = DA_std * scale_factor
        DL_modified = DA_modified * (1 + z_array)**2

        return DA_modified, DL_modified, H_modified

# ------------------------------------------------
# 3. Likelihood Functions
# ------------------------------------------------

class CosmologicalLikelihood:
    """Combined likelihood for cosmological parameters"""

    def __init__(self, data, model):
        self.data = data
        self.model = model

    def cmb_likelihood(self, theta):
        """CMB distance priors likelihood"""
        try:
            # Get acoustic scale
            pars = self.model.get_camb_params(theta)
            results = camb.get_results(pars)

            # Calculate theta_star (acoustic angle)
            z_star = results.get_derived_params()['zstar']  # Redshift of last scattering
            DA_star = results.angular_diameter_distance(z_star)
            r_s_calc = results.get_derived_params()['rstar']  # Sound horizon at last scattering

            theta_star_calc = r_s_calc / DA_star

            # Chi-squared for CMB
            chi2_cmb = ((theta_star_calc - self.data.theta_star) / self.data.theta_star_err)**2

            return -0.5 * chi2_cmb

        except:
            return -np.inf

    def bao_likelihood(self, theta):
        """BAO likelihood"""
        try:
            DA, DL, H_z = self.model.get_background_quantities(theta, self.data.z_bao)

            # Calculate DV/rs for each redshift
            # DV = [(1+z)²DA²z/H(z)]^(1/3)
            DV = ((1 + self.data.z_bao)**2 * DA**2 * self.data.z_bao / H_z)**(1/3)

            # Get sound horizon at drag epoch
            pars = self.model.get_camb_params(theta)
            results = camb.get_results(pars)
            r_s_drag = results.get_derived_params()['rdrag']

            DV_over_rs_calc = DV / r_s_drag

            # Chi-squared for BAO
            chi2_bao = np.sum(((DV_over_rs_calc - self.data.DV_over_rs) / self.data.DV_over_rs_err)**2)

            return -0.5 * chi2_bao

        except:
            return -np.inf

    def sn_likelihood(self, theta):
        """Supernova likelihood"""
        try:
            DA, DL, H_z = self.model.get_background_quantities(theta, self.data.z_sn)

            # Calculate distance modulus
            mu_calc = 5 * np.log10(DL) + 25

            # Marginalize over absolute magnitude uncertainty
            # Simple approach: minimize chi-squared over absolute magnitude offset
            def chi2_sn(M_offset):
                return np.sum(((mu_calc + M_offset - self.data.mu_obs) / self.data.mu_err)**2)

            result = minimize(chi2_sn, 0.0)
            chi2_min = result.fun

            return -0.5 * chi2_min

        except:
            return -np.inf

    def log_likelihood(self, theta):
        """Combined log-likelihood"""
        # Check parameter bounds first
        if not self.check_bounds(theta):
            return -np.inf

        logL_cmb = self.cmb_likelihood(theta)
        logL_bao = self.bao_likelihood(theta)
        logL_sn = self.sn_likelihood(theta)

        return logL_cmb + logL_bao + logL_sn

    def check_bounds(self, theta):
        """Check if parameters are within reasonable bounds"""
        if len(theta) == 6:  # ΛCDM
            omega_b, omega_c, H0, tau, A_s, n_s = theta
            bounds_check = (0.005 < omega_b < 0.1 and
                          0.01 < omega_c < 0.99 and
                          50 < H0 < 100 and
                          0.01 < tau < 0.8 and
                          0.5 < A_s < 5.0 and
                          0.8 < n_s < 1.2)
        elif len(theta) == 8:  # w0wa CDM
            omega_b, omega_c, H0, tau, A_s, n_s, w0, wa = theta
            bounds_check = (0.005 < omega_b < 0.1 and
                          0.01 < omega_c < 0.99 and
                          50 < H0 < 100 and
                          0.01 < tau < 0.8 and
                          0.5 < A_s < 5.0 and
                          0.8 < n_s < 1.2 and
                          -3.0 < w0 < 1.0 and
                          -3.0 < wa < 3.0)
        elif len(theta) == 7:  # RSII
            omega_b, omega_c, H0, tau, A_s, n_s, log_rc = theta
            bounds_check = (0.005 < omega_b < 0.1 and
                          0.01 < omega_c < 0.99 and
                          50 < H0 < 100 and
                          0.01 < tau < 0.8 and
                          0.5 < A_s < 5.0 and
                          0.8 < n_s < 1.2 and
                          2.0 < log_rc < 8.0)
        else:
            bounds_check = False

        return bounds_check

# ------------------------------------------------
# 4. MCMC Sampling
# ------------------------------------------------

def run_mcmc_sampling(model, data, nwalkers=32, nsteps=5000, burn_in=1000):
    """Run MCMC sampling for a given model"""

    print(f"Running MCMC for {model.__class__.__name__}...")

    # Set up likelihood
    likelihood = CosmologicalLikelihood(data, model)

    # Initial parameter values (approximate Planck 2018 values)
    if len(model.param_names) == 6:  # ΛCDM
        initial = np.array([0.0224, 0.120, 67.5, 0.06, 2.1, 0.965])
        scales = np.array([0.001, 0.005, 2.0, 0.01, 0.1, 0.01])
    elif len(model.param_names) == 8:  # w0wa CDM
        initial = np.array([0.0224, 0.120, 67.5, 0.06, 2.1, 0.965, -1.0, 0.0])
        scales = np.array([0.001, 0.005, 2.0, 0.01, 0.1, 0.01, 0.1, 0.2])
    elif len(model.param_names) == 7:  # RSII
        initial = np.array([0.0224, 0.120, 67.5, 0.06, 2.1, 0.965, 5.0])
        scales = np.array([0.001, 0.005, 2.0, 0.01, 0.1, 0.01, 0.5])

    # Initialize walkers around the initial position
    ndim = len(initial)
    pos = initial + scales * np.random.randn(nwalkers, ndim) * 0.1

    # Set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood.log_likelihood)

    # Run burn-in
    print("Running burn-in...")
    pos, _, _ = sampler.run_mcmc(pos, burn_in, progress=True)
    sampler.reset()

    # Run production
    print("Running production...")
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Get samples
    samples = sampler.get_chain(discard=0, thin=1, flat=True)

    return samples, sampler

# ------------------------------------------------
# 5. Analysis and Comparison
# ------------------------------------------------

def analyze_and_compare_models():
    """Main analysis function"""

    # Load data
    data = CosmologicalData()

    # Initialize models
    lcdm_model = LCDMModel()
    w0wa_model = W0WaCDMModel()
    rsii_model = RSIIModel()

    models = [lcdm_model, w0wa_model, rsii_model]
    model_names = ['ΛCDM', 'w₀w_a CDM', 'RSII']

    # Run MCMC for each model (reduced steps for demo)
    all_samples = {}
    all_samplers = {}

    for model, name in zip(models, model_names):
        samples, sampler = run_mcmc_sampling(model, data, nwalkers=16, nsteps=1000, burn_in=200)
        all_samples[name] = samples
        all_samplers[name] = sampler

        print(f"\n{name} Results:")
        print(f"  Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

        # Print parameter constraints
        for i, (param_name, param_label) in enumerate(zip(model.param_names, model.param_labels)):
            mean_val = np.mean(samples[:, i])
            std_val = np.std(samples[:, i])
            print(f"  {param_label}: {mean_val:.4f} ± {std_val:.4f}")

    # Create corner plots for comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # ΛCDM corner plot
    corner.corner(all_samples['ΛCDM'][:, [0, 1, 2]],
                 labels=['$\\Omega_b h^2$', '$\\Omega_c h^2$', '$H_0$'],
                 ax=axes[0,0])
    axes[0,0].set_title('ΛCDM')

    # w0wa CDM corner plot
    corner.corner(all_samples['w₀w_a CDM'][:, [0, 1, 2]],
                 labels=['$\\Omega_b h^2$', '$\\Omega_c h^2$', '$H_0$'],
                 ax=axes[0,1])
    axes[0,1].set_title('w₀w_a CDM')

    # RSII corner plot
    corner.corner(all_samples['RSII'][:, [0, 1, 2]],
                 labels=['$\\Omega_b h^2$', '$\\Omega_c h^2$', '$H_0$'],
                 ax=axes[1,0])
    axes[1,0].set_title('RSII')

    # Comparison plot for H0
    axes[1,1].hist(all_samples['ΛCDM'][:, 2], bins=30, alpha=0.7, label='ΛCDM', density=True)
    axes[1,1].hist(all_samples['w₀w_a CDM'][:, 2], bins=30, alpha=0.7, label='w₀w_a CDM', density=True)
    axes[1,1].hist(all_samples['RSII'][:, 2], bins=30, alpha=0.7, label='RSII', density=True)
    axes[1,1].set_xlabel('$H_0$ [km/s/Mpc]')
    axes[1,1].set_ylabel('Probability Density')
    axes[1,1].legend()
    axes[1,1].set_title('$H_0$ Comparison')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate evidence (simplified Harmonic mean - not recommended for real analysis)
    print("\nModel Comparison (Simplified Evidence Calculation):")
    print("Note: This is a rough approximation. Use nested sampling for accurate evidence.")

    for name in model_names:
        sampler = all_samplers[name]
        # Get log-likelihood values
        log_prob_samples = sampler.get_log_prob(flat=True)
        # Harmonic mean estimator (biased, but for demonstration)
        log_evidence_approx = -np.log(np.mean(np.exp(-log_prob_samples)))
        print(f"{name}: ln(Evidence) ≈ {log_evidence_approx:.1f}")

    return all_samples, all_samplers

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