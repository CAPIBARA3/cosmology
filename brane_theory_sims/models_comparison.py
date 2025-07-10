import numpy as np
import matplotlib.pyplot as plt
from mpmath import linspace
from scipy.integrate import cumulative_trapezoid

# ===========================
# 1. Parameter definition
# ===========================
class CosmologicalParameters:
    """
    Base class to store cosmological parameters with easy access methods.
    """

    def __init__(self, **kwargs):
        # Standard cosmological parameters
        self.H0 = 70.0  # km/s/Mpc
        self.Omega_m0 = 0.3  # Matter density parameter today
        self.Omega_r0 = 0.001  # Radiation density parameter today
        self.Omega_Lambda0 = 1 - self.Omega_m0 - self.Omega_r0  # Dark energy (flat universe)

        # Dark energy equation of state parameters (following DESI 2025)
        self.w0 = -0.7  # Present value of w
        self.wa = -1.0  # Evolution parameter for w(z) = w0 + wa * z/(1+z)

        # RSII brane model parameters
        self.rho_c = 3 * (self.H0 * 3.086e19) ** 2 / (8 * np.pi * 6.674e-11)  # Critical density in SI
        self.lambda_brane = 1e-4  # Brane tension parameter (adjustable)
        self.lambda_5 = 0.0

        # Update with any provided parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # Recalculate dependent parameters
                if key in ['Omega_m0', 'Omega_r0']:
                    self.Omega_Lambda0 = 1 - self.Omega_m0 - self.Omega_r0
                elif key == 'H0':
                    self.rho_c = 3 * (self.H0 * 3.086e19) ** 2 / (8 * np.pi * 6.674e-11)

    def update_parameters(self, **kwargs):
        """Convenient method to update multiple parameters at once"""
        self.__init__(**{**self.__dict__, **kwargs})

    def get_all_parameters(self):
        """Return dictionary of all parameters"""
        return {key: value for key, value in self.__dict__.items()
                if not key.startswith('_')}

# ===========================
# 2. Model definition
# ===========================
class LambdaCDM(CosmologicalParameters):
    """
    Lambda-CDM cosmological model with inherited parameter access.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def H(self, z):
        """
        Hubble parameter for Lambda-CDM model
        H(z) = H0 * sqrt(Omega_m0*(1+z)^3 + Omega_r0*(1+z)^4 + Omega_Lambda0)
        """
        E_squared = (self.Omega_m0 * (1 + z) ** 3 +
                     self.Omega_r0 * (1 + z) ** 4 +
                     self.Omega_Lambda0)
        return self.H0 * np.sqrt(E_squared)

class wowaCDM(CosmologicalParameters):
    """
    w0wa-CDM cosmological model with inherited parameter access.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def H(self, z):
        """
        Hubble parameter for w0wa-CDM model
        Dark energy density: rho_DE(z) = rho_DE0 * (1+z)^(3*(1+w0+wa)) * exp(-3*wa*z/(1+z))
        """

        def integrand(zp):
            return (1 + self.w0 + self.wa * zp / (1 + zp)) / (1 + zp)

        if isinstance(z, np.ndarray):
            integral_values = []
            for z_val in z:
                if z_val == 0:
                    integral_val = 0
                else:
                    n_points = max(100, int(z_val * 50))
                    z_points = np.linspace(0, z_val, n_points)
                    y_values = integrand(z_points)
                    integral_result = cumulative_trapezoid(y_values, z_points, initial=0)
                    integral_val = integral_result[-1]
                integral_values.append(integral_val)
            integral_values = np.array(integral_values)
        else:
            if z == 0:
                integral_values = 0
            else:
                n_points = max(100, int(z * 50))
                z_points = np.linspace(0, z, n_points)
                y_values = integrand(z_points)
                integral_result = cumulative_trapezoid(y_values, z_points, initial=0)
                integral_values = integral_result[-1]

        rho_DE_evolution = np.exp(3 * integral_values)
        E_squared = (self.Omega_m0 * (1 + z) ** 3 +
                     self.Omega_r0 * (1 + z) ** 4 +
                     self.Omega_Lambda0 * rho_DE_evolution)
        return self.H0 * np.sqrt(E_squared)

class RSIIBrane(CosmologicalParameters):
    """
    RSII brane-world cosmological model with inherited parameter access.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def H(self, z):
        """
        Modified Friedmann equation: H^2 = 8πG*rho/3 * (1 + rho/(2*lambda))
        where lambda is the brane tension
        """
        rho_total = (self.rho_c * self.Omega_m0 * (1 + z) ** 3 +
                     self.rho_c * self.Omega_r0 * (1 + z) ** 4 +
                     self.rho_c * self.Omega_Lambda0)

        lambda_SI = self.lambda_brane * self.rho_c
        H_squared = (8 * np.pi * 6.674e-11 * rho_total / 3) * (1 + (rho_total / (2 * lambda_SI)))
        H_brane = np.sqrt(H_squared) / (3.086e19)
        return H_brane

    def plot_comparison(self, z_max=5, n_points=1000):
        """
        Plot comparison of Hubble parameter evolution for all three models
        """
        z = np.linspace(0, z_max, n_points)

        # Calculate Hubble parameters for each model
        H_lcdm = self.lambdaCDM_hubble(z)
        H_wowa = self.wowaCDM_hubble(z)
        H_rsii = self.rsii_brane_hubble(z)

        # Create the plot
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(z, H_lcdm, 'b-', linewidth=2, label='Λ-CDM')
        plt.plot(z, H_wowa, 'r--', linewidth=2, label=f'w₀wₐ-CDM (w₀={self.w0}, wₐ={self.wa})')
        plt.plot(z, H_rsii, 'g:', linewidth=2, label=f'RSII Brane (λ={self.lambda_brane})')

        plt.xlabel('Redshift z')
        plt.ylabel('H(z) [km/s/Mpc]')
        plt.title('Hubble Parameter Evolution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, z_max)

        # Plot relative differences
        plt.subplot(2, 1, 2)
        rel_diff_wowa = (H_wowa - H_lcdm) / H_lcdm * 100
        rel_diff_rsii = (H_rsii - H_lcdm) / H_lcdm * 100

        plt.plot(z, rel_diff_wowa, 'r--', linewidth=2, label='(w₀wₐ-CDM - Λ-CDM)/Λ-CDM × 100%')
        plt.plot(z, rel_diff_rsii, 'g:', linewidth=2, label='(RSII - Λ-CDM)/Λ-CDM × 100%')

        plt.xlabel('Redshift z')
        plt.ylabel('Relative Difference [%]')
        plt.title('Relative Differences from Λ-CDM Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, z_max)

        plt.tight_layout()
        plt.show()

        return z, H_lcdm, H_wowa, H_rsii

    def set_parameters(self, w0=None, wa=None, lambda_brane=None, Omega_m0=None):
        """
        Update model parameters
        """
        if w0 is not None:
            self.w0 = w0
        if wa is not None:
            self.wa = wa
        if lambda_brane is not None:
            self.lambda_brane = lambda_brane
        if Omega_m0 is not None:
            self.Omega_m0 = Omega_m0
            self.Omega_Lambda0 = 1 - self.Omega_m0 - self.Omega_r0

    def print_parameters(self):
        """
        Print current model parameters
        """
        print("Current Cosmological Parameters:")
        print(f"H₀ = {self.H0} km/s/Mpc")
        print(f"Ωₘ₀ = {self.Omega_m0}")
        print(f"Ωᵣ₀ = {self.Omega_r0}")
        print(f"ΩΛ₀ = {self.Omega_Lambda0:.3f}")
        print(f"w₀ = {self.w0}")
        print(f"wₐ = {self.wa}")
        print(f"λ_brane = {self.lambda_brane}")

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # initialize models
    params = CosmologicalParameters
    LCDM = LambdaCDM()
    wwCDM = wowaCDM()
    rsii = RSIIBrane()

    z = np.linspace(.0, 8.0, 100)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(z, LCDM.H(z), 'b-', linewidth=2, label=r'$\Lambda\text{CDM}$')
    ax.plot(z, wwCDM.H(z), 'r--', linewidth=2, label=r'$w_0w_a\text{CDM}$')
    ax.plot(z, rsii.H(z), 'g:', linewidth=2, label='RSII Brane-World')
    plt.yscale('log')

    ax.set_xlabel('Redshift z')
    ax.set_ylabel('H(z) [km/s/Mpc]')
    ax.set_title('Hubble Parameter Evolution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ADD INSET HERE
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Create inset
    axins = inset_axes(ax, width="35%", height="35%", loc='lower right')

    # Plot zoomed region (0 < z < 1)
    z_inset = np.linspace(0.0, 1.0, 50)
    axins.plot(z_inset, LCDM.H(z_inset), 'b-', linewidth=2)
    axins.plot(z_inset, wwCDM.H(z_inset), 'r--', linewidth=2)
    # axins.plot(z_inset, rsii.H(z_inset), 'g:', linewidth=2)

    # Set inset limits
    axins.set_xlim(0, 1)
    # Automatically adjust y-limits or set manually
    axins.set_ylim(min(LCDM.H(z_inset)), max(LCDM.H(z_inset)))

    # Style the inset
    axins.grid(True, alpha=0.3)
    axins.set_xlabel('z', fontsize=9)
    axins.set_ylabel('H(z)', fontsize=9)
    axins.tick_params(labelsize=8)

    # Optional: highlight the zoomed region on main plot
    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.7, linewidth=1)

    plt.show()



# Example usage and demonstration
# if __name__ == "__main__":
    # # Create cosmological models instance
    # cosmo = CosmologicalModels()
#
    # # Print current parameters
    # cosmo.print_parameters()
#
    # # Plot comparison with default parameters
    # print("\n1. Comparison with default parameters:")
    # z_vals, H_lcdm, H_wowa, H_rsii = cosmo.plot_comparison(z_max=3)
#
    # # Example with different dark energy parameters
    # print("\n2. Comparison with different dark energy evolution:")
    # cosmo.set_parameters(w0=-0.9, wa=-0.2)
    # cosmo.plot_comparison(z_max=3)
#
    # # Example with different brane tension
    # print("\n3. Comparison with different brane tension:")
    # cosmo.set_parameters(w0=-1.0, wa=0.0, lambda_brane=1e-3)
    # cosmo.plot_comparison(z_max=3)
#
    # # Calculate specific values at interesting redshifts
    # print("\n4. Hubble parameter values at specific redshifts:")
    # test_redshifts = [0, 0.5, 1.0, 2.0, 3.0]
#
    # cosmo.set_parameters(w0=-1.0, wa=0.0, lambda_brane=1e-4)  # Reset to defaults
#
    # print("z\tΛ-CDM\t\tw₀wₐ-CDM\tRSII Brane")
    # print("-" * 50)
    # for z in test_redshifts:
        # H_lcdm_val = cosmo.lambdaCDM_hubble(z)
        # H_wowa_val = cosmo.wowaCDM_hubble(z)
        # H_rsii_val = cosmo.rsii_brane_hubble(z)
        # print(f"{z}\t{H_lcdm_val:.1f}\t\t{H_wowa_val:.1f}\t\t{H_rsii_val:.1f}")