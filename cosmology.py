import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Constants
H0 = 72.0
Omega_m = 0.3
Omega_lambda = 0.7
c = 299792.458  # km/s

class Cosmology:
    # Unit 1
    def __init__(self, H0, Omega_m, Omega_lambda):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.Omega_k = 1.0 - self.Omega_m - self.Omega_lambda

    #Compute the integrand of the distance formula
    def distance_integrand(self, z):
        """Calculate the integrand for the comoving distance."""
        return 1.0 / np.sqrt(
            self.Omega_m * (1 + z)**3 + 
            self.Omega_k * (1 + z)**2 + 
            self.Omega_lambda
            )
 
    def whether_flat(self, tol=1e-6):
        """Return True if |Omega_k| < tolerance (Universe is approximately flat)."""
        return abs(self.Omega_k) < tol
    
    def set_Omega_m(self, new_Omega_m):
        """Set Omega_m and adjust Omega_lambda to keep curvature constant."""
        self.Omega_m = new_Omega_m
        self.Omega_lambda = 1.0 - self.Omega_k - self.Omega_m

    def set_Omega_lambda(self, new_Omega_lambda):
        """Set Omega_lambda and adjust Omega_m to keep curvature constant."""
        self.Omega_lambda = new_Omega_lambda
        self.Omega_m = 1.0 - self.Omega_k - self.Omega_lambda
    
    def Omega_m_h2(self):
        '''calculate the physical matter density parameter'''
        h = self.H0 / 100.0 #km/s/Mpc
        return self.Omega_m * h**2
    
    def __str__(self):
        return f"Cosmology with H0={self.H0}, Omega_m={self.Omega_m}, Omega_lambda={self.Omega_lambda}, Omega_k={self.Omega_k}."
    #"__str__" is for returning a string representation of the object.

    def plot_distance_integrand(self, z_max, num_points):
        """Plot the distance integrand as a function of redshift."""
        z = np.linspace(0, z_max, num_points)
        integrand = self.distance_integrand(z)

        plt.plot(z, integrand, label='Base Model')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Integrand')
        plt.title('Distance Integrand vs Redshift')
        plt.legend()
        plt.show()

    def plot_distance_integrand_with_varying_Omega_m(self, z_max, num_points, Omega_m_values=[0.2, 0.3, 0.4]):
        """Plot the distance integrand for varying Omega_m values."""
        
        for Omega_m in Omega_m_values:
            #Creating new cosmology models and change Omega_m directly
            model = Cosmology(self.H0, Omega_m, self.Omega_lambda)
            #Plot the shape of the integrand curve by varying Omega_m while fixing Omega_lambda
            z = np.linspace(0, z_max, num_points)
            integrand = model.distance_integrand(z)
            plt.plot(
                z, integrand,
                label=f'Omega_m={Omega_m:.2f}, Omega_lambda={model.Omega_lambda:.2f}, Omega_k={model.Omega_k:.2f}'
            )

        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Integrand')
        plt.title('Distance Integrand vs Redshift for Varying Omega_m without fixing curvature')
        plt.legend()
        plt.show()
        
        for Omega_m in Omega_m_values:
            #Creating new cosmology models and change Omega_m by setter method
            model = Cosmology(self.H0, self.Omega_m, self.Omega_lambda)
            model.set_Omega_m(Omega_m)
            #Plot the shape of the integrand curve by varying Omega_m while adjusting Omega_lambda to keep curvature constant
            integrand = model.distance_integrand(z)
            plt.plot(
                z, integrand,
                label=f'Omega_m={Omega_m:.2f}, Omega_lambda={model.Omega_lambda:.2f}, Omega_k={model.Omega_k:.2f}'
            )
            
        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Integrand')
        plt.title('Distance Integrand vs Redshift for Varying Omega_m by fixing curvature')
        plt.legend()
        plt.show()
    

    # Unit 3
    # 4.1.1 Rectangle Rule
    def distance_integral_rectangle(self, z_max, num_points, c):       
        z = np.linspace(0, z_max, num_points)
        dz = z[1] - z[0]
        f = self.distance_integrand(z[:-1])
        integral = np.sum(f) * dz
        return (c / self.H0) * integral  # in Mpc

   # 4.1.2 Trapezoid Rule
    def distance_integral_trapezoid(self, z_max, num_points, c):
        z = np.linspace(0, z_max, num_points)
        dz = z[1] - z[0]
        f = self.distance_integrand(z)
        integral = (dz / 2.0) * (f[0] + 2.0 * np.sum(f[1:-1]) + f[-1])
        return (c / self.H0) * integral

    # 4.1.3 Simpson's Rule
    def distance_integral_simpson(self, z_max, num_points, c):
        """Composite Simpson's rule: num_points must be odd (even number of intervals).
        If an even number of points is passed we increment to the next odd value."""

        if num_points % 2 == 0:
            num_points += 1

        z = np.linspace(0, z_max, num_points)
        dz = z[1] - z[0]
        f = self.distance_integrand(z)
        integral = (dz / 3.0) * (
            f[0]
            + 4.0 * np.sum(f[1:-1:2])
            + 2.0 * np.sum(f[2:-2:2])
            + f[-1]
        )
        return (c / self.H0) * integral
    
    # 4.1.4 Test Numerical Integration Methods
    def test_numerical_integration_methods(self, z_max, n, c):
        D_rect = self.distance_integral_rectangle(z_max, n, c)
        D_trap = self.distance_integral_trapezoid(z_max, n, c)
        D_simp = self.distance_integral_simpson(z_max, n, c)

        print(f"Rectangle rule: {D_rect:.5f} Mpc")
        print(f"Trapezoid rule: {D_trap:.5f} Mpc")
        print(f"Simpson's rule: {D_simp:.5f} Mpc")
    
    # 4.2 Convergence Test
    def convergence_test(self, c, z_max=1.0):
            
        D_true = self.distance_integral_simpson(z_max, num_points=1000000, c=c)
        n_values = np.logspace(1, 5, 50, dtype=int)

        rect_errors = []
        trap_errors = []
        simp_errors = []

        for n in n_values:
            D_rect = self.distance_integral_rectangle(z_max, n, c)
            D_trap = self.distance_integral_trapezoid(z_max, n, c)
            D_simp = self.distance_integral_simpson(z_max, n, c)

            rect_errors.append(abs(D_rect - D_true) / D_true)
            trap_errors.append(abs(D_trap - D_true) / D_true)
            simp_errors.append(abs(D_simp - D_true) / D_true)

        plt.figure(figsize=(8,6))
        plt.loglog(n_values, rect_errors, 'o-', label='Rectangle rule')
        plt.loglog(n_values, trap_errors, 's-', label='Trapezoid rule')
        plt.loglog(n_values, simp_errors, '^-', label="Simpson's rule")

        plt.xlabel('Number of steps n')
        plt.ylabel('Absolute fractional error')
        plt.title('Convergence of Numerical Integration Methods')
        plt.legend()
        plt.savefig("convergence_plot.png", dpi=300)
        plt.show()

        print("Final Rectangle result (n=10000):", self.distance_integral_rectangle(z_max, 10000, c))
        print("Final Trapezoid result (n=10000):", self.distance_integral_trapezoid(z_max, 10000, c))
        print("Final Simpson result (n=10000):", self.distance_integral_simpson(z_max, 10000, c))

    # 4.3.1 Cumulative Trapezoid Rule
    def distance_cumulative_trapezoid(self, z_max, num_points, c):
        """
        Compute comoving distance array D(z) from z=0 to z_max
        using the cumulative trapezoid rule.
        """
        z_values = np.linspace(0, z_max, num_points + 1)
        dz = z_values[1] - z_values[0]

        f = self.distance_integrand(z_values)

        D_integral = np.zeros_like(z_values)
 
        for i in range(1, len(z_values)):
            D_integral[i] = D_integral[i - 1] + 0.5 * dz * (f[i] + f[i - 1])

        D_integral *= (c / self.H0)
        return z_values, D_integral

    # 4.3.2 Test cumulative trapezoid rule
    def test_cumulative_trapezoid(self, z_max, n, c):
        """Compute and plot D(z) using the cumulative trapezoid rule."""
        z_values, D_values = self.distance_cumulative_trapezoid(z_max, n, c)

        plt.figure(figsize=(8,6))
        plt.plot(z_values, D_values, label='Cumulative Trapezoid D(z)')
        plt.xlabel('Redshift z')
        plt.ylabel('Comoving Distance D(z) [Mpc]')
        plt.title('Comoving Distance vs Redshift')
        plt.legend()
        plt.savefig("cumulative_trapezoid_Dz.png", dpi=300)
        plt.show()

        print(f"D(z={z_max}) = {D_values[-1]:.2f} Mpc")

    # 4.3.3 Interpolated Distance and Distance Modulus
    def interpolated_distance(self, z_array, n, c):
        """
        Given arbitrary z_array (any order), return array of comoving distances D(z) [Mpc].
        Uses cumulative trapezoid integration + interpolation.
        """
        z_array = np.array(z_array)
        z_max = np.max(z_array)
        z_grid, D_grid = self.distance_cumulative_trapezoid(z_max, n, c)
  
        interp_func = interp1d(z_grid, D_grid, kind='cubic', bounds_error=False, fill_value="extrapolate")
 
        return interp_func(z_array)
    
    def distance_modulus(self, z_array, n, c):
        """
        Compute distance modulus μ(z) directly from D(z) including curvature.
        """
        z_array = np.array(z_array)
        D_z = self.interpolated_distance(z_array, n, c)

        ok = self.Omega_k

        if abs(ok) < 1e-8:
            # Flat universe
            D_L = (1 + z_array) * D_z
        else:
            sqrt_ok = np.sqrt(abs(ok))
            x = sqrt_ok * self.H0 * D_z / c
            if ok > 0:
                Sx = np.sinh(x)
            else:
                Sx = np.sin(x)
            D_L = (1 + z_array) * (c / self.H0) / sqrt_ok * Sx

        eps = 1e-12
        D_L = np.maximum(D_L, eps)
        mu = 5 * np.log10(D_L) + 25
        return mu

    # 4.3.4 Test interpolation and distance modulus
    def test_interpolation_and_distance_modulus(self, n, c):
        z_test = [0.001, 0.1, 0.3, 0.5, 0.7, 1.0]
        D_interp = self.interpolated_distance(z_test, n=1000, c=c)
        mu_interp = self.distance_modulus(z_test, n=1000, c=c)

        print("\nInterpolated distances:")
        for z, D in zip(z_test, D_interp):
            print(f"z={z:.2f}, D(z)={D:.2f} Mpc")

        print("\nDistance moduli:")
        for z, mu in zip(z_test, mu_interp):
            print(f"z={z:.2f}, μ(z)={mu:.2f}")

        # Plot μ(z)
        z_fine = np.linspace(0.001, 1.0, 200)
        mu_fine = self.distance_modulus(z_fine, n=1000, c=c)
        plt.figure(figsize=(8,6))
        plt.plot(z_fine, mu_fine, label="Distance Modulus μ(z)")
        plt.xlabel("Redshift z")
        plt.ylabel("Distance Modulus μ(z)")
        plt.title("Interpolated Distance Modulus vs Redshift")
        plt.legend()
        plt.savefig("distance_modulus_vs_redshift.png", dpi=300)
        plt.show()
    
    # 4.4 Exploring the effects of changing the three main parametres
    def changing_parametres(self, n, c):
        """
        Explore how varying H0, Omega_m, and Omega_lambda affects the distance modulus μ(z).
        Output as three subplots arranged horizontally in one combined figure.
        """
        z = np.linspace(0.001, 1.0, 1000)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Changing Omega_m
        for Om in [0.2, 0.3, 0.4]:
            model = Cosmology(self.H0, Om, 1 - Om)  # assume flat universe
            mu = model.distance_modulus(z, n, c)
            axes[0].plot(z, mu, label=f"Omega_m={Om:.2f}")
        axes[0].set_xlabel("Redshift z")
        axes[0].set_title("Different Omega_m")
        axes[0].legend()

        # Changing Omega_lambda
        for Ol in [0.6, 0.7, 0.8]:
            model = Cosmology(self.H0, 1 - Ol, Ol)
            mu = model.distance_modulus(z, n, c)
            axes[1].plot(z, mu, label=f"Omega_lambda={Ol:.2f}")
        axes[1].set_xlabel("Redshift z")
        axes[1].set_title("Different Omega_lambda")
        axes[1].legend()

        # Changing H0
        for H in [65.0, 72.0, 80.0]:
            model = Cosmology(H, self.Omega_m, self.Omega_lambda)
            mu = model.distance_modulus(z, n, c)
            axes[2].plot(z, mu, label=f"H0={H:.0f} km/s/Mpc")
        axes[2].set_xlabel("Redshift z")
        axes[2].set_title("Different H0")
        axes[2].legend()

        fig.text(0.03, 0.5, 'Distance Modulus μ(z)', rotation='vertical', va='center', ha='left')
        fig.suptitle("Exploration of Parameter Effects on Distance Modulus μ(z)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        fig.savefig("parameter_effects_3in1.png", dpi=300)
        plt.show()

def main():
    n = 1000  # Number of points/intervals
    z_max = 1.0
    
    base_model = Cosmology(H0, Omega_m, Omega_lambda)
    print(base_model)
    
    base_model.plot_distance_integrand(z_max, n)
    base_model.plot_distance_integrand_with_varying_Omega_m(z_max, n)
    
    base_model.test_numerical_integration_methods(z_max, n, c)
    base_model.convergence_test(c)
    base_model.test_cumulative_trapezoid(z_max, n, c)
    base_model.test_interpolation_and_distance_modulus(n, c)
    base_model.changing_parametres(n, c)

if __name__ == "__main__":
    main()
