import numpy as np
import matplotlib.pyplot as plt

class Cosmology:
    def __init__(self, H0, Omega_m, Omega_lambda):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.Omega_k = 1.0 - self.Omega_m - self.Omega_lambda

    #"__init__" is a magic method in Python,  
    # which is called to initialize the parameters of a newly created object.

    #"self" refers to the object itself.

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

    
def main():
    H0 = 72.0
    Omega_m = 0.3
    Omega_lambda = 0.72
    base_model = Cosmology(H0, Omega_m, Omega_lambda)

    num_points = 1000
    z_max = 1.0

    def plot_distance_integrand(base_model, z_max, num_points):
        """Plot the distance integrand as a function of redshift."""
        z = np.linspace(0, z_max, num_points)
        integrand = base_model.distance_integrand(z)

        plt.plot(z, integrand, label='Base Model')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Integrand')
        plt.title('Distance Integrand vs Redshift')
        plt.legend()
        plt.show()

    def plot_distance_integrand_with_varying_Omega_m(base_model, z_max, num_points, Omega_m_values=[0.2, 0.3, 0.4]):
        """Plot the distance integrand for varying Omega_m values."""
        z = np.linspace(0, z_max, num_points)
        
        for Omega_m in Omega_m_values:
            #Creating new cosmology models and change Omega_m directly
            model = Cosmology(base_model.H0, Omega_m, base_model.Omega_lambda)
            #Plot the shape of the integrand curve by varying Omega_m while fixing Omega_lambda
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
            model = Cosmology(base_model.H0, base_model.Omega_m, base_model.Omega_lambda)
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

    plot_distance_integrand(base_model, z_max, num_points)
    plot_distance_integrand_with_varying_Omega_m(base_model, z_max, num_points)
    print(base_model)

# This is a special python idiom that
# allows the code to be run from the command line,
# but if you import this module in another script
# the code below will not be executed.
if __name__ == "__main__":
    main()
