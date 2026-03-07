import os
import numpy as np
from cosmology import Cosmology
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Unit 4 Codes
class Likelihood:
    """
    This class loads the Pantheon supernova dataset and computes the log-likelihood for a given set 
    of cosmological parameters (Omega_m, Omega_lambda, H0) fits the data.
    """
    def __init__(self, data_file):
        # obtain absolute path of data file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_file_path = os.path.join(script_dir, data_file)
        
        # load data：z, mu_obs, mu_err
        data = np.loadtxt(abs_file_path)
        self.z = data[:, 0]
        self.mu_obs = data[:, 1]
        self.mu_err = data[:, 2]

    def model_prediction(self, theta, n=1000, model='flat'):
        """
        Calculate model magnitudes. 
        model: 'flat' (standard) or 'no_lambda' (Omega_lambda fixed to 0)
        """
        if model == 'flat':
            # Standard case: theta has 3 parameters
            Omega_m, Omega_lambda, H0 = theta
        elif model == 'no_lambda':
            # No Dark Energy case: theta has 2 parameters, force Ol=0
            Omega_m, H0 = theta
            Omega_lambda = 0.0
        else:
            raise ValueError("Unknown model type")
        
        # compute m(z) = mu(z) + M (M = -19.3)
        cosmo = Cosmology(H0, Omega_m, Omega_lambda)
        dist_mod = cosmo.distance_modulus(self.z, n, c=299792.458) #km/s
        return dist_mod - 19.3

    def __call__(self, theta, n=1000, model='flat'):
        # compute Log-Likelihood
        m_model = self.model_prediction(theta, n, model)
        diff = self.mu_obs - m_model
        chi2 = np.sum((diff / self.mu_err)**2)
        return -0.5 * chi2

def convergence_test():
    """
    Test how the Log-Likelihood converges as N increases.
    Inputs: Pantheon supernova data
    Outputs: Convergence plot
    """
    print("Computing the convergence of Likelihood calculation...")
    
    lik = Likelihood("pantheon_data.txt")
    # Test parameters
    theta_test = [0.3, 0.7, 70.0] 
    
    # Set N=10000 as the high-precision reference
    log_L_true = lik(theta_test, n=10000)
    print(f"High-precision Reference Log-L: {log_L_true:.5f}")
    
    #Set up different N values and test them
    n_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
    log_L_values = []
    for n in n_values:
        val = lik(theta_test, n)
        log_L_values.append(val)
        diff = abs(val - log_L_true)
        print(f"N={n:4d} | Log-L={val:.5f} | Diff={diff:.5f}")
    errors = np.abs(np.array(log_L_values) - log_L_true)

    # Plotting the convergence
    plt.figure(figsize=(8, 6))
    plt.loglog(n_values, errors, 'o-', label='Numerical Error')
    plt.axhline(y=0.01, color='b', linestyle='--', label='Target Accuracy (0.01)')
    plt.axhline(y=0.001, color='g', linestyle='--', label='High Accuracy (0.001)')
    for n, err in zip(n_values, errors):
        label_text = f"N={n}\nErr={err:.1e}" 
        plt.annotate(label_text, xy=(n, err), xytext=(10, 5), textcoords='offset points', fontsize=9, color='purple', alpha=0.8)
    plt.xlabel('Number of integration steps (N)')
    plt.ylabel('Abs Error in Log-Likelihood w.r.t N=10000')
    plt.title('Convergence of Likelihood Calculation')
    plt.legend()
    plt.show()

def optimization():
    """
    Determine the best-fit cosmological parameters by maximizing the Log-Likelihood.
    Inputs: Pantheon supernova data
    Outputs: Best-fit parameters and plots
    """
    lik = Likelihood("pantheon_data.txt")
    n_used = 500 
    
    # Return -Log-Likelihood for minimization
    def objective_func(theta):
        return -1.0 * lik(theta, n=n_used)

    # Initial Guess [Omega_m=0.3, Omega_lambda=0.7, H0=70.0]
    x0 = [0.3, 0.7, 70.0]
    print(f"Initial guess: {x0}")
    
    # Bounds in form ((min_Om, max_Om), (min_Ol, max_Ol), (min_H0, max_H0))to restrict parameters to be positive
    bnds = ((0.0, 1.5), (0.0, 1.5), (50.0, 100.0))

    print("Optimizing...")
    res = minimize(objective_func, x0, bounds=bnds)
    if not res.success:
        print("Optimization failed:", res.message)
        return
    best_params = res.x
    Om_best, Ol_best, H0_best = best_params
    max_log_L = -res.fun

    print("\n--- Optimization Results ---")
    print(f"Best fit Omega_m      : {Om_best:.5f}")
    print(f"Best fit Omega_lambda : {Ol_best:.5f}")
    print(f"Best fit H0           : {H0_best:.5f}")
    print(f"Maximum Log-Likelihood: {max_log_L:.5f}")

    # Plotting data and best-fit model
    plt.figure(figsize=(10, 6))
    plt.errorbar(lik.z, lik.mu_obs, yerr=lik.mu_err, fmt='.', color='gray', alpha=0.5, label='Pantheon Data')
    z_smooth = np.linspace(min(lik.z), max(lik.z), 200)
    cosmo_best = Cosmology(H0_best, Om_best, Ol_best)
    mu_best = cosmo_best.distance_modulus(z_smooth, n=n_used, c=299792.458) - 19.3
    plt.plot(z_smooth, mu_best, 'r-', linewidth=2, label='Best Fit Model')
    plt.xlabel('Redshift z')
    plt.ylabel('Distance Modulus mu')
    plt.title('Supernova Data vs Best Fit Model')
    plt.legend()
    plt.savefig("best_fit_model.png")
    plt.show()

    # Plotting residuals
    m_model = lik.model_prediction(best_params, n=n_used)
    residuals = (lik.mu_obs - m_model) / lik.mu_err
    
    # Calculate statistics of residuals
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    print(f"\nResiduals Mean: {res_mean:.5f}")
    print(f"Residuals Std Dev: {res_std:.5f} (Should be close to 1)")

    plt.figure(figsize=(10, 4))
    plt.plot(lik.z, residuals, '.', color='blue', alpha=0.6)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Redshift z')
    plt.ylabel('Normalized Residuals (Delta mu)/(sigma)')
    plt.title(f'Residuals (Mean={res_mean:.3f}, Std={res_std:.3f})')
    plt.savefig("residuals.png")
    plt.show()

    # Task 4.3: Optimization with Omega_lambda = 0
    def objective_func_no_lambda(theta):
        # We pass model='no_lambda' here
        return -1.0 * lik(theta, n=n_used, model='no_lambda')

    # Initial Guess & Bounds for 2 parameters [Omega_m, H0]
    x0_2 = [0.3, 70.0] 
    bnds_2 = ((0.0, 1.5), (50.0, 100.0))


    print("Optimizing Model 2 (No Dark Energy)...")
    res2 = minimize(objective_func_no_lambda, x0_2, bounds=bnds_2)

    if not res2.success:
        print("Model 2 optimization failed:", res2.message)
    else:
        # Extract results
        Om_best_2, H0_best_2 = res2.x
        max_log_L_2 = -res2.fun
        
        print(f"Best fit Omega_m      : {Om_best_2:.5f}")
        print(f"Best fit Omega_lambda : 0.00000 (Fixed)")
        print(f"Best fit H0           : {H0_best_2:.5f}")
        print(f"Maximum Log-Likelihood: {max_log_L_2:.5f}")

        # Compare the two models
        print("\n--- Model Comparison ---")
        print(f"Log-L (Standard)  : {max_log_L:.5f}")
        print(f"Log-L (No Lambda) : {max_log_L_2:.5f}")
        
        if max_log_L > max_log_L_2:
            print("Conclusion: Standard Model (with Dark Energy) fits BETTER.")
        else:
            print("Conclusion: No Lambda Model fits BETTER.")

# Unit 5 Codes


if __name__ == "__main__":
    
    lik = Likelihood("pantheon_data.txt")
    theta_test = [0.3, 0.7, 70.0]
    print(f"Log-L: {lik(theta_test):.4f}")
    
    convergence_test()
    optimization()

    