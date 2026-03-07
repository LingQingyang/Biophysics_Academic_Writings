import numpy as np
import matplotlib.pyplot as plt

def task_1a(m,n):
    """
    This function takes as input two integer values, m and n, and returns three numpy arrays.
    The first array is of size m and filled with zeros.
    The second array is of size n and contains integers from 1 to n.
    The third array is of size (m,n) and contains random float values between 0 and 1.
    """
    array_1 = np.zeros(m)
    array_2 = np.arange(1,n+1)
    array_3 = np.random.random((m,n))
    return array_1, array_2, array_3

def task_1c(m,n):
    """
    This function calls from task_1a,
    returns the mean of array_2 and the maximum value of array_3.
    """
    a2, a3 = task_1a(m,n)[1], task_1a(m,n)[2]
    mean_a2 = np.mean(a2)
    max_a3 = np.max(a3)
    return mean_a2, max_a3

def task_1d(a):
    """
    This function modifies every elements in array a to be the square of their original values,
    but does not return anything.
    """
    a **= 2

# Difference between mutable and immutable objects in Python.

# Mutable objects can be changed after they are created, such as lists and dictionaries.
# You can add, remove, or change their contents.

# Immutable objects cannot be changed after they are created, such as integers, floats, strings, and tuples.
# Any operation that seems to "change" them actually creates a new object.

def task_2a(n):
    """
    This function takes an integer n and makes a square 2D array M of shape n*n.
    Each elements filled in so that M[i][j] = i+2j where i and j start from zero.
    """
    array_M = np.fromfunction(lambda i, j: i + 2 * j, (n, n), dtype=int)
    return array_M

def task_2b(n):
    """
     This function takes an integer n as input, and computes a 1D array Y,
     where each element is the sum of each column in the 2D array M from task_2a.
     """
    array_Y = np.sum(task_2a(n), axis=0)
    return array_Y

def task_3(d, mu, sigma):
    """
    This function computes a Gaussian log-likelihood from three vectors d, mu and sigma.
    """
    log_likelihood = -0.5 * np.sum(((d - mu) / sigma) ** 2)
    return log_likelihood

def task_4(omegamh2, omegabh2, H0, mode):
    """
    This function does basic data analysis, including
    I. means and standard deviations
    II. histograms of individual data points
    III. scatter plots of pairs of data points
    """  

    def means_and_stds(omegamh2, omegabh2, H0):
        omegamh2_mean, omegamh2_std = np.mean(omegamh2), np.std(omegamh2)
        omegabh2_mean, omegabh2_std = np.mean(omegabh2), np.std(omegabh2)
        H0_mean, H0_std = np.mean(H0), np.std(H0)
    
        # Print results to console
        print(f"omegamh2: mean = {omegamh2_mean}, std = {omegamh2_std}")
        print(f"omegabh2: mean = {omegabh2_mean}, std = {omegabh2_std}")
        print(f"H0: mean = {H0_mean}, std = {H0_std}")

        # Save nicely formatted table to file
        with open(r"C:\Users\DELL\Desktop\summary_table.txt", "w") as f:
            # Please note that the path above is for my own computer!
            f.write("Parameter      Mean       Std\n")
            f.write(f"omegamh2   {omegamh2_mean:.4f}   {omegamh2_std:.4f}\n")
            f.write(f"omegabh2   {omegabh2_mean:.4f}   {omegabh2_std:.4f}\n")
            f.write(f"H0         {H0_mean:.4f}   {H0_std:.4f}\n")

    def histograms(omegamh2, omegabh2, H0):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.hist(omegamh2, bins=20, density=True)
        plt.title('Histogram of omegamh2')
        plt.xlabel('omegamh2')
        plt.ylabel('Frequency density')

        plt.subplot(1, 3, 2)
        plt.hist(omegabh2, bins=20, density=True)
        plt.title('Histogram of omegabh2')
        plt.xlabel('omegabh2')
        plt.ylabel('Frequency density')

        plt.subplot(1, 3, 3)
        plt.hist(H0, bins=20, density=True)
        plt.title('Histogram of H0')
        plt.xlabel('H0')
        plt.ylabel('Frequency density')

        plt.tight_layout()
        plt.savefig("histograms.png")
        plt.show()

    def scatter_plots(omegamh2, omegabh2, H0):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.scatter(omegamh2, omegabh2, alpha=0.5)
        plt.title('Scatter plot of omegamh2 vs omegabh2')
        plt.xlabel('omegamh2')
        plt.ylabel('omegabh2')

        plt.subplot(1, 3, 2)
        plt.scatter(omegamh2, H0, alpha=0.5)
        plt.title('Scatter plot of omegamh2 vs H0')
        plt.xlabel('omegamh2')
        plt.ylabel('H0')

        plt.subplot(1, 3, 3)
        plt.scatter(omegabh2, H0, alpha=0.5)
        plt.title('Scatter plot of omegabh2 vs H0')
        plt.xlabel('omegabh2')
        plt.ylabel('H0')

        plt.tight_layout()
        plt.savefig("scatter_plots.png")
        plt.show()

    if mode == "all":
        means_and_stds(omegamh2, omegabh2, H0)
        histograms(omegamh2, omegabh2, H0)
        scatter_plots(omegamh2, omegabh2, H0)
    
    elif mode == "mean_and_std":
        means_and_stds(omegamh2, omegabh2, H0)
    
    elif mode == "histogram":
        histograms(omegamh2, omegabh2, H0)
    
    elif mode == "scatter":
        scatter_plots(omegamh2, omegabh2, H0)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    m = 6  # n can be any integer
    n = 8  # m can be any integer
    print(task_1a(m,n))
    print(task_1c(m,n))
    a= np.array([1,2,3,4,5])  # a is an arbitrary array for task_1d
    
    print(task_2a(n))
    print(task_2b(n))
    
    d, mu, sigma = np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5]), np.array([0.1, 0.2, 0.3])
    # Defining example arrays for d, mu, and sigma
    print(task_3(d, mu, sigma))

    data = np.loadtxt(r"C:\Users\DELL\Desktop\data.txt", skiprows=1)
    # Please note that the path above is for my own computer!
    omegamh2 = data[:,0]
    omegabh2 = data[:,1]
    H0 = data[:,2]
    
    mode = "all" 
    # mode = "mean_and_std" -- for means and standard deviations only
    # mode = "histogram" -- for histograms only
    # mode = "scatter" -- for scatter plots only
    # mode = "all" -- for all of the above
    task_4(omegamh2, omegabh2, H0, mode)

if __name__ == "__main__":
    main()  