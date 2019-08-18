import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def f(x):
    return x

def fprime(x):
    return 1
  
def g(x):
    return (x-2)**2 + 1

def gprime(x):
    return 2*x - 4
  
def kelley(x0, xmin, xmax, f, fprime, eps, max_iters):
    # Initialize array to hold alpha, beta, and x
    alpha = []
    beta = []
    x = []
    lk = []
    # Append initial x value to list
    x = np.append(x, x0)
    # Initialize uk
    uk = f(x[0])
    # Initialize value of c for linprog
    c = np.array([0,1])
    # Initialize bounds for linprog
    x0_bounds = (xmin, xmax)
    x1_bounds = (None, None)
    # Generate supporting hyperplanes to find minimum
    for k in range(max_iters):
        # Compute beta value
        beta = np.append(beta, fprime(x[k]))
        # Compute alpha value
        alpha = np.append(alpha, f(x[k]) - beta[k] * x[k])
        # ----- Compute eta value -----
        # Initialize A
        A = np.stack((beta, -np.ones(k+1)), axis = 1)
        # Initialize b
        b = -np.copy(alpha)
        # Solve LP to get eta 
        eta = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds))
        # Use solution from linear program to store x_{k+1} and lk
        x = np.append(x, eta.x[0])
        lk.append(eta.fun)
        # Update uk value
        uk = min(uk, f(x[k+1]))
        # Update temp
        t = x[k+1]
        # Check stopping criteria
        if (uk - lk[k] <= eps):
            # Stopping criterion satisfied
            print("Stopping criterion satisfied in %i iterations." %(k+1)) 
            print("Terminating Program and returning minimizer.")
            break
        if k == max_iters - 1:
            print("Maximum number of iterations reached without satisfying stopping criteria.")
            print("Terminating program and returning current value for x_%i." %(k+1))
    # Return solution
    return x, lk
  
# Run kelley program on problem of my choice
x, lk = kelley(1, -1, 1, f, fprime, 1e-6, 50)
print("x = ", x)
print("lk = ", lk)

# Run kelley program for part (b)
x, lk = kelley(-1, -1, 4, g, gprime, 1e-6, 50)
print("x = ", x)
print("lk = ", lk)    

# ---------- Generate Plot -------------
# Set x values
xplt = x[0:len(x)]
lk = np.insert(lk, 0, g(x[0]))

# Set plot size
plt.figure(figsize=(20,10))
# Plot function
t = np.arange(0., 5., 0.1)
plt.plot(t, (t-2)**2 + 1, 'b-')
# Plot iterates and fk values
plt.plot(xplt, lk, 'ro')
# Generate labels for points
labels = [r'$f{0}(x{0})$'.format(i) for i in range(1,len(lk)+1)]
# Place labels on plot
i = 0
for label, a, b in zip(labels, xplt, lk):
    i = i + 1
    plt.annotate(
            label,
            xy=(a, b), xytext=(a - (-1)**i * (20 + 1.4**i), b - (-1)**i * ( 20 + 1.4**i)),
            textcoords='offset points', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.25),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
# Set axes for plot
plt.axis([1.2, 2.8, -0.25, 1.8])
# Save figure to file
plt.savefig('kelley_plot.png')
# Show plot
plt.show()

