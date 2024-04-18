import numpy as np

# Function describing the velocity: v = a(u, x, t) (general form)
def velocity(v, x, t):
    return v(x, t)

# Function describing the initial condition: u = u0(x)
def initial_condition_Creno(x):
    if 0.1 <= x <= 0.2:
        return 10
    else:
        return 0

def initial_condition(x):
    if -10 <= x < -6:
        return 2
    elif -6 <= x < -1:
        return 2
    else:
        return 0

def flux(u,a=None):
    return u



def flux_burgers(u,a=None): #added a for convenience regarding the following code
    return (u ** 2) / 2

def flux_burgers_prime(u):
    return u

# Functions describing the numerical fluxes for each of the schemes
def lax_friedrichs_flux(u, v, labda, flux_function, a=None):
    return (1 / 2) * (flux_function(u, a) + flux_function(v, a) - 0.5 * (1 / labda) * (v - u))

def lax_wendroff_flux(u, v, labda, flux_function, a=None):
    return (1 / 2) * (flux_function(u, a) + flux_function(v, a) - ((u + v) / 2) * (labda) * (flux_function(v, a) - flux_function(u, a)))

def murman_roe_flux(u, v, flux_function, a=None):
    u = np.array(u)
    v = np.array(v)
    flux_vectorized = np.vectorize(lambda u: flux_function(u, a))  # Ensure flux can be vectorized

    flux_diff = np.where(u != v, (flux_vectorized(v) - flux_vectorized(u)) / (u - v), u)
    return 0.5 * (flux_vectorized(u) + flux_vectorized(v) + (u - v) * np.abs(flux_diff))

def center_flux(u, v):
    return (1 / 2) * (flux(u) + flux(v))

def upwind_forward_flux(u, v):
    return flux(v)

def upwind_backward_flux(u, v):
    return flux(u)

# Function to initialize parameters and create spatial grid
def initialize_simulation(a, b, T0, T,Nx, cfl):
    dx = (b-a)/Nx ## dx fixe, d


    x = np.linspace(a, b, Nx + 1)
    U0 = np.array([initial_condition(xi) for xi in x])
    a_max = max(abs(flux_burgers_prime(U0)))
    dt = (dx * cfl)/a_max
    lmbda = dt / dx
    ## maximum de valeur absolue de f'(u) 
    
    return x, U0, dx, dt, lmbda, Nx

def numerical_method(method, u, v, labda,flux_function,a=None):
    if method == "lax_friedrichs":
        return lax_friedrichs_flux(u, v, labda,flux_function,a=None)
    elif method == "lax_wendroff":
        return lax_wendroff_flux(u, v, labda,flux_function,a=None)
    elif method == "murman_roe":
        return murman_roe_flux(u, v,flux_function,a=None)
    else:
        raise ValueError(f"Invalid method choice '{method}'. Valid options are 'lax_friedrichs', 'lax_wendroff', 'murman_roe'.")
