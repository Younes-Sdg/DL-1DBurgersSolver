import numpy as np

def initial_condition(x):
    """Defines the initial condition based on position x."""
    if -10 <= x < -6:
        return 0.2
    elif -6 <= x < -1:
        return 1
    return 0.4

def flux(u):
    """Computes the flux for the Burgers' equation."""
    return (u ** 2) / 2

def flux_derivative(u):
    """Derivative of the flux function for numerical flux calculations."""
    return u

def numerical_flux(u, v, method, lambda_value):
    """Generalized function to calculate numerical flux based on the method."""
    if method == 'lax_Friedrichs':
        return 0.5 * (flux(u) + flux(v) - (v - u) / (2 * lambda_value))
    elif method == 'lax_Wendroff':
        return 0.5 * (flux(u) + flux(v) - 0.5 * (u + v) * lambda_value * (flux(v) - flux(u)))
    elif method == 'Murman_Roe':
        vectorized_flux = np.vectorize(flux)
        flux_difference = np.where(u != v, (vectorized_flux(v) - vectorized_flux(u)) / (v - u), flux_derivative(u))
        return 0.5 * (vectorized_flux(u) + vectorized_flux(v) + (u - v) * np.abs(flux_difference))
    elif method == 'Engquist_Osher':
        integral_part = np.trapz(np.abs(flux_derivative(np.linspace(u, v, num=100))), np.linspace(u, v, num=100))
        return 0.5 * (flux(u) + flux(v) - integral_part)
    else:
        raise ValueError("Invalid numerical flux method")

def initialize_data(a, b, Nx):
    """Initializes the spatial grid and initial condition."""
    x = np.linspace(a, b, Nx + 1)
    Uo = np.array([initial_condition(xi) for xi in x])
    dx = (b - a) / Nx
    return x, Uo, dx

def apply_Dirichlet(Uo_new, left_value=0.2, right_value=0.4):
    """Apply Dirichlet boundary conditions explicitly."""
    Uo_new[0] = left_value
    Uo_new[-1] = right_value
    return Uo_new

def update_solution(Uo, dx, cfl, method, time, T,left_dirichlet=0.2,right_dirichlet=0.4):
    """Update the solution array using the specified numerical method and apply Dirichlet boundary conditions."""
    dt = cfl * dx / max(abs(Uo))
    dt = min(dt, T - time)  # Ensure we do not go beyond the final time
    lambda_value = dt / dx

    # Apply the chosen numerical flux method
    Uo_new = Uo.copy()
    for i in range(1, len(Uo) - 1):
        Uo_new[i] = Uo[i] - lambda_value * (numerical_flux(Uo[i], Uo[i + 1], method, lambda_value) - numerical_flux(Uo[i - 1], Uo[i], method, lambda_value))

    # Apply Dirichlet boundary conditions
    Uo_new = apply_Dirichlet(Uo_new,left_dirichlet,right_dirichlet)

    return Uo_new, dt, time + dt