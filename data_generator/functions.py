import numpy as np
def initial_condition(x):
    
    """
    Defines the initial condition of the solution based on the position x.

    Parameters:
        x (float): The spatial coordinate.

    Returns:
        float: The initial value of the solution at position x.
    """
    
    if  (x >= -10) and (x < -6):
        value = -0.2
    elif (x >= -6) and (x < -1):
        value = 1
    else:
        value = 0.4
    return value

def flux(u):
    
    """
    Computes the flux for the given value of u.

    Parameters:
        u (float): Value of the solution.

    Returns:
        float: Computed flux value.
    """
    
    return (u**2) / 2

def flux_derivative(u):
    
    """
    Computes the derivative of the flux function.

    Parameters:
        u (float): Value of the solution.

    Returns:
        float: Derivative of the flux at u.
    """
    
    return u

def lax_friedrichs_flux(u, v, lambda_value):
    
    """
    Computes the numerical flux using the Lax-Friedrichs method.

    Parameters:
        u (float): Value of the solution at the current grid point.
        v (float): Value of the solution at the adjacent grid point.
        lambda_value (float): Lambda value derived from CFL condition.

    Returns:
        float: Numerical flux computed between two points.
    """
    
    return 0.5 * (flux(u) + flux(v) - 0.5*(1 / lambda_value) * (v - u))

def lax_wendroff_flux(u, v, lambda_value):
    
    """
    Computes the numerical flux using the Lax-Wendroff method.

    Parameters:
        u (float): Value of the solution at the current grid point.
        v (float): Value of the solution at the adjacent grid point.
        lambda_value (float): Lambda value derived from CFL condition.

    Returns:
        float: Numerical flux computed between two points.
    """
    
    return 0.5 * (flux(u) + flux(v) - 0.5 * (u + v) * lambda_value * (flux(v) - flux(u)))

def murman_roe_flux(u, v):
    
    """
    Computes the numerical flux using the Murman-Roe method.

    Parameters:
        u (np.array): Values of the solution at the current grid points.
        v (np.array): Values of the solution at the adjacent grid points.

    Returns:
        np.array: Numerical flux computed between arrays of points.
    """
    
    u = np.array(u)
    v = np.array(v)
    vectorized_flux = np.vectorize(flux)
    flux_difference = np.where(u != v, (vectorized_flux(v) - vectorized_flux(u)) / (u - v), flux_derivative(u))
    return 0.5 * (vectorized_flux(u) + vectorized_flux(v) + (u - v) * np.abs(flux_difference))

def engquist_osher_flux(u, v):
    """
    Computes the Engquist-Osher numerical flux.

    Parameters:
        u (np.array or float): Value of the solution at the current grid point.
        v (np.array or float): Value of the solution at the adjacent grid point.

    Returns:
        float or np.array: Engquist-Osher numerical flux between u and v.
    """
    # Calculate the flux for scalar inputs or arrays
    u, v = np.asarray(u), np.asarray(v)
    
    # Check if u and v are arrays and calculate the integral if they are not equal
    if np.array_equal(u, v):
        integral_part = 0
    else:
        # Use numerical integration to compute the integral of |a(ξ)| from u to v
        integral_part = np.zeros_like(u)
        for idx in range(len(u)):
            u_val, v_val = u[idx], v[idx]
            integral_part[idx] = np.trapz(np.abs(flux_derivative(np.linspace(u_val, v_val, num=100))), np.linspace(u_val, v_val, num=100))
    
    return 0.5 * (flux(u) + flux(v) - integral_part)

def central_flux(u, v):
    
    """
    Computes the central difference flux approximation.

    Parameters:
    u (float): Value of the solution at the current grid point.
    v (float): Value of the solution at the adjacent grid point.

    Returns:
    float: Average of flux at two points.
    """
    
    return 0.5 * (flux(u) + flux(v))

def forward_flux(u, v):
    """
    Computes the forward difference flux approximation.

    Parameters:
        u (float): Value of the solution at the current grid point.
        v (float): Value of the solution at the adjacent grid point.

    Returns:
        float: Flux at the adjacent point.
    """
    return flux(v)

def backward_flux(u, v):
    """
    Computes the backward difference flux approximation.

    Parameters:
        u (float): Value of the solution at the current grid point.
        v (float): Value of the solution at the adjacent grid point.

    Returns:
        float: Flux at the current point.
    """
    return flux(u)

def numerical_method(choice, u, v, lambda_value):
    """
    Selects the numerical method to compute flux based on the given choice.

    Parameters:
        choice (str): Name of the method to use ('lax_Friedrichs', 'lax_Wendroff', 'Murman_Roe').
        u (float): Value of the solution at the current grid point.
        v (float): Value of the solution at the adjacent grid point.
        lambda_value (float): Lambda value derived from CFL condition.

    Returns:
        float: Result of the numerical flux function.
    """
    if choice == "lax_Friedrichs":
        return lax_friedrichs_flux(u, v, lambda_value)
    elif choice == "lax_Wendroff":
        return lax_wendroff_flux(u, v, lambda_value)
    elif choice == "Murman_Roe":
        return murman_roe_flux(u, v)
    elif choice == "Engquist_Oshe":
        return engquist_osher_flux (u,v)
    else:
        raise ValueError(f"Invalid method choice '{choice}'. Valid options are 'lax_Friedrichs', 'lax_Wendroff', 'Murman_Roe', 'Engquist_Oshe'.")

def initialize_data(a, b, Nx):
    """
    Initializes the spatial grid and the initial condition.

    Parameters:
        a (float): Start of the spatial domain.
        b (float): End of the spatial domain.
        Nx (int): Number of grid points.

    Returns:
        tuple: Tuple containing the grid points x (np.array), initial condition values Uo (np.array), and grid spacing dx (float).
    """
    dx = abs((b-a)/Nx)
    x = np.linspace(a, b, Nx + 1)
    Uo = np.zeros(len(x))
    for i in range(len(x)):
        Uo[i] = initial_condition(x[i])
    return x, Uo, dx

def compute_dt_lambda(cfl, dx, Uo, time):
    """
    Computes the time step and lambda value based on the CFL condition and the maximum value of Uo.

    Parameters:
        cfl (float): CFL number.
        dx (float): Spatial grid spacing.
        Uo (np.array): Current solution array.
        time (float): Current simulation time.

    Returns:
        tuple: Tuple containing the time step dt (float) and lambda value (float).
    """
    dt = cfl * dx / max(abs(Uo))
    dt = min(dt, 5 - time)
    lambda_value = dt / dx
    return dt, lambda_value


def update_solution(Uo, lambda_value, method_choice):
    """
    Update the solution array for the next time step using the specified numerical method.

    Parameters:
        Uo (np.array): The current solution array.
        lambda_value (float): The CFL condition scaled lambda value.
        method_choice (str): The numerical method to be used for flux calculation.

    Returns:
        np.array: The updated solution array.
    """
    # Compute the numerical fluxes using the chosen method
    g1 = numerical_method(method_choice, Uo[1:-1], Uo[2:], lambda_value)
    g2 = numerical_method(method_choice, Uo[:-2], Uo[1:-1], lambda_value)
    
    # Copy the current solution to prepare for updating
    Un = Uo.copy()
    
    # Update the internal points using the flux differences
    Un[1:-1] = Uo[1:-1] - lambda_value * (g1 - g2)
    
    # Apply boundary conditions
    Un[0] = Un[1]  # Using the second point as the boundary condition
    Un[-1] = Un[-2]  # Using the penultimate point as the boundary condition
    
    return Un