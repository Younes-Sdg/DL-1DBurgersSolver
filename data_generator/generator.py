import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import explicite_modif as data

# Implemented numerical methods:
# -> lax_Friedrichs
# -> lax_Wendroff
# -> Murman_Roe
# -> Engquist_osher

# Parameters and space-time discretization
a, b, T0, T, Nx, cfl = -10, 10, 0, 5, 100, 0.5
x, Uo, dx = data.initialize_data(a, b, Nx)

# Set up the figure and axis for animation
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, Uo, 'r-')  # Initialize a red line representing the initial condition
ax.set_xlim(a, b)
ax.set_ylim(-0.3,1.1)
ax.set_title("Burgers' Equation")
ax.set_xlabel('X')
ax.set_ylabel('U')

time = T0

def animate(n):
    global Uo, time
    dt, lambda_value = data.compute_dt_lambda(cfl, dx, Uo, time)
    time += dt
    Uo = data.update_solution(Uo, lambda_value, "Engquist_Oshe")
    line.set_ydata(Uo)  # Update the plot data with the new solution
    print(time)
    if time >= T:  # Condition to stop the animation when reaching the final time T
        ani.event_source.stop()

# Create and start the animation
ani = FuncAnimation(fig, animate, frames=None, interval=100, repeat=False)  # Set the interval between frames in milliseconds
plt.show()

