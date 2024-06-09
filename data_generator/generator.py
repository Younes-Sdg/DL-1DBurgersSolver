import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import functions as f

# Parameters and space-time discretization
a, b, T0, T, Nx, cfl = -10, 10, 0, 5, 100, 0.5
x, Uo, dx = f.initialize_data(a, b, Nx)
time = T0

# Initialize data storage
data = []

# Set up the figure and axis for animation
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, Uo, 'r-')
ax.set_xlim(a, b)
ax.set_ylim(-0.3, 1.1)
ax.set_title("Equation de Burgers")
ax.set_xlabel('X')
ax.set_ylabel('U')

def animate(frame):
    global Uo, time, data
    Uo, dt, time = f.update_solution(Uo, dx, cfl, "Engquist_Osher", time, T)
    line.set_ydata(Uo)

    # Append current data to the list
    for xi, u in zip(x, Uo):
        data.append([time, xi, u])

    if time >= T:
        ani.event_source.stop()
        # Save the data to a CSV file once the animation is complete
        np.savetxt("simulation_data.csv", np.array(data), delimiter=",", header="time,x,u", comments='')

ani = FuncAnimation(fig, animate, frames=None, interval=100, repeat=False, cache_frame_data=False)
plt.show()
