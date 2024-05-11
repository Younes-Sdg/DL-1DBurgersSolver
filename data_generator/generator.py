import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import functions as f  # Make sure this module is correctly imported

# Parameters and discretization of space and time
a, b, T0, T, Nx, cfl = -10, 10, 0, 5, 100, 0.05
x, Uo, dx, dt, labda, Nx = f.initialize_simulation(a, b, T0, T,Nx, cfl)
time = 0

a_func = lambda x: x


# Figure and axis for animation
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, Uo, 'r-')  # Initialize a red line
ax.set_xlim(a, b)
ax.set_ylim(min(Uo), max(Uo) * 1.1)
ax.set_title("Burgers' Equation")
ax.set_xlabel('X')
ax.set_ylabel('U')

def animate(n):
    global Uo, time, ani, T  
    
    # Update Uo for the next time step
    g1 = f.numerical_method("murman_roe", Uo[1:-1], Uo[2:], labda,f.flux,a_func)
    g2 = f.numerical_method("murman_roe", Uo[:-2], Uo[1:-1], labda,f.flux,a_func)
    Un = Uo.copy()
    Un[1:-1] = Uo[1:-1] - labda * (g1 - g2)
    Un[0] = Un[1]  # Handle boundary conditions
    Un[-1] = Un[Nx-1]
    
    Uo = Un
    
    line.set_ydata(Uo)  # Update line data

    a_max = max(abs(f.flux_burgers_prime(Uo)))

    dt =  (dx * cfl)/a_max
    dt = min(dt,(T-time))
    
    time += dt 
    print(time)
    if time >= T:  # Condition to stop animation
        ani.event_source.stop()

ani = FuncAnimation(fig, animate, interval=25, repeat=False)  # Interval between frames in milliseconds
plt.show()
