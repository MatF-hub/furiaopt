import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

# 1. Parse log file
path_x, path_y = [], []
# Regex to match coordinates after 'x=' handling signs, decimals, and scientific notation
pattern = re.compile(r"x=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

with open('/home/matteo/optimizer_ws/logs/optimizer.log', 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            path_x.append(float(match.group(1)))
            path_y.append(float(match.group(2)))

path_x, path_y = np.array(path_x), np.array(path_y)

# 2. Define Function and Path Z-values
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

path_z = rosenbrock(path_x, path_y)

# 3. Create Grid for Surface
X_grid = np.linspace(-2.5, 2.5, 400)
Y_grid = np.linspace(-1.5, 2.5, 400)
X, Y = np.meshgrid(X_grid, Y_grid)
Z = rosenbrock(X, Y)

# 4. Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', norm=LogNorm(), alpha=0.5, edgecolor='none')

# Optimization Line
ax.plot(path_x, path_y, path_z, color='red', linewidth=2, label='Optimizer Path', zorder=10)

# Start/End Markers
ax.scatter(path_x[0], path_y[0], path_z[0], color='green', s=100, label='Start')
ax.scatter(path_x[-1], path_y[-1], path_z[-1], color='black', s=100, marker='x',label='End')

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('f(X, Y)')
ax.view_init(elev=35, azim=-120)
plt.legend()
plt.show()