import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# =========================================================================
# 1. PARSE THE 2D TRAJECTORY LOG FILE
# =========================================================================
log_file_path = "/home/matteo/optimizer_ws/logs/qp_2d_trajectory.log"

iterations = {}
costs = {}

# Regex to identify lines containing iteration details and the state vector 'x='
# Handles spaces, scientific notation, and multi-line breaks smoothly
iter_pattern = re.compile(r"iter=(\d+),cost=([0-9.eE+-]+)")

with open(log_file_path, 'r') as f:
    log_content = f.read()

# Split log by lines but keep track of state vectors spanning entries
lines = log_content.split('\n')
current_iter = None

for i, line in enumerate(lines):
    match = iter_pattern.search(line)
    if match:
        current_iter = int(match.group(1))
        costs[current_iter] = float(match.group(2))
        
        # Extract the vector block following 'x='
        vector_tokens = []
        raw_vector_zone = line.split("x=")[1] if "x=" in line else ""
        
        # Look ahead to grab data if the matrix output wrapped lines
        look_ahead = i
        while look_ahead < len(lines):
            # Clean up token fragments
            tokens = lines[look_ahead].split("x=")[-1].strip().split() if look_ahead == i else lines[look_ahead].strip().split()
            
            # Stop if we hit the next timestamp log entry
            if look_ahead > i and ("[" in lines[look_ahead] or "iter=" in lines[look_ahead] or not lines[look_ahead].strip()):
                break
            
            for t in tokens:
                # Filter out raw text tags if any exist in the print block
                try:
                    vector_tokens.append(float(t))
                except ValueError:
                    continue
            look_ahead += 1
            
        if len(vector_tokens) == 100:
            iterations[current_iter] = np.array(vector_tokens)

# =========================================================================
# 2. SETUP DATA ARRAYS
# =========================================================================
sorted_iters = sorted(iterations.keys())
total_points = 50 # 50 points for X, 50 points for Y

# Custom strict optimization parameters declared in your C++ setup
eq_anchors = [(0, 0.0, 0.0), (15, 4.0, 2.0), (35, 7.0, 8.0), (49, 10.0, 10.0)]

# =========================================================================
# 3. GRAPHIC RENDERING: 2D SPATIAL PATH EVOLUTION
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [2, 1.2]})
fig.suptitle("FuriaOptimizer: 2D Trajectory Convergence Metrics", fontsize=14, fontweight='bold')

# Color map to visualize timeline flow across solver iterations
cmap = cm.get_cmap('coolwarm')
norm = Normalize(vmin=min(sorted_iters), vmax=max(sorted_iters))

for it in sorted_iters:
    full_vector = iterations[it]
    x_coords = full_vector[:total_points]
    y_coords = full_vector[total_points:]
    
    if it == 0:
        ax1.plot(x_coords, y_coords, 'k--', label="Initial Guess (u0)", alpha=0.7, linewidth=1.5)
    elif it == sorted_iters[-1]:
        ax1.plot(x_coords, y_coords, color='indigo', marker='o', markersize=4, 
                 linewidth=3, label=r"Optimized Output Profile $\mathbf{u^*}$", zorder=5)
    else:
        # Fade intermediate interior-point updates
        ax1.plot(x_coords, y_coords, color=cmap(norm(it)), alpha=0.4, linestyle='-')

# --- Draw Hard Constraints on Canvas ---
# Equality Constraints (Fixed Coordinates)
for idx, ex, ey in eq_anchors:
    ax1.plot(ex, ey, color='darkgreen', marker='o', markersize=9, mec='k', zorder=6, ls='')
    if idx == 0: ax1.text(ex+0.2, ey-0.1, "Start Anchor", fontweight='bold', color='darkgreen')
    elif idx == 49: ax1.text(ex-1.2, ey+0.2, "Goal Anchor", fontweight='bold', color='darkgreen')
    else: ax1.text(ex+0.2, ey-0.2, f"EQ Fix k={idx}", fontsize=9, color='darkgreen')

# Inequality Constraints (Directional Boundary Walls)
# Waypoint 10: X <= 1.5, Y >= 3.0
w10_x, w10_y = iterations[sorted_iters[-1]][10], iterations[sorted_iters[-1]][total_points + 10]
ax1.errorbar(1.5, 3.0, xerr=[[0.8], [0]], yerr=[[0], [0.8]], fmt='rx', markersize=10, mew=2, label="Bound Windows")
ax1.text(1.7, 3.1, "k=10\nX ≤ 1.5\nY ≥ 3.0", color='red', fontsize=9)

# Waypoint 25: X >= 6.0, Y <= 4.5
w25_x, w25_y = iterations[sorted_iters[-1]][25], iterations[sorted_iters[-1]][total_points + 25]
ax1.errorbar(6.0, 4.5, xerr=[[0], [0.8]], yerr=[[0.8], [0]], fmt='rx', markersize=10, mew=2)
ax1.text(5.0, 4.7, "k=25\nX ≥ 6.0\nY ≤ 4.5", color='red', fontsize=9)

# Format Left Layout
ax1.set_title("2D Parametric Trajectory Space Grid Mapping")
ax1.set_xlabel("X Space Units")
ax1.set_ylabel("Y Space Units")
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_xlim(-0.5, 11.0)
ax1.set_ylim(-0.5, 11.0)
ax1.legend(loc="upper left")

# Add colorbar for iteration timeline reference
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.1)
cbar.set_label('Interior Point Step Iteration Count')

# =========================================================================
# 4. GRAPHIC RENDERING: OBJECTIVE COST LOSS ATTENUATION
# =========================================================================
ordered_cost_keys = sorted(costs.keys())
cost_values = [costs[k] for k in ordered_cost_keys]

ax2.plot(ordered_cost_keys, cost_values, marker='s', color='crimson', linewidth=2, label='Energy Cost')
ax2.set_title("Hessian Acceleration Minimization Profile")
ax2.set_xlabel("Iteration Count")
ax2.set_ylabel("Discrete Cost Value")
ax2.set_yscale('log') # Smooth rendering over sharp drops
ax2.grid(True, which="both", linestyle=':', alpha=0.5)
ax2.set_xticks(ordered_cost_keys)

# Annotate Convergence Target achieved
ax2.annotate(f'Final Value: {cost_values[-1]:.5f}', 
             xy=(ordered_cost_keys[-1], cost_values[-1]), 
             xytext=(ordered_cost_keys[-1] - 4, cost_values[-1] * 3),
             arrowprops=dict(facecolor='black', shrink=0.08, width=0.5, headwidth=5))

plt.tight_layout()
plt.show()