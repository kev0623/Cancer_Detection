import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nx = 28
ny = 28
nz = 28
r_min = 0.2
r_max = 0.8
# Create a figure with a black background
fig = plt.figure(facecolor="black")
ax = plt.axes(projection="3d")

# Define parameters for the sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
r = np.random.uniform(r_min, r_max,1)*nx

# x, y, z coordinates for the sphere
x0 = np.random.uniform(r_min, r_max, 1) * nx
y0 = np.random.uniform(r_min, r_max, 1) * ny 
z0 = np.random.uniform(r_min, r_max, 1) * nz
x = r * np.outer(np.cos(u), np.sin(v)) + x0
y = r * np.outer(np.sin(u), np.sin(v)) + y0
z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z0

# Plot the sphere with the approximated color from your image
ax.plot_surface(x, y, z, color="#F4A460")  # Use a peach-like color

# Set plot limits to keep it centered and remove the axis lines
ax.set_xlim(-28, 28)
ax.set_ylim(-28, 28)
ax.set_zlim(-28, 28)

# Display and save the plot as an image if running in a headless environment
plt.savefig("approximated_sphere.png", dpi=300, bbox_inches='tight', facecolor="black")
plt.show()
