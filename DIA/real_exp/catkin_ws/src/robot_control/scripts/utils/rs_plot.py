import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a list/array of points named 'point_cloud'
# with each point being a list/array [x, y, z, r, g, b]
# For example: point_cloud = [[x1, y1, z1, r1, g1, b1], [x2, y2, z2, r2, g2, b2], ...]

point_cloud = np.load('../../log/rs_data.npy')
# First, convert your point cloud to a numpy array for easier manipulation
point_cloud_np = np.array(point_cloud)

# only consider 0.5m ~ 2m view
point_cloud_np = point_cloud_np[point_cloud_np[:,2]<1.5]
point_cloud_np = point_cloud_np[point_cloud_np[:,2]>0.5]
point_cloud_np = point_cloud_np[point_cloud_np[:,4]>100]

# Split your NumPy array into positions (x, y, z) and colors (r, g, b)
positions = point_cloud_np[:, :3]
colors = point_cloud_np[:, 3:] / 255.0  # Assuming color channels are 0-255, normalize to 0-1 for matplotlib


# Create a new matplotlib figure and axis.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot using the x, y, and z coordinates and the color information
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=1)  # s is the size of the points

# Set labels for axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()