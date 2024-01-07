import numpy as np
import matplotlib.pyplot as plt

def plot_pc():
    point_cloud = np.load('../../log/transformed_pc.npy')
    # First, convert your point cloud to a numpy array for easier manipulation
    point_cloud_np = np.array(point_cloud)

    # only consider 0.5m ~ 2m view
    # point_cloud_np = point_cloud_np[point_cloud_np[:,2]<1.5]
    # point_cloud_np = point_cloud_np[point_cloud_np[:,2]>0.5]
    # point_cloud_np = point_cloud_np[point_cloud_np[:,4]>100]

    # Split your NumPy array into positions (x, y, z) and colors (r, g, b)
    positions = point_cloud_np[:, :3]
    # colors = point_cloud_np[:, 3:] / 255.0  # Assuming color channels are 0-255, normalize to 0-1 for matplotlib


    # Create a new matplotlib figure and axis.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot using the x, y, and z coordinates and the color information
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=[0,0,1], s=1)  # s is the size of the points

    # Set labels for axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()

def plot_depth():

    depth = np.load('../../log/depth.npy')
    plt.imshow(depth)
    plt.show()



if __name__ == '__main__':
    # plot_depth()
    plot_pc()
