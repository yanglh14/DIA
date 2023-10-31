import numpy as np
import matplotlib.pyplot as plt

# initial and final segment vertices
initial_vertices = np.array([[-1, 0], [1, 0]])
final_vertices = np.array([[3, 3], [3, 2]])

# calculate angle of rotation from initial to final segment
angle = np.arctan2(final_vertices[1, 1] - final_vertices[0, 1], final_vertices[1, 0] - final_vertices[0, 0]) - \
        np.arctan2(initial_vertices[1, 1] - initial_vertices[0, 1], initial_vertices[1, 0] - initial_vertices[0, 0])

# number of steps
steps = 200

# calculate angle of rotation for this step
rotation_angle = angle / steps

# translation vector: difference between final and initial centers
translation = (final_vertices.mean(axis=0) - initial_vertices.mean(axis=0))


# calculate incremental translation
translation_step = translation / steps

# initialize list of vertex positions
positions = [initial_vertices]

# apply translation and rotation in each step
for _ in range(steps):
    # translate vertices
    vertices = positions[-1] + translation_step

    # calculate rotation matrix for this step
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])

    # rotate vertices
    center = vertices.mean(axis=0)
    vertices = (rotation_matrix @ (vertices - center).T).T + center

    # append vertices to positions
    positions.append(vertices)

# append final vertices to positions
# positions.append(final_vertices)

# convert list of positions to numpy array
positions = np.array(positions)

# plot initial and final segments and movement paths
fig, ax = plt.subplots()

# plot initial and final segments
ax.plot(*positions[0].T, 'ro-', label='Initial segment')
ax.plot(*positions[-1].T, 'bo-', label='Final segment')
ax.plot(*final_vertices.T, 'go-', label='Desired segment')

# plot movement paths
for path in positions.transpose(1, 2, 0):
    ax.plot(*path, 'g--')

ax.set_aspect('equal', 'box')
ax.grid(True)
ax.legend()
plt.show()