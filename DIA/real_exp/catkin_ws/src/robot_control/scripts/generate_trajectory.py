import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def plot(trajectory, dt):

    vel_x = np.diff(trajectory[:, 0])/dt
    vel_y = np.diff(trajectory[:, 1])/dt
    vel_z = np.diff(trajectory[:, 2])/dt

    acc_x = np.diff(vel_x)/dt
    acc_y = np.diff(vel_y)/dt
    acc_z = np.diff(vel_z)/dt

    # plot the trajectory
    fig, (ax1, ax2, ax3) = plt.subplots(3, 3)
    ax1[0].plot(trajectory[:, 0], label='Trajectory')
    ax1[0].set_title('pos x')
    ax1[0].set_xlabel('Time (s)')
    ax1[0].set_ylabel('Position (m)')
    ax1[0].legend()
    ax1[1].plot(trajectory[:, 1], label='Trajectory')
    ax1[1].set_title('pos y')
    ax1[1].set_xlabel('Time (s)')
    ax1[1].set_ylabel('Position (m)')
    ax1[1].legend()
    ax1[2].plot(trajectory[:, 2], label='Trajectory')
    ax1[2].set_title('pos z')
    ax1[2].set_xlabel('Time (s)')
    ax1[2].set_ylabel('Position (m)')
    ax1[2].legend()

    ax2[0].plot(vel_x, label='Trajectory')
    ax2[0].set_title('vel x')
    ax2[0].set_xlabel('Time (s)')
    ax2[0].set_ylabel('Velocity (m/s)')
    ax2[0].legend()
    ax2[1].plot(vel_y, label='Trajectory')
    ax2[1].set_title('vel y')
    ax2[1].set_xlabel('Time (s)')
    ax2[1].set_ylabel('Velocity (m/s)')
    ax2[1].legend()
    ax2[2].plot(vel_z, label='Trajectory')
    ax2[2].set_title('vel z')
    ax2[2].set_xlabel('Time (s)')
    ax2[2].set_ylabel('Velocity (m/s)')
    ax2[2].legend()

    ax3[0].plot(acc_x, label='Trajectory')
    ax3[0].set_title('acc x')
    ax3[0].set_xlabel('Time (s)')
    ax3[0].set_ylabel('Acceleration (m/s^2)')
    ax3[0].legend()
    ax3[1].plot(acc_y, label='Trajectory')
    ax3[1].set_title('acc y')
    ax3[1].set_xlabel('Time (s)')
    ax3[1].set_ylabel('Acceleration (m/s^2)')
    ax3[1].legend()
    ax3[2].plot(acc_z, label='Trajectory')
    ax3[2].set_title('acc z')
    ax3[2].set_xlabel('Time (s)')
    ax3[2].set_ylabel('Acceleration (m/s^2)')
    ax3[2].legend()


    plt.show()

def generate_trajectory(current_pos, target_pos, acc_max):
    dt = 0.01
    translation = (target_pos - current_pos)

    steps_cost = max(np.sqrt(4*np.abs(translation)/acc_max)/dt)

    time_steps = np.ceil(steps_cost).max().astype(int)

    accel_steps = time_steps/2
    decel_steps = time_steps - accel_steps

    v_max = translation * 2 / (time_steps * dt)
    accelerate = v_max / (accel_steps * dt)
    decelerate = -v_max / (decel_steps * dt)

    incremental_translation = [0, 0, 0]
    positions_xyz = [current_pos]
    for i in range(time_steps):
        if i < accel_steps:
            # Acceleration phase
            incremental_translation = (np.divide(incremental_translation,
                                                 dt) + accelerate * dt) * dt
        else:
            # Deceleration phase
            incremental_translation = (np.divide(incremental_translation,
                                                 dt) + decelerate * dt) * dt

        # translate vertices
        vertices = positions_xyz[-1] + incremental_translation

        positions_xyz.append(vertices)

    positions_xyz = np.array(positions_xyz)

    return positions_xyz

# Test the function
current_pos = np.array([0, 0, 0.5])

middle_pos = np.array([0.3, 0, 0.2])
target_pos = np.array([0.2, 0, 0])
acc = 2

trajectory_1 = generate_trajectory(current_pos, middle_pos, acc)

trajectory_2 = generate_trajectory(middle_pos, target_pos, acc)

trajectory = np.concatenate((trajectory_1, trajectory_2[1:]), axis=0)
plot(trajectory, 0.01)