import numpy as np
import matplotlib.pyplot as plt

traj_log = np.load('DIA/real_exp/catkin_ws/src/robot_control/results/traj_log.npy')
traj_desired = np.load('DIA/real_exp/catkin_ws/src/robot_control/results/traj_desired.npy')

# plot two subplots: one for move log one for trajectory log

time_log = traj_log[:,0] - traj_log[0,0]
traj_log_x = traj_log[:,2]
# find the index of start of trajectory, the x will first decrease then increase
diff_signal = np.diff(traj_log_x)
for i in range(len(diff_signal)):
    if abs(diff_signal[i]) > 5e-4:
        index_start = i
        break

traj_log_x_index = traj_log_x[index_start:index_start+100]
time_log_index = time_log[index_start:index_start+100]

time_desired = traj_desired[:,0] - traj_desired[0,0]
traj_desired_x = traj_desired[:,2]

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(time_log, traj_log_x)
# ax1.set_title('Traj Log')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Traj Log')

# ax2.plot(time_desired, traj_desired_x)
# ax2.set_title('Desired Trajectory')
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Desired Trajectory')

# plt.tight_layout()
# plt.show()

# ignore time, plot move_log and trajectory_log on same plot
plt.plot(traj_log_x_index, label='Move Log')
plt.plot(traj_desired_x, label='Trajectory Log')
plt.title('Move Log and Trajectory Log')
plt.xlabel('Time (s)')
plt.ylabel('Log')
plt.legend()
plt.show()

