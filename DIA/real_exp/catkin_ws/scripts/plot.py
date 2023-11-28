import numpy as np
import matplotlib.pyplot as plt

move_log = np.load('move_log.npy')
trajectory_log = np.load('trajectory_log.npy')
# plot two subplots: one for move log one for trajectory log

time_1 = move_log[:,0] - move_log[0,0]
move_log = move_log[:,2]

lowest_index = np.argmin(move_log)
time_1 = time_1[lowest_index+3:lowest_index+30]
move_log = move_log[lowest_index+3:lowest_index+30]


time_2 = trajectory_log[:,0] - trajectory_log[0,0]
trajectory_log = -trajectory_log[:,1]

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(time_1, move_log)
# ax1.set_title('Move Log')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Move Log')
#
# ax2.plot(time_2, trajectory_log)
# ax2.set_title('Trajectory Log')
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Trajectory Log')
#
# plt.tight_layout()
# plt.show()

# ignore time, plot move_log and trajectory_log on same plot
plt.plot(move_log, label='Move Log')
plt.plot(trajectory_log, label='Trajectory Log')
plt.title('Move Log and Trajectory Log')
plt.xlabel('Time (s)')
plt.ylabel('Log')
plt.legend()
plt.show()

