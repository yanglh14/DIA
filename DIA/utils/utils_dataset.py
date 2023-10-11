import numpy as np
def _collect_policy_rod(self):
    """ Policy for collecting data for sphere shape, swing and drop"""

    acc_delta_value = np.random.uniform(
        self.args.collect_data_delta_acc_min,
        self.args.collect_data_delta_acc_max, size=self.args.time_step)

    action_list = []

    for step in range(1, self.args.time_step):
        if step == 1:
            self.state = np.zeros(6, dtype=np.float32)

        if step <= self.args.time_step * 0.8 / 4:
            acc_direction = np.array([6, -1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step * 0.8 / 2:
            acc_direction = np.array([-6, -1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step * 0.8 * 3 / 4:
            acc_direction = np.array([-3, 1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step * 0.8:
            acc_direction = np.array([3, 1.6, 0], dtype=np.float32)
        else:
            acc_direction = np.array([0, 0, 0], dtype=np.float32)

        acc_direction[0] += acc_delta_value[step]

        self.state[3:6] = acc_direction * self.args.dt

        self.state[0:3] += self.state[3:6] * self.args.dt

        action = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        action[:3], action[4:7] = self.state[0:3], self.state[0:3]

        if not step > self.args.time_step * 0.8:
            action[7], action[3] = 1, 1
        else:
            action[7], action[3] = 0, 0

        action_list.append(action)

    return action_list


def _collect_policy_sphere(self):
    """ Policy for collecting data for sphere shape, swing and drop"""

    acc_delta_value = np.random.uniform(
        self.args.collect_data_delta_acc_min,
        self.args.collect_data_delta_acc_max, size=self.args.time_step)

    action_list = []

    for step in range(1, self.args.time_step):
        if step == 1:
            self.state = np.zeros(6, dtype=np.float32)

        if step <= self.args.time_step * 0.8 / 4:
            acc_direction = np.array([6, -1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step * 0.8 / 2:
            acc_direction = np.array([-6, -1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step * 0.8 * 3 / 4:
            acc_direction = np.array([-3, 1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step * 0.8:
            acc_direction = np.array([3, 1.6, 0], dtype=np.float32)
        else:
            acc_direction = np.array([0, 0, 0], dtype=np.float32)

        acc_direction[0] += acc_delta_value[step]

        self.state[3:6] = acc_direction * self.args.dt

        self.state[0:3] += self.state[3:6] * self.args.dt

        action = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        action[:3], action[4:7] = self.state[0:3], self.state[0:3]

        if not step > self.args.time_step * 0.8:
            action[7], action[3] = 1, 1
        else:
            action[:7] = 0

        action_list.append(action)

    return action_list


def _collect_policy_platform_v3(self):
    """ Policy for collecting data - dia_platform3: not release, keep low vel in the last steps, env to large platform """

    acc_delta_value = np.random.uniform(
        self.args.collect_data_delta_acc_min,
        self.args.collect_data_delta_acc_max, size=self.args.time_step)
    bias = np.sum(acc_delta_value[1:int(self.args.time_step / 2)]) / (
                self.args.time_step - 1 - int(self.args.time_step / 2))

    action_list = []

    for step in range(1, self.args.time_step):
        if step == 1:
            self.state = np.zeros(6, dtype=np.float32)

        if step <= self.args.time_step / 4:
            acc_direction = np.array([6, -1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step / 2:
            acc_direction = np.array([-6, -1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step * 3 / 4:
            acc_direction = np.array([-3 - bias, 1.6, 0], dtype=np.float32)

        else:
            acc_direction = np.array([3 - bias, 1.6, 0], dtype=np.float32)

        if step <= self.args.time_step / 2:
            acc_direction[0] += acc_delta_value[step]

        self.state[3:6] = acc_direction * self.args.dt

        self.state[0:3] += self.state[3:6] * self.args.dt

        action = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        action[:3], action[4:7] = self.state[0:3], self.state[0:3]
        action[7], action[3] = 1, 1

        # if not step> self.args.time_step*3/4:
        #     action[7],action[3] = 1,1
        # else:
        #     action[7],action[3] = 0,0

        action_list.append(action)

    return action_list


def _collect_policy_platform(self):
    """ Policy for collecting data - acceleration control and ref trajectory - platform shape"""
    acc_delta_value = np.random.uniform(
        self.args.collect_data_delta_acc_min,
        self.args.collect_data_delta_acc_max, size=self.args.time_step)

    action_list = []

    for step in range(1, self.args.time_step):
        if step == 1:
            self.state = np.zeros(6, dtype=np.float32)

        if step <= self.args.time_step / 4:
            acc_direction = np.array([6, -1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step / 2:
            acc_direction = np.array([-6, -1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step * 3 / 4:
            acc_direction = np.array([-3, 1.6, 0], dtype=np.float32)

        else:
            acc_direction = np.array([3, 1.6, 0], dtype=np.float32)

        acc_direction[0] += acc_delta_value[step]

        self.state[3:6] = acc_direction * self.args.dt

        self.state[0:3] += self.state[3:6] * self.args.dt

        action = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        action[:3], action[4:7] = self.state[0:3], self.state[0:3]

        if not step > self.args.time_step * 3 / 4:
            action[7], action[3] = 1, 1
        else:
            action[7], action[3] = 0, 0

        action_list.append(action)

    return action_list


def _collect_policy_pick_drop(self):
    """ Policy for collecting data - acceleration control and ref trajectory - no shape - pick and drop """
    acc_delta_value = np.random.uniform(
        self.args.collect_data_delta_acc_min,
        self.args.collect_data_delta_acc_max, size=self.args.time_step)

    action_list = []

    for step in range(1, self.args.time_step):
        if step == 1:
            self.state = np.zeros(6, dtype=np.float32)

        if step <= self.args.time_step / 4:
            acc_direction = np.array([6, -1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step / 2:
            acc_direction = np.array([-6, -1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step * 3 / 4:
            acc_direction = np.array([-3, 1.6, 0], dtype=np.float32)

        else:
            acc_direction = np.array([3, 1.6, 0], dtype=np.float32)

        acc_direction[0] += acc_delta_value[step]

        self.state[3:6] = acc_direction * self.args.dt

        self.state[0:3] += self.state[3:6] * self.args.dt

        action = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        action[:3], action[4:7] = self.state[0:3], self.state[0:3]

        if not step > self.args.time_step * 3 / 4:
            action[7], action[3] = 1, 1
        else:
            action[7], action[3] = 0, 0

        action_list.append(action)

    return action_list


def _collect_policy_v3(self):
    """ Policy for collecting data - acceleration control and ref trajectory - Version3 and platform shape"""
    acc_delta_value = np.random.uniform(
        self.args.collect_data_delta_acc_min,
        self.args.collect_data_delta_acc_max, size=self.args.time_step)

    action_list = []

    for step in range(1, self.args.time_step):
        if step == 1:
            self.state = np.zeros(6, dtype=np.float32)

        if step <= self.args.time_step / 4:
            acc_direction = np.array([6, -1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step / 2:
            acc_direction = np.array([-6, -1.6, 0], dtype=np.float32)

        elif step <= self.args.time_step * 3 / 4:
            acc_direction = np.array([-3, 1.6, 0], dtype=np.float32)

        else:
            acc_direction = np.array([3, 1.6, 0], dtype=np.float32)

        acc_direction[0] += acc_delta_value[step]

        self.state[3:6] = acc_direction * self.args.dt

        self.state[0:3] += self.state[3:6] * self.args.dt

        action = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        action[:3], action[4:7] = self.state[0:3], self.state[0:3]
        action[7], action[3] = 1, 1

        action_list.append(action)

    return action_list


def _collect_policy_v2(self):
    """ Policy for collecting data - acceleration control and ref trajectory"""
    acc_delta_value = np.random.uniform(
        self.args.collect_data_delta_acc_min,
        self.args.collect_data_delta_acc_max, size=self.args.time_step)

    action_list = []

    for step in range(1, self.args.time_step):
        if step == 1:
            self.state = np.zeros(6, dtype=np.float32)

        if step <= self.args.time_step / 4:
            acc_direction = np.array([2., -0.3, 0], dtype=np.float32)

        elif step <= self.args.time_step / 2:
            acc_direction = np.array([-2., -0.3, 0], dtype=np.float32)

        elif step <= self.args.time_step * 3 / 4:
            acc_direction = np.array([-0.5, 0.3, 0], dtype=np.float32)

        else:
            acc_direction = np.array([0.5, 0.3, 0], dtype=np.float32)

        acc_direction[0] += acc_delta_value[step]

        self.state[3:6] = acc_direction * self.args.dt

        self.state[0:3] += self.state[3:6] * self.args.dt

        action = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        action[:3], action[4:7] = self.state[0:3], self.state[0:3]
        action[7], action[3] = 1, 1

        action_list.append(action)

    return action_list