from DIA.utils.utils import downsample, load_data, load_data_list, store_h5_data, voxelize_pointcloud, pc_reward_model
import numpy as np

picker_position_list = []
for i in range(1800):
    data_cur = load_data('./data/dia_baseline/train', i, 99, ['picker_position'])
    picker_position_list.append(data_cur['picker_position'][0][0])
picker_position_list = np.array(picker_position_list)
print(picker_position_list.shape)