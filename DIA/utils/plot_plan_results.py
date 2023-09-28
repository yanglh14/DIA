import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('seaborn-whitegrid')

# Define your font settings
font = {
    'family': 'Times New Roman',  # Font family (e.g., 'serif', 'sans-serif', 'monospace')
    'weight': 'normal',  # Font weight ('normal', 'bold', 'light', 'heavy')
    'size': 18,  # Font size
    'style': 'normal'  # Font style ('normal', 'italic', 'oblique')
}

# Set the font properties
plt.rcParams['font.family'] = font['family']
plt.rcParams['font.weight'] = font['weight']
plt.rcParams['font.size'] = font['size']
plt.rcParams['font.style'] = font['style']

performance_dir = {}

sourece_folde = '../data/'
exp_names = ['dia_v4_pick_drop','dia_platform2', 'dia_sphere','dia_rod']
for exp_name in exp_names:
    exp_dir = os.path.join(sourece_folde, exp_name)
    trials = os.listdir(exp_dir)
    trials = [trial for trial in trials if 'plan' in trial]
    for trial in trials:
        #load a csv file
        normalized_performance_file = os.path.join(exp_dir, trial, 'all_normalized_performance.pkl')
        performance_file = os.path.join(exp_dir, trial, 'all_performance.npy')

        if os.path.exists(performance_file) and os.path.exists(normalized_performance_file):
            performance = np.load(performance_file, allow_pickle=True)
            normalized_performance = np.load(normalized_performance_file, allow_pickle=True)

            performance_dir[os.path.join(exp_name,trial)] = [performance.mean(), np.array(normalized_performance)[:,-1].mean()]

# create the table
table_data = []
for key, value in performance_dir.items():
    table_data.append([key] + value)

headers = ["", "performance", "normalized performance"]

# Create the table using matplotlib
fig, ax = plt.subplots()
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

# Save the table as a PNG image
plt.show()
# plt.savefig(os.path.join(source_dir, 'table.png'))
plt.close()





