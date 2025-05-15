import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np



# Load the CSV file
csv_file = "hand_3d_data.csv"
data = pd.read_csv(csv_file)

# Landmark labels
landmark_names = ['Wrist', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
fingertip_ids = [4, 8, 12, 16, 20]

# Extract positions
frames = data['Frame'].values
wrist_xyz = data[['Wrist_X', 'Wrist_Y', 'Wrist_Z']].values
finger_xyz = [data[[f'F{fid}_X', f'F{fid}_Y', f'F{fid}_Z']].values for fid in fingertip_ids]

scale = 10
wrist_xyz *= scale
finger_xyz = [f * scale for f in finger_xyz]

# Optional: set up animation
enable_animation = True

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')

def plot_frame(idx):
    ax.cla()  # Clear axes

    # Set axes limits dynamically based on full dataset
    all_points = np.concatenate([wrist_xyz] + finger_xyz)
    buffer = 50
    ax.set_xlim([np.min(all_points[:, 0])-buffer, np.max(all_points[:, 0])+buffer])
    ax.set_ylim([np.min(all_points[:, 1])-buffer, np.max(all_points[:, 1])+buffer])
    ax.set_zlim([np.min(all_points[:, 2])-buffer, np.max(all_points[:, 2])+buffer])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame {frames[idx]} - Hand in 3D')

    # Plot wrist
    wrist = wrist_xyz[idx]
    ax.scatter(*wrist, color='red', s=100, label='Wrist')
    ax.text(*wrist, 'Wrist', fontsize=9)

    # Plot fingertips and connect them to wrist
    for i, fid in enumerate(fingertip_ids):
        fingertip = finger_xyz[i][idx]
        ax.scatter(*fingertip, s=60, label=landmark_names[i+1])
        ax.plot([wrist[0], fingertip[0]], [wrist[1], fingertip[1]], [wrist[2], fingertip[2]], color='gray')
        ax.text(*fingertip, landmark_names[i+1], fontsize=9)

    ax.legend(loc='upper right')

    

if enable_animation:
    import time
    for i in range(len(frames)):
        plot_frame(i)
        plt.pause(0.1)  # Delay between frames (adjust as needed)
    plt.show()
else:
    plot_frame(0)
    plt.show()
