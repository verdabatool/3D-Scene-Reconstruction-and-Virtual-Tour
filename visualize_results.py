# visualize_week3_matplotlib.py
import numpy as np
import os
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_ply(filename):
    """Load PLY point cloud and return Nx3 numpy array"""
    plydata = PlyData.read(filename)
    vertex = plydata['vertex']
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    return points

def visualize_week3_results():
    print("WEEK 3: 3D Scene Reconstruction Visualization (Matplotlib)")
    print("=" * 60)

    files = {
        "outputs/week3_initial.ply": {"color": "b", "name": "INITIAL (2 cameras)"},
        "outputs/week3_before_ba.ply": {"color": "r", "name": "BEFORE Bundle Adjustment"},
        "outputs/week3_after_ba.ply": {"color": "g", "name": "AFTER Bundle Adjustment - FINAL"}
    }

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for filename, config in files.items():
        if not os.path.exists(filename):
            print(f"❌ {filename} not found")
            continue

        print(f"📁 Loading {filename}...")
        try:
            points = load_ply(filename)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       s=1, c=config["color"], label=config["name"])
            print(f"✅ {config['name']}: {len(points):,} points")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    visualize_week3_results()
