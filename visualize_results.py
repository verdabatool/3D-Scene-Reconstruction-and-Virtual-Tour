# visualize_week3_open3d.py
import open3d as o3d
import numpy as np
import os

def visualize_week3_results():
    """Visualize your complete Week 3 reconstruction"""
    print("WEEK 3: 3D Scene Reconstruction Visualization")
    print("=" * 60)
    
    # Your reconstruction files
    files = {
        "outputs/week3_initial.ply": {"color": [0.1, 0.1, 0.8], "name": "INITIAL (2 cameras)"},
        "outputs/week3_before_ba.ply": {"color": [0.8, 0.1, 0.1], "name": "BEFORE Bundle Adjustment"},
        "outputs/week3_after_ba.ply": {"color": [0.1, 0.8, 0.1], "name": "AFTER Bundle Adjustment - FINAL"}
    }
    
    all_geometries = []
    
    for filename, config in files.items():
        if not os.path.exists(filename):
            print(f"❌ {filename} not found")
            continue
            
        print(f"📁 Loading {filename}...")
        
        try:
            # Load point cloud
            pcd = o3d.io.read_point_cloud(filename)
            
            # Color the point cloud
            pcd.paint_uniform_color(config["color"])
            
            # Add to visualization
            all_geometries.append(pcd)
            
            print(f"✅ {config['name']}: {len(pcd.points):,} points")
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
    
    if not all_geometries:
        print("❌ No point clouds found!")
        return
    
    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    all_geometries.append(coord_frame)
    
    # Visualize
    print("\n🎯 Controls:")
    print("  - Mouse drag: Rotate")
    print("  - Mouse wheel: Zoom") 
    print("  - 'H': Show help")
    print("  - 'Q': Quit")
    print("\n🟦 Blue: Initial reconstruction")
    print("🟥 Red: Before Bundle Adjustment") 
    print("🟩 Green: After Bundle Adjustment (Your submission)")
    
    o3d.visualization.draw_geometries(all_geometries,
                                    window_name="WEEK 3: 3D Scene Reconstruction",
                                    width=1200,
                                    height=800)

if __name__ == "__main__":
    visualize_week3_results()