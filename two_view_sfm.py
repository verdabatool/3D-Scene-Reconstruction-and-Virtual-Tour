# Week 2: Two-View Reconstruction
# This script implements the two-view SfM pipeline per assignment requirements:
# - K constructed as fx = fy = image width, cx,cy = image center
# - SIFT feature detection, FLANN matching, Lowe's ratio
# - Essential matrix estimation with cv2.findEssentialMat (RANSAC)
# - Decompose E into 4 poses with cv2.decomposeEssentialMat
# - Triangulate for each pose (cv2.undistortPoints + cv2.triangulatePoints)
# - Cheirality check to select correct [R|t]
# - Save sparse point cloud to an ASCII .ply with RGB colors sampled from image1
#
# Note: A screenshot related to the assignment was uploaded and is available locally at:
# /mnt/data/Screenshot 2025-11-20 at 10.37.39 AM.heic

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def read_image_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def build_intrinsics_from_width(img_bgr):
    H, W = img_bgr.shape[:2]
    fx = float(W)
    fy = float(W)
    cx = W / 2.0
    cy = H / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K


def get_sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    try:
        return cv2.xfeatures2d.SIFT_create()
    except Exception:
        raise RuntimeError("SIFT not available. Install opencv-contrib-python.")


def detect_and_compute(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    sift = get_sift()
    kps, des = sift.detectAndCompute(gray, None)
    return kps, des


def match_flann(des1, des2, ratio_thresh=0.75):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good


def draw_matches(img1_bgr, kp1, img2_bgr, kp2, matches, title=None, max_draw=200):
    imgm = cv2.drawMatches(img1_bgr, kp1, img2_bgr, kp2, matches[:max_draw], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(14,7))
    plt.imshow(cv2.cvtColor(imgm, cv2.COLOR_BGR2RGB))
    if title: plt.title(title)
    plt.axis('off')
    plt.show()


def write_ply_ascii(filename, points, colors=None):
    n = points.shape[0]
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x,y,z = points[i]
            if colors is not None:
                r,g,b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def triangulate_and_count_cheirality(K, R, t, pts1_px, pts2_px, inlier_mask):
    # Select inliers
    pts1 = pts1_px[inlier_mask].reshape(-1,1,2)
    pts2 = pts2_px[inlier_mask].reshape(-1,1,2)

    # Convert to normalized coordinates via undistortPoints
    pts1_norm = cv2.undistortPoints(pts1, K, None)
    pts2_norm = cv2.undistortPoints(pts2, K, None)

    # Projection matrices in normalized coords
    P0 = np.hstack((np.eye(3), np.zeros((3,1))))
    P1 = np.hstack((R, t.reshape(3,1)))

    pts4d = cv2.triangulatePoints(P0, P1, pts1_norm.reshape(-1,2).T, pts2_norm.reshape(-1,2).T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    # Cheirality
    z0 = pts3d[:,2]
    pts_cam1 = (R.dot(pts3d.T) + t.reshape(3,1)).T
    z1 = pts_cam1[:,2]
    mask_front = (z0 > 0) & (z1 > 0)
    return pts3d, mask_front, pts1.reshape(-1,2)


def two_view_strict(img1_path, img2_path, ply_out='outputs/reconstruction_two_view.ply',
                    ratio_thresh=0.75, ransac_thresh=1.0, show_matches=False, visualize=False):
    img1_bgr = read_image_bgr(img1_path)
    img2_bgr = read_image_bgr(img2_path)

    K = build_intrinsics_from_width(img1_bgr)
    print("K =\n", K)

    kp1, des1 = detect_and_compute(img1_bgr)
    kp2, des2 = detect_and_compute(img2_bgr)
    print("Detected keypoints:", len(kp1), "and", len(kp2))

    matches = match_flann(des1, des2, ratio_thresh=ratio_thresh)
    print("Matches after Lowe ratio:", len(matches))
    if len(matches) < 8:
        raise RuntimeError("Too few matches after ratio test - pick another image pair or loosen ratio.")

    if show_matches:
        draw_matches(img1_bgr, kp1, img2_bgr, kp2, matches, title="Filtered matches")

    pts1_px = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2_px = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    E, mask = cv2.findEssentialMat(pts1_px, pts2_px, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh)
    if E is None:
        raise RuntimeError("Essential matrix estimation failed.")
    mask = mask.ravel().astype(bool)
    print("Inliers from findEssentialMat:", int(mask.sum()), "/", len(matches))

    R1, R2, t = cv2.decomposeEssentialMat(E)
    poses = [
        (R1,  t),
        (R2,  t),
        (R1, -t),
        (R2, -t),
    ]

    best = None
    best_count = -1
    best_pts3d = None
    best_mask_front = None
    best_pts1_used = None

    for i, (R, c_t) in enumerate(poses):
        pts3d, mask_front, pts1_used = triangulate_and_count_cheirality(K, R, c_t, pts1_px, pts2_px, inlier_mask=mask)
        count_front = int(mask_front.sum())
        print(f"Pose {i}: {count_front} / {pts3d.shape[0]} points in front of both cameras")
        if count_front > best_count:
            best_count = count_front
            best = (R, c_t)
            best_pts3d = pts3d
            best_mask_front = mask_front
            best_pts1_used = pts1_used

    if best is None:
        raise RuntimeError("Pose disambiguation failed.")

    R_final, t_final = best
    print("Selected pose with", best_count, "points in front.")

    pts3d_valid = best_pts3d[best_mask_front]
    pts1_coords_for_colors = best_pts1_used[best_mask_front].astype(int)

    if pts3d_valid.shape[0] == 0:
        raise RuntimeError("No valid 3D points after cheirality test.")

    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    colors = []
    for (x,y) in pts1_coords_for_colors:
        x = np.clip(int(round(x)), 0, img1_rgb.shape[1]-1)
        y = np.clip(int(round(y)), 0, img1_rgb.shape[0]-1)
        colors.append(img1_rgb[y, x])
    colors = np.array(colors, dtype=np.uint8)

    os.makedirs(os.path.dirname(ply_out) or '.', exist_ok=True)
    write_ply_ascii(ply_out, pts3d_valid, colors)
    print("Saved PLY:", ply_out)

    if visualize:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts3d_valid[:,0], pts3d_valid[:,1], pts3d_valid[:,2], c=colors/255.0, s=2)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        plt.title('Sparse 3D point cloud')
        plt.show()

    return {
        "K": K, "R": R_final, "t": t_final, "points": pts3d_valid, "colors": colors,
        "matches": matches
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img1", default="./Data/0001.jpeg", help="first image")
    p.add_argument("--img2", default="./Data/0002.jpeg", help="second image")
    p.add_argument("--out", default="outputs/reconstruction_two_view.ply", help="output PLY file")
    p.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio")
    p.add_argument("--ransac_thresh", type=float, default=1.0, help="RANSAC threshold (pixels) for findEssentialMat")
    p.add_argument("--show_matches", action="store_true", help="show 2D matches plot")
    p.add_argument("--visualize", action="store_true", help="show 3D scatter plot")
    args = p.parse_args()

    two_view_strict(args.img1, args.img2, ply_out=args.out, ratio_thresh=args.ratio,
                    ransac_thresh=args.ransac_thresh, show_matches=args.show_matches,
                    visualize=args.visualize)
