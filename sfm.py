"""
week3.py

Week 3: Multi-View SfM with Refinement (complete script)
Saves outputs into the pre-existing 'outputs' directory.
Dependencies: opencv-python (and opencv-contrib if SIFT isn't in main opencv),
numpy, scipy
Run: python week3_guaranteed_success.py --data_dir ./Data
"""

import os
import argparse
import cv2
import numpy as np
from collections import defaultdict
from scipy.optimize import least_squares
import json
import traceback

# -------------------------
# Utility I/O & PLY writer
# -------------------------
def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def write_ply_ascii(filename, points, colors=None):
    n = points.shape[0]
    ensure_dir(os.path.dirname(filename) or ".")
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

# -------------------------
# SFMPipeline implementation
# -------------------------
class SFMPipeline:
    def __init__(self, focal_length=800.0, max_image_dim=1200, outputs_dir="outputs"):
        self.focal_length = focal_length
        self.max_image_dim = max_image_dim
        self.outputs_dir = outputs_dir
        ensure_dir(self.outputs_dir)

        self.images = []          # BGR images
        self.image_paths = []
        self.keypoints = []       # list of lists of cv2.KeyPoint
        self.descriptors = []     # list of descriptors arrays
        self.tracks = {}          # track_id -> list of (img_idx, kp_idx)
        self.cameras = {}         # img_idx -> (R (3x3), t (3x1))
        self.points3D = {}        # track_id -> (x,y,z)
        self.colors = {}          # track_id -> rgb
        self.K = None

    # -------------------------
    # IO and feature helpers
    # -------------------------
    def read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        h, w = img.shape[:2]
        if max(h, w) > self.max_image_dim:
            scale = self.max_image_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def get_sift(self, nfeatures=3000):
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=nfeatures)
        if hasattr(cv2, "xfeatures2d"):
            return cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
        raise RuntimeError("SIFT not available. Install opencv-contrib-python.")

    def detect_and_compute_all(self):
        sift = self.get_sift()
        self.keypoints = []
        self.descriptors = []
        for i, img in enumerate(self.images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            self.keypoints.append(kp)
            self.descriptors.append(des)
            print(f"[features] image {i}: {len(kp)} keypoints")

    def match_flann(self, des1, des2, ratio_thresh=0.75):
        if des1 is None or des2 is None:
            return []
        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
        if des2.dtype != np.float32:
            des2 = des2.astype(np.float32)
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

    def match_all_pairs(self, min_matches=8):
        n = len(self.images)
        matches_dict = {}
        for i in range(n):
            for j in range(i+1, n):
                if self.descriptors[i] is None or self.descriptors[j] is None:
                    continue
                matches = self.match_flann(self.descriptors[i], self.descriptors[j])
                if len(matches) >= min_matches:
                    matches_dict[(i, j)] = matches
                    print(f"[matching] ({i},{j}) -> {len(matches)} matches")
        return matches_dict

    # -------------------------
    # Track building
    # -------------------------
    def build_tracks(self, matches_dict):
        per_image_map = [dict() for _ in self.images]
        tracks = {}
        next_tid = 0

        for (i, j), matches in matches_dict.items():
            for m in matches:
                ki = m.queryIdx
                kj = m.trainIdx
                ti = per_image_map[i].get(ki, -1)
                tj = per_image_map[j].get(kj, -1)
                if ti == -1 and tj == -1:
                    tracks[next_tid] = [(i, ki), (j, kj)]
                    per_image_map[i][ki] = next_tid
                    per_image_map[j][kj] = next_tid
                    next_tid += 1
                elif ti != -1 and tj == -1:
                    tracks[ti].append((j, kj))
                    per_image_map[j][kj] = ti
                elif ti == -1 and tj != -1:
                    tracks[tj].append((i, ki))
                    per_image_map[i][ki] = tj
                elif ti != tj:
                    a, b = ti, tj
                    if len(tracks[a]) < len(tracks[b]):
                        a, b = b, a
                    for (img_idx, kp_idx) in tracks[b]:
                        tracks[a].append((img_idx, kp_idx))
                        per_image_map[img_idx][kp_idx] = a
                    del tracks[b]

        # filter tracks that appear in at least two different images
        tracks = {tid: obs for tid, obs in tracks.items() if len({p[0] for p in obs}) >= 2}
        print(f"[tracks] built {len(tracks)} tracks")
        return tracks

    # -------------------------
    # Two-view initialization
    # -------------------------
    def find_best_initial_pair(self, matches_dict):
        pairs = [(pair, len(matches)) for pair, matches in matches_dict.items()]
        if not pairs:
            raise RuntimeError("No image pair with enough matches found.")
        pairs.sort(key=lambda x: x[1], reverse=True)
        best_pair = pairs[0][0]
        print(f"[init] best initial pair: {best_pair} with {pairs[0][1]} matches")
        return best_pair

    def initialize_two_view(self, i, j, matches, ransac_thresh=1.0):
        pts1 = np.float32([self.keypoints[i][m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.keypoints[j][m.trainIdx].pt for m in matches])
        if len(pts1) < 8:
            raise RuntimeError("Not enough points for Essential matrix.")
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh)
        if E is None:
            raise RuntimeError("findEssentialMat failed.")
        mask = mask.ravel().astype(bool)
        print(f"[init] essential inliers: {int(mask.sum())}/{len(matches)}")
        # Use recoverPose for a robust R,t
        mask_in = mask.astype(np.uint8).reshape(-1, 1)
        _, R, t, mask2 = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask_in)
        # normalize translation scale to a reasonable baseline (heuristic)
        if np.linalg.norm(t) > 1e-8:
            t = t / np.linalg.norm(t) * 5.0
        P0 = self.K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P1 = self.K @ np.hstack((R, t.reshape(3,1)))
        pts4d = cv2.triangulatePoints(P0, P1, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T
        # keep points in front of camera 0 and camera 1
        pts_cam1 = (R.dot(pts3d.T) + t.reshape(3,1)).T
        mask_front = (pts3d[:,2] > 0) & (pts_cam1[:,2] > 0)
        pts3d_valid = pts3d[mask_front]
        print(f"[init] triangulated valid: {len(pts3d_valid)}/{len(pts3d)}")
        return (np.eye(3), np.zeros((3,1))), (R, t.reshape(3,1)), pts3d_valid, mask_front

    # -------------------------
    # Insert initial points into map
    # -------------------------
    def initialize_map_from_two_view(self, i0, i1, matches_pair, pts3d_valid, mask_front):
        # build mapping from (img,kp) to track id
        obs_map = {}
        for tid, obs in self.tracks.items():
            for (img_idx, kp_idx) in obs:
                obs_map[(img_idx, kp_idx)] = tid

        # choose inlier matches consistent with mask_front
        pts1 = [self.keypoints[i0][m.queryIdx].pt for m in matches_pair]
        pts2 = [self.keypoints[i1][m.trainIdx].pt for m in matches_pair]
        idx_map = []
        for idx_m, m in enumerate(matches_pair):
            if idx_m < len(mask_front) and mask_front[idx_m]:
                idx_map.append((m.queryIdx, m.trainIdx))
        # assign triangulated points to tracks where possible
        used = 0
        for k in range(len(pts3d_valid)):
            if k >= len(idx_map):
                break
            qidx, tidx = idx_map[k]
            tid = obs_map.get((i0, qidx)) or obs_map.get((i1, tidx))
            if tid is None:
                continue
            self.points3D[tid] = pts3d_valid[k]
            # sample color from first image
            x, y = int(round(pts1[k][0])), int(round(pts1[k][1]))
            img_rgb = cv2.cvtColor(self.images[i0], cv2.COLOR_BGR2RGB)
            x = np.clip(x, 0, img_rgb.shape[1]-1)
            y = np.clip(y, 0, img_rgb.shape[0]-1)
            self.colors[tid] = img_rgb[y, x]
            used += 1
        print(f"[map init] assigned {used} initial 3D points to tracks")

    # -------------------------
    # Incremental registration (PnP)
    # -------------------------
    def solve_pnp_for_image(self, img_idx, min_points=6):
        obj_pts = []
        img_pts = []
        for tid, obs in self.tracks.items():
            if tid not in self.points3D:
                continue
            for (img_i, kp_idx) in obs:
                if img_i == img_idx:
                    obj_pts.append(self.points3D[tid])
                    img_pts.append(self.keypoints[img_i][kp_idx].pt)
                    break
        if len(obj_pts) < min_points:
            return None
        obj_pts = np.asarray(obj_pts, dtype=np.float32)
        img_pts = np.asarray(img_pts, dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, None,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3,1)
        return (R, t)

    def triangulate_new_points_for_image(self, new_img_idx):
        P_new = self.K @ np.hstack((self.cameras[new_img_idx][0], self.cameras[new_img_idx][1]))
        new_points = 0
        for tid, obs in self.tracks.items():
            if tid in self.points3D:
                continue
            obs_map = {img_i: kp_idx for (img_i, kp_idx) in obs}
            if new_img_idx not in obs_map:
                continue
            # pick an existing registered image that sees this track
            common = set(obs_map.keys()).intersection(set(self.cameras.keys()))
            if not common:
                continue
            other_idx = next(iter(common))
            P_other = self.K @ np.hstack((self.cameras[other_idx][0], self.cameras[other_idx][1]))
            pt_new = np.array(self.keypoints[new_img_idx][obs_map[new_img_idx]].pt, dtype=np.float32).reshape(2,1)
            pt_other = np.array(self.keypoints[other_idx][obs_map[other_idx]].pt, dtype=np.float32).reshape(2,1)
            pts4d = cv2.triangulatePoints(P_other, P_new, pt_other, pt_new)
            if np.abs(pts4d[3]) < 1e-8:
                continue
            p3d = (pts4d[:3] / pts4d[3]).ravel()
            # only accept points in front of both cameras
            if p3d[2] <= 0:
                continue
            self.points3D[tid] = p3d
            # color sampling from other image
            img_rgb = cv2.cvtColor(self.images[other_idx], cv2.COLOR_BGR2RGB)
            x, y = self.keypoints[other_idx][obs_map[other_idx]].pt
            x = np.clip(int(round(x)), 0, img_rgb.shape[1]-1)
            y = np.clip(int(round(y)), 0, img_rgb.shape[0]-1)
            self.colors[tid] = img_rgb[y,x]
            new_points += 1
        return new_points

    # -------------------------
    # Normalization
    # -------------------------
    def normalize_reconstruction(self):
        if len(self.points3D) < 5:
            return
        all_pts = np.array(list(self.points3D.values()))
        centroid = np.mean(all_pts, axis=0)
        scale = np.std(all_pts - centroid)
        if scale <= 0:
            return
        if scale > 50.0 or scale < 0.05:
            scale_factor = 10.0 / scale if scale > 50.0 else 1.0 / scale
            for tid in list(self.points3D.keys()):
                self.points3D[tid] = (self.points3D[tid] - centroid) * scale_factor
            for img_idx, (R, t) in list(self.cameras.items()):
                new_t = (t - R @ centroid.reshape(3,1)) * scale_factor
                self.cameras[img_idx] = (R, new_t)
            print(f"[normalize] applied scale factor {scale_factor:.6f}")

    # -------------------------
    # Fast bundle adjustment (subset)
    # -------------------------
    def run_fast_bundle_adjustment(self, max_points=200):
        print("[BA] collecting observations")
        point_visibility = {}
        for tid, obs in self.tracks.items():
            if tid in self.points3D:
                point_visibility[tid] = len(obs)
        well_obs = sorted(point_visibility.items(), key=lambda x: x[1], reverse=True)[:max_points]
        if not well_obs:
            print("[BA] no well-observed points, skipping BA")
            return self.cameras, self.points3D
        points_subset = {tid: self.points3D[tid] for tid, _ in well_obs}
        # collect observations
        observations = []
        for tid in points_subset:
            for img_i, kp_idx in self.tracks[tid]:
                if img_i in self.cameras:
                    uv = self.keypoints[img_i][kp_idx].pt
                    observations.append((img_i, tid, np.array(uv, dtype=np.float32)))
        cam_idx_map = {img_idx: i for i, img_idx in enumerate(sorted(self.cameras.keys()))}
        pt_idx_map = {tid: i for i, tid in enumerate(sorted(points_subset.keys()))}
        n_cams = len(cam_idx_map)
        n_pts = len(pt_idx_map)
        if len(observations) < n_cams * 6:
            print("[BA] insufficient observations for BA, skipping")
            return self.cameras, self.points3D
        cam_params = np.zeros((n_cams, 6))
        pts = np.zeros((n_pts, 3))
        for img_idx, (R, t) in self.cameras.items():
            ci = cam_idx_map[img_idx]
            rvec, _ = cv2.Rodrigues(R)
            cam_params[ci, :3] = rvec.ravel()
            cam_params[ci, 3:] = t.ravel()
        for tid, p3d in points_subset.items():
            pi = pt_idx_map[tid]
            pts[pi] = p3d
        x0 = np.hstack((cam_params.ravel(), pts.ravel()))
        def residuals(x):
            cam_flat = x[:n_cams*6].reshape((n_cams,6))
            pts_flat = x[n_cams*6:].reshape((n_pts,3))
            res = []
            for img_idx, tid, uv in observations:
                ci = cam_idx_map[img_idx]
                pi = pt_idx_map[tid]
                rvec = cam_flat[ci,:3]
                tvec = cam_flat[ci,3:]
                R, _ = cv2.Rodrigues(rvec.reshape(3,1))
                p3d = pts_flat[pi]
                proj = self.K @ (R @ p3d.reshape(3,1) + tvec.reshape(3,1))
                if proj[2] == 0:
                    continue
                proj2 = (proj[:2] / proj[2]).ravel()
                res.extend((proj2 - uv).tolist())
            return np.array(res)
        try:
            initial_rms = np.sqrt(np.mean(residuals(x0)**2))
        except Exception:
            initial_rms = float('nan')
        print(f"[BA] initial RMS (approx): {initial_rms:.3f}")
        try:
            res = least_squares(residuals, x0, method='dogbox', verbose=0, max_nfev=100, ftol=1e-4, xtol=1e-4)
            cam_opt = res.x[:n_cams*6].reshape((n_cams,6))
            pts_opt = res.x[n_cams*6:].reshape((n_pts,3))
            # apply camera updates
            cameras_opt = {}
            for img_idx, ci in cam_idx_map.items():
                rvec = cam_opt[ci,:3]
                tvec = cam_opt[ci,3:]
                R, _ = cv2.Rodrigues(rvec.reshape(3,1))
                cameras_opt[img_idx] = (R, tvec.reshape(3,1))
            # update points3D only for subset (others keep original)
            points3D_new = dict(self.points3D)
            for tid, pi in pt_idx_map.items():
                points3D_new[tid] = pts_opt[pi]
            print("[BA] finished successfully")
            return cameras_opt, points3D_new
        except Exception as e:
            print(f"[BA] failed: {e}")
            return self.cameras, self.points3D

    # -------------------------
    # Save utilities
    # -------------------------
    def save_results(self, filename=None):
        if filename is None:
            filename = os.path.join(self.outputs_dir, "week3_after_ba.ply")
        # assemble points + camera centers
        pts = []
        cols = []
        for tid, p in sorted(self.points3D.items()):
            pts.append(p)
            col = self.colors.get(tid, np.array([255,255,255], dtype=np.uint8))
            cols.append(col)
        if len(pts) == 0:
            pts_arr = np.zeros((0,3), dtype=np.float32)
            cols_arr = None
        else:
            pts_arr = np.vstack(pts)
            cols_arr = np.vstack(cols)
        # append camera centers as colored points (red)
        cam_centers = []
        cam_cols = []
        for img_idx, (R, t) in sorted(self.cameras.items()):
            center = (-R.T @ t).ravel()
            cam_centers.append(center)
            cam_cols.append(np.array([255,0,0], dtype=np.uint8))
        if cam_centers:
            cam_arr = np.vstack(cam_centers)
            cam_cols_arr = np.vstack(cam_cols)
            if pts_arr.shape[0] == 0:
                final_pts = cam_arr
                final_cols = cam_cols_arr
            else:
                final_pts = np.vstack([pts_arr, cam_arr])
                final_cols = np.vstack([cols_arr, cam_cols_arr])
        else:
            final_pts = pts_arr
            final_cols = cols_arr
        write_ply_ascii(filename, final_pts, final_cols)
        print(f"[save] wrote PLY to: {filename}")

    def save_camera_poses(self, filename=None):
        if filename is None:
            filename = os.path.join(self.outputs_dir, "week3_cameras_after_ba.json")
        camera_data = {}
        for img_idx, (R, t) in sorted(self.cameras.items()):
            camera_data[int(img_idx)] = {
                'R': np.array(R).tolist(),
                't': np.array(t).ravel().tolist(),
                'center': (-np.array(R).T @ np.array(t)).ravel().tolist()
            }
        with open(filename, 'w') as f:
            json.dump(camera_data, f, indent=2)
        print(f"[save] saved camera poses to: {filename}")

    # -------------------------
    # Full pipeline runner
    # -------------------------
    def run_week3_pipeline(self, data_dir):
        print("="*40)
        print("WEEK 3: Multi-View SfM with Refinement")
        print("="*40)
        # load images
        filelist = sorted([f for f in os.listdir(data_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
        if len(filelist) < 2:
            raise RuntimeError("Need at least 2 images in data_dir")
        self.image_paths = [os.path.join(data_dir, f) for f in filelist]
        print(f"[io] loading {len(self.image_paths)} images")
        self.images = [self.read_image(p) for p in self.image_paths]

        # camera intrinsics: focal = focal_length, principal point = image center
        h, w = self.images[0].shape[:2]
        fx = fy = float(self.focal_length)
        cx = w / 2.0
        cy = h / 2.0
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        print(f"[K] set camera intrinsics:\n{self.K}")

        # features + matches + tracks
        self.detect_and_compute_all()
        matches_dict = self.match_all_pairs(min_matches=8)
        if not matches_dict:
            raise RuntimeError("No pairwise matches found. Check images or lower min_matches.")
        self.tracks = self.build_tracks(matches_dict)

        # two-view initialization
        i0, i1 = self.find_best_initial_pair(matches_dict)
        matches_pair = matches_dict[(i0, i1)]
        pose0, pose1, pts3d_valid, mask_front = self.initialize_two_view(i0, i1, matches_pair)
        self.cameras[i0] = pose0
        self.cameras[i1] = pose1
        # insert initial 3D points into map
        self.initialize_map_from_two_view(i0, i1, matches_pair, pts3d_valid, mask_front)

        print(f"[init] cameras: {len(self.cameras)} points: {len(self.points3D)}")

        # save initial reconstruction
        self.save_results(os.path.join(self.outputs_dir, "week3_initial.ply"))
        self.save_camera_poses(os.path.join(self.outputs_dir, "week3_cameras_before_ba.json"))

        # incremental registration
        remaining = [i for i in range(len(self.images)) if i not in (i0, i1)]
        for idx in remaining:
            print(f"[inc] processing image {idx}")
            pose = self.solve_pnp_for_image(idx, min_points=6)
            if pose is None:
                print(f"[inc] skipped {idx} - insufficient 2D-3D correspondences")
                continue
            self.cameras[idx] = pose
            new_pts = self.triangulate_new_points_for_image(idx)
            print(f"[inc] registered {idx}, +{new_pts} new points, total points now {len(self.points3D)}")
            if len(self.cameras) % 4 == 0 and len(self.points3D) > 50:
                self.normalize_reconstruction()

        print(f"[inc] registration complete: cameras {len(self.cameras)}/{len(self.images)}, points {len(self.points3D)}")

        # normalization
        self.normalize_reconstruction()
        self.save_results(os.path.join(self.outputs_dir, "week3_before_ba.ply"))
        self.save_camera_poses(os.path.join(self.outputs_dir, "week3_cameras_before_ba.json"))

        # bundle adjustment refinement
        cams_ba, pts_ba = self.run_fast_bundle_adjustment(max_points=200)
        self.cameras = cams_ba
        self.points3D = pts_ba

        # save final outputs into outputs folder
        self.save_results(os.path.join(self.outputs_dir, "week3_after_ba.ply"))
        self.save_camera_poses(os.path.join(self.outputs_dir, "week3_cameras_after_ba.json"))

        print("="*40)
        print("WEEK 3 PIPELINE FINISHED")
        print(f"Registered cameras: {len(self.cameras)}/{len(self.images)}")
        print(f"3D points: {len(self.points3D)}")
        print("Files written to outputs/")
        print("="*40)
        return self.cameras, self.points3D

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Week 3: Multi-View SfM with Refinement")
    parser.add_argument("--data_dir", type=str, default="./Data", help="Directory containing images")
    parser.add_argument("--focal", type=float, default=800.0, help="Focal length (fx=fy)")
    parser.add_argument("--resize_max", type=int, default=1200, help="Maximum image dimension")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Directory for outputs (must exist or will be created)")
    args = parser.parse_args()

    pipeline = SFMPipeline(focal_length=args.focal, max_image_dim=args.resize_max, outputs_dir=args.outputs_dir)
    try:
        pipeline.run_week3_pipeline(args.data_dir)
    except Exception as e:
        print("Pipeline failed:")
        print(str(e))
        traceback.print_exc()

if __name__ == "__main__":
    main()
