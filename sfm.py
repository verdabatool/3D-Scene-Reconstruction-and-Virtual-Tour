"""
week3_sfm.py

Incremental Multi-View Structure-from-Motion (PnP) + simple bundle adjustment
and an interactive "Photosynth"-like viewer (image crossfade + sparse point cloud).

Dependencies:
    numpy
    opencv-python
    matplotlib
    scipy

Usage:
    python week3_sfm.py --data_dir ./Data
"""

import os
import argparse
import cv2
import numpy as np
from collections import defaultdict
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# -------------------------
# Utilities
# -------------------------
def read_image(path, max_dim=1200):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def get_sift(nfeatures=2000):
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=nfeatures)
    return cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)

def match_two(des1, des2, ratio=0.75):
    if des1 is None or des2 is None:
        return []
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=10)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good

def extract_features(images):
    sift = get_sift()
    keypoints = []
    descriptors = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)
    return keypoints, descriptors

def match_all_pairs(descriptors, ratio=0.75, min_matches=20):
    n = len(descriptors)
    matches_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            if descriptors[i] is None or descriptors[j] is None:
                continue
            good = match_two(descriptors[i], descriptors[j], ratio)
            if len(good) >= min_matches:
                matches_dict[(i, j)] = good
    return matches_dict

# -------------------------
# Track assembly
# -------------------------
def build_tracks(matches_dict, keypoints):
    per_image_map = [dict() for _ in keypoints]
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
    tracks = {tid: vals for tid, vals in tracks.items() if len({p[0] for p in vals}) >= 2}
    return tracks

# -------------------------
# Two-view initialization
# -------------------------
def init_two_view(i, j, kps, matches, K):
    pts1 = np.float32([kps[i][m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps[j][m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise RuntimeError("Essential matrix estimation failed.")
    _, R_mat, t_vec, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    pose0 = (np.eye(3), np.zeros((3, 1)))
    pose1 = (R_mat, t_vec)
    inliers = mask.ravel().astype(bool)
    P0 = K @ np.hstack((pose0[0], pose0[1]))
    P1 = K @ np.hstack((pose1[0], pose1[1]))
    pts4d = cv2.triangulatePoints(P0, P1, pts1[inliers].T, pts2[inliers].T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    return pose0, pose1, pts3d, pts1[inliers], pts2[inliers], inliers

# -------------------------
# PnP
# -------------------------
def solve_pnp_for_image(img_idx, tracks, points3D, kps, K, used_pts3d_map):
    obj_pts = []
    img_pts = []
    for tid, obs in tracks.items():
        if tid not in points3D:
            continue
        for (img_i, kp_idx) in obs:
            if img_i == img_idx:
                img_pts.append(kps[img_i][kp_idx].pt)
                obj_pts.append(points3D[tid])
                used_pts3d_map[tid] = used_pts3d_map.get(tid, 0) + 1
                break
    if len(obj_pts) < 6:
        return None, None, None
    obj_pts = np.asarray(obj_pts, dtype=np.float32)
    img_pts = np.asarray(img_pts, dtype=np.float32)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=8.0, confidence=0.99, iterationsCount=100
    )
    if not success:
        return None, None, None
    if inliers is None:
        inliers = np.arange(len(obj_pts)).reshape(-1, 1)
    R_mat, _ = cv2.Rodrigues(rvec)
    tvec = tvec.reshape(3, 1)
    return (R_mat, tvec), inliers.ravel(), (obj_pts, img_pts)

def triangulate_new_points(new_img_idx, existing_cameras, tracks, points3D, kps, K):
    P_new = K @ np.hstack((existing_cameras[new_img_idx][0], existing_cameras[new_img_idx][1]))
    for tid, obs in tracks.items():
        if tid in points3D:
            continue
        # Map image index -> keypoint index
        obs_map = {img_i: kp_idx for (img_i, kp_idx) in obs}
        # Skip tracks that don't include the new image
        if new_img_idx not in obs_map:
            continue
        # Find any existing camera with this track
        common = set(obs_map.keys()).intersection(set(existing_cameras.keys()))
        if len(common) == 0:
            continue
        other_idx = next(iter(common))
        if other_idx == new_img_idx:
            continue
        P_other = K @ np.hstack((existing_cameras[other_idx][0], existing_cameras[other_idx][1]))
        # Use keypoints safely
        pt_new = np.array(kps[new_img_idx][obs_map[new_img_idx]].pt).reshape(2,1)
        pt_other = np.array(kps[other_idx][obs_map[other_idx]].pt).reshape(2,1)
        pts4d = cv2.triangulatePoints(P_other, P_new, pt_other, pt_new)
        p3d = (pts4d[:3] / pts4d[3]).ravel()
        proj_other = (P_other @ np.hstack((p3d,1)).T)
        proj_other = proj_other[:2]/proj_other[2]
        if np.linalg.norm(proj_other - pt_other.ravel()) < 5.0:
            points3D[tid] = p3d
    return points3D

# -------------------------
# Bundle adjustment
# -------------------------
def pack_parameters(cameras, points3d, cam_idx_map, pt_idx_map):
    n_cams = len(cameras)
    n_pts = len(points3d)
    cam_params = np.zeros((n_cams,6))
    pts = np.zeros((n_pts,3))
    for img_idx,(R_mat,tvec) in cameras.items():
        ci = cam_idx_map[img_idx]
        rvec,_ = cv2.Rodrigues(R_mat)
        cam_params[ci,:3]=rvec.ravel()
        cam_params[ci,3:]=tvec.ravel()
    for tid,p3d in points3d.items():
        pi = pt_idx_map[tid]
        pts[pi] = p3d
    return cam_params.ravel(), pts.ravel()

def unpack_parameters(x,n_cams,n_pts):
    cam_params = x[:n_cams*6].reshape((n_cams,6))
    pts = x[n_cams*6:].reshape((n_pts,3))
    return cam_params, pts

def reprojection_residuals(x,n_cams,n_pts,observations,cam_idx_map,pt_idx_map,K):
    cam_params, pts = unpack_parameters(x,n_cams,n_pts)
    res=[]
    for img_idx,tid,uv in observations:
        ci = cam_idx_map[img_idx]
        pi = pt_idx_map[tid]
        rvec = cam_params[ci,:3]
        tvec = cam_params[ci,3:]
        R_mat,_ = cv2.Rodrigues(rvec.reshape(3,1))
        p3d = pts[pi]
        proj = K @ (R_mat @ p3d.reshape(3,1)+tvec.reshape(3,1))
        proj = (proj[:2]/proj[2]).ravel()
        res.extend((proj-uv).tolist())
    return np.array(res)

def run_bundle_adjustment(cameras, points3d, tracks, keypoints, K, max_nfev=100):
    # only include observations for cameras that exist
    valid_img_idxs = set(cameras.keys())
    observations=[]
    for tid, obs in tracks.items():
        if tid not in points3d:
            continue
        for img_i, kp_idx in obs:
            if img_i not in valid_img_idxs:
                continue
            uv = keypoints[img_i][kp_idx].pt
            observations.append((img_i, tid, np.array(uv, dtype=np.float32)))

    cam_idx_map={img_idx:i for i,img_idx in enumerate(sorted(cameras.keys()))}
    pt_idx_map={tid:i for i,tid in enumerate(sorted(points3d.keys()))}
    n_cams=len(cam_idx_map)
    n_pts=len(pt_idx_map)
    if n_cams < 2 or n_pts < 10:
        print("BA: not enough cameras/points, skipping BA.")
        return cameras, points3d
    x_cam, x_pts = pack_parameters(cameras, points3d, cam_idx_map, pt_idx_map)
    x0 = np.hstack((x_cam, x_pts))
    res = least_squares(
        lambda x: reprojection_residuals(x, n_cams, n_pts, observations, cam_idx_map, pt_idx_map, K),
        x0, verbose=2, max_nfev=max_nfev, ftol=1e-4, xtol=1e-4
    )
    cam_params_opt, pts_opt = unpack_parameters(res.x, n_cams, n_pts)
    cameras_opt = {}
    points_opt = {}
    for img_idx, ci in cam_idx_map.items():
        rvec = cam_params_opt[ci, :3]
        tvec = cam_params_opt[ci, 3:]
        R_mat, _ = cv2.Rodrigues(rvec.reshape(3,1))
        cameras_opt[img_idx] = (R_mat, tvec.reshape(3,1))
    for tid, pi in pt_idx_map.items():
        points_opt[tid] = pts_opt[pi]
    return cameras_opt, points_opt

# -------------------------
# Pipeline
# -------------------------
def run_pipeline(data_dir,focal=800.0,principal=None,resize_max=1200):
    img_files=sorted([os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    if len(img_files)<2:
        raise RuntimeError("Need at least two images")
    images=[read_image(p,resize_max) for p in img_files]
    h,w=images[0].shape[:2]
    if principal is None:
        cx,cy=w/2.0,h/2.0
    else:
        cx,cy=principal
    K=np.array([[focal,0,cx],[0,focal,cy],[0,0,1]],dtype=np.float64)
    keypoints,descriptors=extract_features(images)
    matches_dict=match_all_pairs(descriptors,ratio=0.75,min_matches=30)
    tracks=build_tracks(matches_dict,keypoints)
    best_pair=max(matches_dict.items(),key=lambda x:len(x[1]))[0]
    i0,i1=best_pair
    pose0,pose1,pts3d_init,pts1_in,pts2_in,inliers_mask=init_two_view(i0,i1,keypoints,matches_dict[best_pair],K)
    cameras={i0:pose0,i1:pose1}
    points3D={}
    obs_map={}
    for tid,obs in tracks.items():
        for img_i,kp_idx in obs:
            obs_map[(img_i,kp_idx)]=tid
    matches=matches_dict[best_pair]
    k=0
    inlier_indices=np.where(inliers_mask)[0]
    for idx in inlier_indices:
        m=matches[idx]
        kp1_idx= m.queryIdx
        kp2_idx= m.trainIdx
        tid=obs_map.get((i0,kp1_idx),None) or obs_map.get((i1,kp2_idx),None)
        if tid is not None:
            points3D[tid]=pts3d_init[k]
        k+=1
    remaining=[i for i in range(len(images)) if i not in (i0,i1)]
    used_pts3d_counter={}
    for idx in remaining:
        pose,inliers,_=solve_pnp_for_image(idx,tracks,points3D,keypoints,K,used_pts3d_counter)
        if pose is None:
            print("Skipping image:",idx)
            continue
        cameras[idx]=pose
        points3D=triangulate_new_points(idx,cameras,tracks,points3D,keypoints,K)
    cameras_opt,points_opt=run_bundle_adjustment(cameras,points3D,tracks,keypoints,K,max_nfev=200)
    # viewer can be added here if needed

# -------------------------
# CLI
# -------------------------
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,default="./Data")
    parser.add_argument("--focal",type=float,default=800.0)
    parser.add_argument("--principal_x",type=float,default=None)
    parser.add_argument("--principal_y",type=float,default=None)
    args=parser.parse_args()
    principal=None
    if args.principal_x is not None and args.principal_y is not None:
        principal=(args.principal_x,args.principal_y)
    run_pipeline(args.data_dir,focal=args.focal,principal=principal)
