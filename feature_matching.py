# Week 1: Setup & Feature Matching
# This script sets up the project environment, loads two images,
# detects SIFT features, matches them with Lowe's ratio test,
# and visualizes the filtered matches.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def get_sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    return cv2.xfeatures2d.SIFT_create()


def detect_and_compute(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = get_sift()
    kps, des = sift.detectAndCompute(gray, None)
    return kps, des


def match_features(des1, des2, ratio=0.75):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


def draw_matches(img1, kp1, img2, kp2, matches, max_draw=200):
    matched = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:max_draw],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.figure(figsize=(14, 7))
    plt.imshow(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Default arguments so script can run without CLI args
    img1_path = "./Data/0001.jpeg"
    img2_path = "./Data/0002.jpeg"

    img1 = read_image(img1_path)
    img2 = read_image(img2_path)

    kp1, des1 = detect_and_compute(img1)
    kp2, des2 = detect_and_compute(img2)

    print("Detected keypoints:", len(kp1), len(kp2))

    matches = match_features(des1, des2, ratio=0.75)
    print("Good matches:", len(matches))

    draw_matches(img1, kp1, img2, kp2, matches)