import os
import csv
import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose

ANGLE_TRIPLETS = [
    ("LEFT_SHOULDER",  "LEFT_ELBOW",    "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW",   "RIGHT_WRIST"),
    ("LEFT_HIP",       "LEFT_SHOULDER", "LEFT_ELBOW"),
    ("RIGHT_HIP",      "RIGHT_SHOULDER","RIGHT_ELBOW"),
    ("LEFT_HIP",       "LEFT_KNEE",     "LEFT_ANKLE"),
    ("RIGHT_HIP",      "RIGHT_KNEE",    "RIGHT_ANKLE"),
    ("LEFT_SHOULDER",  "LEFT_HIP",      "LEFT_KNEE"),
    ("RIGHT_SHOULDER", "RIGHT_HIP",     "RIGHT_KNEE"),
    ("LEFT_HIP",       "RIGHT_HIP",     "RIGHT_KNEE"),
    ("LEFT_SHOULDER",  "RIGHT_SHOULDER","RIGHT_HIP"),
    ("LEFT_KNEE",      "LEFT_HIP",      "RIGHT_HIP"),
    ("RIGHT_KNEE",     "RIGHT_HIP",     "LEFT_HIP"),
]

LANDMARK_NAMES = [e.name for e in mp_pose.PoseLandmark]

def angle_between(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))

def extract_features(landmarks, h, w):
    pts = {}
    for name in LANDMARK_NAMES:
        idx = mp_pose.PoseLandmark[name].value
        lm = landmarks[idx]
        pts[name] = [lm.x * w, lm.y * h]

    features = []

    # 1. Normalised x,y coordinates for all 33 landmarks = 66 values
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    cx = np.mean(xs)
    cy = np.mean(ys)
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    for name in LANDMARK_NAMES:
        features.append((pts[name][0] - cx) / scale)
        features.append((pts[name][1] - cy) / scale)

    # 2. Joint angles = 12 values
    for a_name, b_name, c_name in ANGLE_TRIPLETS:
        ang = angle_between(pts[a_name], pts[b_name], pts[c_name])
        features.append(ang)

    return features  # total 78 features

def process_dataset(dataset_dir, output_csv):
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    )

    # Build CSV header
    header = ["label"]
    for name in LANDMARK_NAMES:
        header += [f"{name}_x", f"{name}_y"]
    for a, b, c in ANGLE_TRIPLETS:
        header.append(f"angle_{a}_{b}_{c}")

    rows = []
    class_folders = [f for f in os.listdir(dataset_dir)
                     if os.path.isdir(os.path.join(dataset_dir, f))]

    print(f"Found {len(class_folders)} pose classes")

    for label in class_folders:
        label_dir = os.path.join(dataset_dir, label)
        images = [f for f in os.listdir(label_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  Processing '{label}': {len(images)} images...")

        success = 0
        for img_file in images:
            path = os.path.join(label_dir, img_file)
            img = cv2.imread(path)
            if img is None:
                continue

            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(img_rgb)

            if not results.pose_landmarks:
                continue

            # Skip low confidence detections
            scores = [lm.visibility for lm in results.pose_landmarks.landmark]
            if np.mean(scores) < 0.5:
                continue

            features = extract_features(results.pose_landmarks.landmark, h, w)
            rows.append([label] + features)
            success += 1

        print(f"    Saved {success}/{len(images)} images successfully")

    pose_detector.close()

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nDone! Saved {len(rows)} rows to '{output_csv}'")

if __name__ == "__main__":
    process_dataset(
        dataset_dir="dataset/dataset",
        output_csv="keypoints.csv"
    )


