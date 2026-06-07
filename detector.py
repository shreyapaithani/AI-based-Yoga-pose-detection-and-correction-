import cv2
import mediapipe as mp
import numpy as np
import math
import pickle
from poses import get_pose

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


with open("model/yoga_model.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open("model/label_encoder.pkl", "rb") as f:
    ENCODER = pickle.load(f)

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
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    cx = np.mean(xs)
    cy = np.mean(ys)
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    for name in LANDMARK_NAMES:
        features.append((pts[name][0] - cx) / scale)
        features.append((pts[name][1] - cy) / scale)

    for a_name, b_name, c_name in ANGLE_TRIPLETS:
        features.append(angle_between(pts[a_name], pts[b_name], pts[c_name]))

    return np.array(features).reshape(1, -1)

def compute_angles(landmarks, h, w):
    """Compute key angles for rule based classification"""
    def pt(name):
        idx = mp_pose.PoseLandmark[name].value
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    angles = {}
    try:
        angles["left_knee"]    = angle_between(pt("LEFT_HIP"),      pt("LEFT_KNEE"),   pt("LEFT_ANKLE"))
        angles["right_knee"]   = angle_between(pt("RIGHT_HIP"),     pt("RIGHT_KNEE"),  pt("RIGHT_ANKLE"))
        angles["left_elbow"]   = angle_between(pt("LEFT_SHOULDER"), pt("LEFT_ELBOW"),  pt("LEFT_WRIST"))
        angles["right_elbow"]  = angle_between(pt("RIGHT_SHOULDER"),pt("RIGHT_ELBOW"), pt("RIGHT_WRIST"))
        angles["left_hip"]     = angle_between(pt("LEFT_SHOULDER"), pt("LEFT_HIP"),    pt("LEFT_KNEE"))
        angles["right_hip"]    = angle_between(pt("RIGHT_SHOULDER"),pt("RIGHT_HIP"),   pt("RIGHT_KNEE"))

        ls = pt("LEFT_SHOULDER");  rs = pt("RIGHT_SHOULDER")
        lh = pt("LEFT_HIP");       rh = pt("RIGHT_HIP")
        la = pt("LEFT_ANKLE");     ra = pt("RIGHT_ANKLE")
        lw = pt("LEFT_WRIST");     rw = pt("RIGHT_WRIST")
        lk = pt("LEFT_KNEE");      rk = pt("RIGHT_KNEE")

        ms = (ls + rs) / 2
        mh = (lh + rh) / 2
        ma = (la + ra) / 2

        dy = mh[1] - ms[1]
        dx = mh[0] - ms[0]
        angles["torso_lean"] = abs(math.degrees(math.atan2(dx, dy + 1e-6)))

        angles["hip_spread"]    = abs(lh[0] - rh[0]) / (w + 1e-6)
        angles["arm_spread"]    = abs(lw[0] - rw[0]) / (w + 1e-6)
        angles["hip_height"]    = 1.0 - (mh[1] / (h + 1e-6))
        angles["wrist_height"]  = 1.0 - ((lw[1] + rw[1]) / 2 / (h + 1e-6))
        angles["knee_spread"]   = abs(lk[0] - rk[0]) / (w + 1e-6)
        angles["ankle_spread"]  = abs(la[0] - ra[0]) / (w + 1e-6)
        angles["leg_asym"]      = abs(la[1] - ra[1]) / (h + 1e-6)
        angles["hip_asym"]      = abs(lh[1] - rh[1]) / (h + 1e-6)
        angles["shoulder_asym"] = abs(ls[1] - rs[1]) / (h + 1e-6)
        angles["knee_height"]   = 1.0 - ((lk[1] + rk[1]) / 2 / (h + 1e-6))
    except Exception:
        pass
    return angles

def rule_based_classify(angles):
    """Strong rule based classifier using joint angles"""
    lk  = angles.get("left_knee", 180)
    rk  = angles.get("right_knee", 180)
    le  = angles.get("left_elbow", 180)
    re  = angles.get("right_elbow", 180)
    lh  = angles.get("left_hip", 180)
    rh  = angles.get("right_hip", 180)
    tl  = angles.get("torso_lean", 0)
    hs  = angles.get("hip_spread", 0)
    as_ = angles.get("arm_spread", 0)
    hh  = angles.get("hip_height", 0)
    wh  = angles.get("wrist_height", 0)
    la  = angles.get("leg_asym", 0)
    ha  = angles.get("hip_asym", 0)
    sa  = angles.get("shoulder_asym", 0)
    kh  = angles.get("knee_height", 0)
    ks  = angles.get("knee_spread", 0)
    ans = angles.get("ankle_spread", 0)

    avg_knee = (lk + rk) / 2

    # Downward Dog — hips very high, torso angled
    if hh > 0.55 and tl > 45 and avg_knee > 140:
        return "adho mukha svanasana", 0.90

    # Plank — body flat horizontal, arms straight
    if 55 < tl < 100 and avg_knee > 155 and le > 150 and re > 150 and hh < 0.55:
        return "phalakasana", 0.88

    # Warrior II — wide stance, arms spread wide, knee bent
    if hs > 0.25 and as_ > 0.45 and avg_knee < 140:
        return "virabhadrasana ii", 0.87

    # Warrior I — wide stance, arms up high, knee bent
    if hs > 0.22 and wh > 0.65 and avg_knee < 140 and as_ < 0.35:
        return "virabhadrasana i", 0.86

    # Warrior III — torso horizontal, one leg raised
    if tl > 55 and la > 0.28 and avg_knee > 150:
        return "virabhadrasana iii", 0.85

    # Tree pose — one leg raised, standing straight
    if ha > 0.15 and tl < 20 and avg_knee > 150:
        return "vriksasana", 0.88

    # Chair pose — knees bent, arms raised
    if 85 < avg_knee < 145 and wh > 0.60 and tl < 35:
        return "utkatasana", 0.87

    # Triangle — wide stance, torso tilted sideways
    if hs > 0.28 and sa > 0.12 and tl < 35 and as_ > 0.30:
        return "utthita trikonasana", 0.85

    # Cobra — torso leaning back, lying down
    if tl > 35 and hh < 0.38 and lh < 130 and rh < 130:
        return "bhujangasana", 0.86

    # Child pose — deep knee bend, torso folded forward
    if avg_knee < 70 and tl > 55:
        return "balasana", 0.87

    # Bridge — hips very high, knees bent
    if hh > 0.52 and avg_knee < 130 and tl > 50:
        return "setu bandha sarvangasana", 0.86

    # Boat pose — sitting, legs and torso raised
    if kh > 0.45 and hh > 0.40 and tl < 45 and avg_knee > 110:
        return "paripurna navasana", 0.84

    # Seated forward bend — sitting, torso folded over legs
    if tl > 50 and hh < 0.45 and avg_knee > 140:
        return "paschimottanasana", 0.83

    # Standing forward bend — standing, torso folded down
    if tl > 60 and hh > 0.45 and avg_knee > 140:
        return "uttanasana", 0.84

    # Camel — kneeling, arching back
    if tl > 25 and hh < 0.45 and avg_knee < 100 and wh < 0.45:
        return "ustrasana", 0.82

    # Half moon — one leg raised, body sideways
    if la > 0.32 and as_ > 0.35 and tl > 45:
        return "ardha chandrasana", 0.83

    # Low lunge — deep lunge, one knee down
    if la > 0.20 and avg_knee < 115 and tl < 35:
        return "anjaneyasana", 0.82

    # Upward dog — arms straight, chest lifted, thighs off ground
    if tl > 30 and hh < 0.40 and le > 155 and re > 155:
        return "urdhva mukha svanasana", 0.83

    # Eagle — arms crossed, one leg wrapped
    if as_ < 0.12 and ha > 0.08 and avg_knee < 155:
        return "garudasana", 0.81

    # Wide leg forward fold — very wide stance, folded
    if ans > 0.40 and tl > 50:
        return "prasarita padottanasana", 0.82

    # Extended side angle — wide stance, one arm down one up
    if hs > 0.25 and tl > 18 and as_ > 0.32 and avg_knee < 140:
        return "utthita parsvakonasana", 0.81

    # Pyramid — medium stance, folded forward
    if 0.12 < ans < 0.30 and tl > 45 and avg_knee > 145:
        return "parsvottanasana", 0.80

    # Mountain pose — standing straight, arms at sides
    if tl < 18 and avg_knee > 160 and as_ < 0.25 and wh < 0.55:
        return "tadasana", 0.85

    # Upward salute — standing, arms raised straight up
    if tl < 20 and wh > 0.70 and avg_knee > 158:
        return "urdhva hastasana", 0.84

    return None, 0.0

def predict_pose(features):
    proba = MODEL.predict_proba(features)[0]
    idx = np.argmax(proba)
    label = ENCODER.classes_[idx]
    confidence = proba[idx]
    return label, confidence

class YogaDetector:
    def __init__(self):
        self.pose_live = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_static = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

    def _classify(self, landmarks, h, w):
        # Step 1 — rule based (fast and accurate)
        angles = compute_angles(landmarks, h, w)
        rule_label, rule_conf = rule_based_classify(angles)

        # Step 2 — ML model
        features = extract_features(landmarks, h, w)
        ml_label, ml_conf = predict_pose(features)

        # Use rule based if confident, else use ML model
        if rule_conf >= 0.82:
            return rule_label, rule_conf
        elif ml_conf >= 0.70:
            return ml_label, ml_conf
        elif rule_label is not None:
            return rule_label, rule_conf
        else:
            return ml_label, ml_conf

    def _draw_and_predict(self, frame, landmarks, h, w):
        mp_drawing.draw_landmarks(
            frame, landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_styles.get_default_pose_landmarks_style()
        )
        label, confidence = self._classify(landmarks.landmark, h, w)
        return label, confidence

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, None, 0.0
        h, w = img.shape[:2]
        results = self.pose_static.process(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return img, None, 0.0
        label, conf = self._draw_and_predict(
            img, results.pose_landmarks, h, w)
        pose_data = get_pose(label) if label else get_pose("default")
        cv2.putText(img,
            f"{pose_data['name']}  {conf*100:.0f}%",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (50, 220, 120), 2, cv2.LINE_AA)
        return img, label, conf

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        results = self.pose_live.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return frame, None, 0.0
        label, conf = self._draw_and_predict(
            frame, results.pose_landmarks, h, w)
        pose_data = get_pose(label) if label else get_pose("default")
        cv2.putText(frame,
            f"{pose_data['name']}  {conf*100:.0f}%",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (50, 220, 120), 2, cv2.LINE_AA)
        return frame, label, conf
