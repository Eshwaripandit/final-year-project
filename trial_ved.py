# single_video_preprocess.py

import cv2
import os
import numpy as np
import dlib
from imutils import face_utils

# 📌 MODIFY THESE
VIDEO_PATH = "D:/LipReadingProject/dataset/videos/s4/bbae9n.mpg"
LANDMARK_MODEL_PATH = "D:/LipReadingProject/model/shape_predictor_68_face_landmarks.dat"
OUTPUT_DIR = "D:/LipReadingProject/preprocessing/latest1"

# ✅ Load Dlib models
assert os.path.exists(LANDMARK_MODEL_PATH), "❌ Landmark model file not found!"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)

def preprocess_frame(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    landmarks = predictor(gray, faces[0])
    landmarks = face_utils.shape_to_np(landmarks)
    lips = landmarks[48:68]

    x_min, y_min = np.min(lips, axis=0)
    x_max, y_max = np.max(lips, axis=0)

    pad = 10
    x_min, x_max = max(0, x_min - pad), min(gray.shape[1], x_max + pad)
    y_min, y_max = max(0, y_min - pad), min(gray.shape[0], y_max + pad)

    lip_crop = gray[y_min:y_max, x_min:x_max]
    resized = cv2.resize(lip_crop, (128, 128), interpolation=cv2.INTER_CUBIC)
    normalized = resized.astype(np.float32) / 255.0

    return normalized

def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    preprocessed_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pre = preprocess_frame(frame)
        if pre is not None:
            preprocessed_frames.append(pre)
        frame_count += 1

    cap.release()

    if preprocessed_frames:
        arr = np.array(preprocessed_frames)[..., np.newaxis]  # (T, 128, 128, 1)
        os.makedirs(output_dir, exist_ok=True)

        # Dynamically generate .npy file name
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_basename}.npy")

        np.save(output_path, arr)
        print(f"✅ Saved {len(preprocessed_frames)} preprocessed frames to: {output_path}")
    else:
        print("⚠️ No valid frames found in the video.")

if __name__ == "__main__":
    process_video(VIDEO_PATH, OUTPUT_DIR)
