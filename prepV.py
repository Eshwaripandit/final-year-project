import cv2
import os
import numpy as np
import dlib
from imutils import face_utils
from multiprocessing import Pool, cpu_count

# ✅ MODIFY PATHS FOR WINDOWS (Use raw strings or double backslashes)
INPUT_PATH = r"D:\LipReadingProject\videos_to_frames\FRAME_13"
OUTPUT_PATH = r"D:\LipReadingProject\preprocessing\pre_13"
LANDMARK_MODEL_PATH = r"D:\LipReadingProject\model\shape_predictor_68_face_landmarks.dat"

# Ensure output path exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Global Dlib model variables
detector = None
predictor = None

def init_dlib():
    global detector, predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)

def preprocess_frame(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    landmarks = predictor(gray, faces[0])
    landmarks = face_utils.shape_to_np(landmarks)

    lip_points = landmarks[48:68]
    x_min, y_min = np.min(lip_points, axis=0)
    x_max, y_max = np.max(lip_points, axis=0)

    padding = 10
    x_min, x_max = max(0, x_min - padding), min(gray.shape[1], x_max + padding)
    y_min, y_max = max(0, y_min - padding), min(gray.shape[0], y_max + padding)

    lip_region = gray[y_min:y_max, x_min:x_max]
    lip_resized = cv2.resize(lip_region, (128, 128), interpolation=cv2.INTER_CUBIC)
    lip_normalized = lip_resized.astype(np.float32) / 255.0

    return lip_normalized

def process_folder(folder_name):
    input_folder = os.path.join(INPUT_PATH, folder_name)
    output_folder = os.path.join(OUTPUT_PATH, folder_name)
    npy_output_path = os.path.join(output_folder, f"{folder_name}.npy")

    if os.path.exists(npy_output_path):
        print(f"⏩ Skipping already processed folder: {folder_name}")
        return

    os.makedirs(output_folder, exist_ok=True)
    frame_files = sorted(os.listdir(input_folder))
    frames_list = []

    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        preprocessed_frame = preprocess_frame(frame_path)
        if preprocessed_frame is not None:
            frames_list.append(preprocessed_frame)

    if frames_list:
        frames_array = np.array(frames_list)
        frames_array = np.expand_dims(frames_array, axis=-1)

        with open(npy_output_path, 'wb') as f:
            np.save(f, frames_array)

        print(f"✅ Saved: {npy_output_path}")
    else:
        print(f"⚠️ No valid frames found in folder: {folder_name}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"❌ INPUT_PATH not found: {INPUT_PATH}")

    folders = sorted(os.listdir(INPUT_PATH))
    print(f"📁 Found {len(folders)} folders to process.")

    with Pool(cpu_count(), initializer=init_dlib) as pool:
        pool.map(process_folder, folders)

    print("🎉 All folders processed successfully!")
