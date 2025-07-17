import cv2
import dlib
import os
import numpy as np 

# Paths
FRAMES_PATH = "D:\LipReadingProject\preprocessing\frames2"  # Input frames folder
OUTPUT_PATH = "D:\LipReadingProject\preprocessing\Landmark2"  # Folder to save processed frames

# Load Dlib's face detector and facial landmark predictor
predictor_path = "C:\\Users\\Kankana\\Downloads\\shape_predictor_68_face_landmarks (2).dat"  # Ensure this file is in the same directory or provide the full path
if not os.path.exists(predictor_path):
    raise FileNotFoundError("Download shape_predictor_68_face_landmarks.dat from http://dlib.net/")
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Process frames for each video
for video_folder in os.listdir(FRAMES_PATH):
    video_folder_path = os.path.join(FRAMES_PATH, video_folder)
    if os.path.isdir(video_folder_path):  # Ensure it's a folder
        output_folder = os.path.join(OUTPUT_PATH, video_folder)
        os.makedirs(output_folder, exist_ok=True)

        for frame_file in os.listdir(video_folder_path):
            if frame_file.endswith(".jpg"):  # Ensure only image files are processed
                frame_path = os.path.join(video_folder_path, frame_file)
                frame = cv2.imread(frame_path)

                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = detector(gray)

                for face in faces:
                    # Predict facial landmarks
                    landmarks = predictor(gray, face)

                    # Extract lip landmarks (points 48 to 67)
                    lip_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])

                    # Draw landmarks on the frame
                    for (x, y) in lip_points:
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw green dots for landmarks

                    # Save processed frame with landmarks
                    output_frame_path = os.path.join(output_folder, frame_file)
                    cv2.imwrite(output_frame_path, frame)

print("Lip landmark detection completed. Processed frames saved in:", OUTPUT_PATH)
