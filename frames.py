import cv2
import dlib
import os
import numpy as np

# Path to dataset
VIDEO_PATH = "D:\\LipReadingProject\\dataset\\videos\\s2"
OUTPUT_PATH = "D:\\LipReadingProject\\preprocessing\\frames1"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Loop through each video file in the dataset
for video_file in os.listdir(VIDEO_PATH):
    if video_file.endswith(".mpg"):  # Check video file format
        video_path = os.path.join(VIDEO_PATH, video_file)
        video_capture = cv2.VideoCapture(video_path)
        frame_count = 0
        success, frame = video_capture.read()
        while success:
            # Save each frame as an image
            video_name = os.path.splitext(video_file)[0]  # Get video name without extension
            video_output_path = os.path.join(OUTPUT_PATH, video_name)  # Subfolder path
            os.makedirs(video_output_path, exist_ok=True)  # Create subfolder
            frame_file = os.path.join(video_output_path, f"{frame_count}.jpg")  # Frame path
            cv2.imwrite(frame_file, frame)
            frame_count += 1
            success, frame = video_capture.read()
        video_capture.release()
