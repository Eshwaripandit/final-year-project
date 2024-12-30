import cv2
import dlib
import os
import numpy as np

# Paths
LANDMARKED_FRAMES_PATH = "D:\\LipReadingProject\\preprocessing\\landmark1"  # Input frames folder with landmarks
OUTPUT_CROPPED_PATH = "D:\\LipReadingProject\\preprocessing\\normal-rgb"  # Path to save cropped, resized, and processed frames

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_CROPPED_PATH, exist_ok=True)

# Function to preprocess frames
def preprocess_frames():
    # Loop through video folders in the landmarked frames directory
    for video_folder in os.listdir(LANDMARKED_FRAMES_PATH):
        video_folder_path = os.path.join(LANDMARKED_FRAMES_PATH, video_folder)
        if os.path.isdir(video_folder_path):  # Ensure it's a directory
            output_folder = os.path.join(OUTPUT_CROPPED_PATH, video_folder)
            os.makedirs(output_folder, exist_ok=True)

            for frame_file in os.listdir(video_folder_path):
                if frame_file.endswith(".jpg"):  # Ensure only image files are processed
                    frame_path = os.path.join(video_folder_path, frame_file)
                    frame = cv2.imread(frame_path)

                    # Skip grayscale conversion since landmarks are detected and grayscale isn't required
                    # Detect face and landmarks
                    detector = dlib.get_frontal_face_detector()
                    predictor = dlib.shape_predictor("C:\\Users\\Kankana\\Downloads\\shape_predictor_68_face_landmarks (2).dat")  # Ensure the file is available
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale conversion for landmark detection
                    faces = detector(gray)

                    for face in faces:
                        landmarks = predictor(gray, face)

                        # Extract lip landmarks (points 48 to 67)
                        lip_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
                        x_min, y_min = np.min(lip_points, axis=0)
                        x_max, y_max = np.max(lip_points, axis=0)

                        # Add padding to the bounding box
                        padding = 15
                        x_min = max(0, x_min - padding)
                        y_min = max(0, y_min - padding)
                        x_max = min(frame.shape[1], x_max + padding)
                        y_max = min(frame.shape[0], y_max + padding)

                        # Crop the lip region
                        cropped_lip = frame[y_min:y_max, x_min:x_max]  # Use original frame for better quality cropping

                        # Resize to a consistent resolution
                        resized_frame = cv2.resize(cropped_lip, (224, 224), interpolation=cv2.INTER_AREA)

                        # Normalize pixel values to [0, 1]
                        normalized_frame = resized_frame / 255.0

                        # Save the processed frame
                        output_frame_path = os.path.join(output_folder, frame_file)
                        cv2.imwrite(output_frame_path, (normalized_frame * 255).astype(np.uint8))  # Convert back to uint8 for saving

                    print(f"Processed {frame_file} in {video_folder}")

# Run the preprocessing function
if __name__ == "__main__":
    preprocess_frames()
