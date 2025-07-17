import cv2
import os

# 📌 Paths
VIDEO_PATH = "D:\\LipReadingProject\\dataset\\videos\\s18.mpg_vcd\\s18"
OUTPUT_PATH = "D:\\LipReadingProject\\preprocessing\\new_frames_24Mar\\sp_18"

# 📌 Ensure Output Directory Exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 📌 Get Sorted List of Video Files
video_files = sorted([f for f in os.listdir(VIDEO_PATH) if f.endswith(".mpg")])

if not video_files:
    print("❌ Error: No video files found in", VIDEO_PATH)
else:
    print(f"📂 Found {len(video_files)} video files. Processing...")

# 📌 Process Each Video File
for video_file in video_files:
    video_path = os.path.join(VIDEO_PATH, video_file)
    video_name = os.path.splitext(video_file)[0]  # Get video name without extension
    video_output_path = os.path.join(OUTPUT_PATH, video_name)  # Subfolder path

    # ✅ Create Subfolder for Frames
    os.makedirs(video_output_path, exist_ok=True)

    # ✅ Open Video File
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"❌ Error: Cannot open video {video_file}")
        continue

    print(f"🎥 Processing: {video_file}")

    frame_count = 0
    while True:
        success, frame = video_capture.read()
        
        if not success:
            print(f"✅ Completed: {video_file} (Extracted {frame_count} frames)")
            break  # Exit loop if no more frames

        if frame is None:
            print(f"⚠️ Warning: Skipped empty frame {frame_count} in {video_file}")
            continue

        # ✅ Save Frame (Ensure Frame is Not Empty)
        frame_file = os.path.join(video_output_path, f"{frame_count:06d}.jpg")  
        if cv2.imwrite(frame_file, frame):
            print(f"✅ Saved: {frame_file}")
        else:
            print(f"❌ Failed to save: {frame_file}")

        frame_count += 1

    video_capture.release()

print("🎉 Frame extraction completed successfully.")
