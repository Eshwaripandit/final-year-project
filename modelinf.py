import os
import sys
import subprocess

# Path to LipCoordNet model directory
lipcoordnet_dir = "D://LipReadingProject//New_model_code//LipCoordNet//LipCoordNet_Project//LipCoordNet"

def predict_from_video(video_path):
    try:
        weights_path = os.path.join(lipcoordnet_dir, "pretrain", "LipCoordNet_coords_loss_0.025581153109669685_wer_0.01746208431890914_cer_0.006488426950253695.pt")
        predictor_path = os.path.join(lipcoordnet_dir, "lip_coordinate_extraction", "shape_predictor_68_face_landmarks_GTX.dat")

        command = [
            sys.executable,
            os.path.join(lipcoordnet_dir, "inference.py"),
            "--weights", weights_path,
            "--input_video", video_path,
            "--device", "cpu",
            "--output_path", os.path.join(lipcoordnet_dir, "output_videos")
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            return lines[-1]  # Final predicted output from print()
        else:
            return f"Prediction failed: {result.stderr.strip()}"

    except Exception as e:
        return f"Exception during prediction: {str(e)}"
