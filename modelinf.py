# # # # from lipnet_model import LipNet
# # # # from preprocess import preprocess_video
# # # # from model_utils import decode_prediction  # Assuming you have a decoder
# # # # import numpy as np

# # # # def predict_from_video(video_path):
# # # #     # Step 1: Preprocess the video (frame extraction & normalization)
# # # #     input_data = preprocess_video(video_path)

# # # #     # Step 2: Load the trained LipNet model
# # # #     model = LipNet()
# # # #     model.model.load_weights("path/to/your/lipnet_weights.h5")

# # # #     # Step 3: Perform prediction
# # # #     y_pred = model.model.predict(input_data)  # shape: (1, 75, output_dim)

# # # #     # Step 4: Decode the prediction to readable text
# # # #     predicted_text = decode_prediction(y_pred)

# # # #     return predicted_text

# # # # import numpy as np

# # # # def decode_prediction(y_pred):
# # # #     # Assume y_pred shape = (1, time_steps, output_size)
# # # #     pred = y_pred[0]  # Remove batch dimension

# # # #     # Greedy decoding (argmax)
# # # #     decoded_indices = np.argmax(pred, axis=-1)

# # # #     # Map indices to characters
# # # #     char_list = "abcdefghijklmnopqrstuvwxyz "  # Adjust if needed
# # # #     decoded_text = ''.join([char_list[i] if i < len(char_list) else '' for i in decoded_indices])

# # # #     # Remove consecutive duplicates (CTC collapsing)
# # # #     final_text = ''
# # # #     prev_char = ''
# # # #     for char in decoded_text:
# # # #         if char != prev_char:
# # # #             final_text += char
# # # #             prev_char = char

# # # #     return final_text.strip()



# # # # predict_from_video.py
# # # import numpy as np
# # # from lipnet_model import LipNet
# # # from preprocess import preprocess_video
# # # from model_utils import decode_prediction

# # # def predict_from_video(video_path):
# # #     # Step 1: Preprocess
# # #     input_data = preprocess_video(video_path)
# # #     input_data = np.expand_dims(input_data, axis=0)  # (1, T, 128, 128, 1)

# # #     # Step 2: Load model
# # #     model = LipNet()
# # #     model.model.load_weights("model/your_model_weights.h5")

# # #     # Step 3: Predict
# # #     y_pred = model.model.predict(input_data)

# # #     # Step 4: Decode
# # #     predicted_text = decode_prediction(y_pred)
# # #     return predicted_text





# # # from tensorflow.keras.models import load_model
# # # import numpy as np

# # # def decode_prediction(y_pred):
# # #     char_list = "abcdefghijklmnopqrstuvwxyz "  # Add blank or punctuation if needed
# # #     decoded = np.argmax(y_pred[0], axis=-1)

# # #     final_text = ''
# # #     prev = -1
# # #     for c in decoded:
# # #         if c != prev:
# # #             if c < len(char_list):
# # #                 final_text += char_list[c]
# # #             prev = c
# # #     return final_text.strip()

# # # def predict_from_video(np_array):
# # #     if np_array.ndim == 3:  # (T, 128, 128)
# # #         np_array = np_array[..., np.newaxis]
# # #     np_array = np.expand_dims(np_array, axis=0)  # Add batch dim

# # #     model = load_model(r"D:\LipReadingProject\model\lipreading_model.pth")  # Update this path if needed
# # #     y_pred = model.predict(np_array)
# # #     return decode_prediction(y_pred)




# # # import torch
# # # import torch.nn as nn
# # # import numpy as np

# # # # 🔁 Step 1: Define your model class structure (must match training)
# # # class LipReadingModel(nn.Module):
# # #     def __init__(self):
# # #         super(LipReadingModel, self).__init__()
# # #         self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
# # #         self.pool = nn.MaxPool3d(2)
# # #         self.fc1 = nn.Linear(16 * 32 * 32 * 4, 128)  # Adjust based on output of conv-pool
# # #         self.fc2 = nn.Linear(128, 27)  # 26 letters + space

# # #     def forward(self, x):
# # #         x = self.pool(torch.relu(self.conv1(x)))
# # #         x = x.view(x.size(0), -1)
# # #         x = torch.relu(self.fc1(x))
# # #         return self.fc2(x)

# # # # 🔁 Step 2: Decoder
# # # def decode_prediction(y_pred):
# # #     char_list = "abcdefghijklmnopqrstuvwxyz "  # index 0-26
# # #     decoded = torch.argmax(y_pred, dim=-1).cpu().numpy()[0]

# # #     final_text = ''
# # #     prev = -1
# # #     for c in decoded:
# # #         if c != prev:
# # #             if c < len(char_list):
# # #                 final_text += char_list[c]
# # #             prev = c
# # #     return final_text.strip()

# # # # 🔁 Step 3: Inference function
# # # def predict_from_video(np_array):
# # #     # (T, 128, 128, 1) → (1, 1, T, 128, 128)
# # #     if np_array.ndim == 3:
# # #         np_array = np_array[..., np.newaxis]
# # #     input_tensor = torch.tensor(np_array).permute(3, 0, 1, 2).unsqueeze(0).float()  # (1, 1, T, H, W)

# # #     # Load the model
# # #     model = LipReadingModel()
# # #     model.load_state_dict(torch.load(r"D:\LipReadingProject\model\lipreading_model.pth", map_location=torch.device('cpu')))
# # #     model.eval()

# # #     # Run inference
# # #     with torch.no_grad():
# # #         y_pred = model(input_tensor)
    
# # #     return decode_prediction(y_pred)






# # import torch
# # import torch.nn as nn
# # import numpy as np

# # # ✅ Define actual architecture from checkpoint
# # class LipReadingModel(nn.Module):
# #     def __init__(self):
# #         super(LipReadingModel, self).__init__()
# #         self.conv3d = nn.Sequential(
# #             nn.Conv3d(1, 32, kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.MaxPool3d((1, 2, 2)),
# #             nn.Conv3d(32, 64, kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.MaxPool3d((1, 2, 2))
# #         )

# #         self.lstm = nn.LSTM(input_size=64 * 32 * 32, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
# #         self.fc = nn.Linear(128 * 2, 27)  # 26 letters + space

# #     def forward(self, x):  # x shape: (B, 1, T, H, W)
# #         x = self.conv3d(x)  # (B, C, T, H, W)
# #         x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
# #         x = x.flatten(2)  # (B, T, C*H*W)
# #         x, _ = self.lstm(x)  # (B, T, 2*hidden_size)
# #         return self.fc(x)  # (B, T, 27)

# # # 🔁 Decoder
# # def decode_prediction(y_pred):
# #     char_list = "abcdefghijklmnopqrstuvwxyz "
# #     decoded = torch.argmax(y_pred, dim=-1).cpu().numpy()[0]

# #     final_text = ''
# #     prev = -1
# #     for c in decoded:
# #         if c != prev and c < len(char_list):
# #             final_text += char_list[c]
# #             prev = c
# #     return final_text.strip()

# # # 🔁 Inference
# # def predict_from_video(np_array):
# #     if np_array.ndim == 3:
# #         np_array = np_array[..., np.newaxis]
# #     input_tensor = torch.tensor(np_array).permute(3, 0, 1, 2).unsqueeze(0).float()  # (1, 1, T, H, W)

# #     model = LipReadingModel()
# #     model.load_state_dict(torch.load(r"D:\LipReadingProject\model\lipreading_model.pth", map_location='cpu'))
# #     model.eval()

# #     with torch.no_grad():
# #         y_pred = model(input_tensor)
# #     return decode_prediction(y_pred)





# import torch
# import torch.nn as nn
# import numpy as np

# # Decoder
# def decode_prediction(y_pred):
#     char_list = "abcdefghijklmnopqrstuvwxyz "
#     decoded = torch.argmax(y_pred, dim=-1).squeeze().tolist()
    
#     final_text = ''
#     prev = -1
#     for c in decoded:
#         if c != prev:
#             if c < len(char_list):
#                 final_text += char_list[c]
#             prev = c
#     return final_text.strip()

# # Model architecture matching .pth
# class LipReadingModel(nn.Module):
#     def __init__(self):
#         super(LipReadingModel, self).__init__()
#         self.conv3d = nn.Sequential(
#             nn.Conv3d(1, 16, kernel_size=3, padding=1),  # shape match: [16, 1, 3, 3, 3]
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=(1,2,2)),
#             nn.Conv3d(16, 32, kernel_size=3, padding=1),  # shape match: [32, 16, 3, 3, 3]
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=(1,2,2))
#         )

#         self.lstm = nn.LSTM(
#             input_size=32 * 32 * 32,  # flatten shape from conv output
#             hidden_size=256,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True
#         )

#         self.fc = nn.Linear(256 * 2, 27)  # 26 letters + space

#     def forward(self, x):  # x: (B, 1, T, 128, 128)
#         x = self.conv3d(x)  # output: (B, C, T, H, W)
#         x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
#         x = x.contiguous().view(x.size(0), x.size(1), -1)  # (B, T, C*H*W)
#         x, _ = self.lstm(x)  # (B, T, 2*hidden)
#         x = self.fc(x)  # (B, T, vocab)
#         return x

# # Prediction function
# def predict_from_video(np_array):
#     if np_array.ndim == 3:
#         np_array = np_array[..., np.newaxis]  # (T, H, W, 1)

#     np_array = np.expand_dims(np_array, axis=0)  # (1, T, H, W, 1)
#     np_array = np.transpose(np_array, (0, 4, 1, 2, 3))  # (1, 1, T, H, W)

#     tensor_input = torch.tensor(np_array, dtype=torch.float32)

#     model = LipReadingModel()
#     model.load_state_dict(torch.load("D:/LipReadingProject/model/lipreading_model.pth", map_location='cpu'))
#     model.eval()

#     with torch.no_grad():
#         output = model(tensor_input)
#         return decode_prediction(output)






import sys
import os

# 🔧 Add path to access model.py and dataset.py
sys.path.append("D:/LipReadingProject/New_model_code/LipCoordNet/LipCoordNet_Project/LipCoordNet")

# ✅ Now import
from model import LipCoordNet
from dataset import MyDataset
import torch
import os
import cv2
import numpy as np
import face_alignment
import dlib
import glob 

# --- Constants ---
MODEL_PATH = r"D:\LipReadingProject\New_model_code\LipCoordNet\LipCoordNet_Project\LipCoordNet\pretrain\LipCoordNet_coords_loss_0.025581153109669685_wer_0.01746208431890914_cer_0.006488426950253695.pt"
SHAPE_PREDICTOR_PATH = r"D:\LipReadingProject\New_model_code\LipCoordNet\LipCoordNet_Project\LipCoordNet\lip_coordinate_extraction\shape_predictor_68_face_landmarks_GTX.dat"


# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model once ---
model = LipCoordNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

# --- Dlib ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

def get_position(size, padding=0.25):
    # (same as in inference.py)
    ...
    # return np.array(list(zip(x, y)))

def transformation_from_points(points1, points2):
    ...

def extract_lip_coordinates(detector, predictor, img_path):
    ...

def generate_lip_coordinates(frame_images_directory, detector, predictor):
    ...

def ctc_decode(y):
    y = y.argmax(-1)
    t = y.size(0)
    result = []
    for i in range(t + 1):
        result.append(MyDataset.ctc_arr2txt(y[:i], start=1))
    return result

def preprocess_and_align(video_path, temp_folder="samples"):
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    os.system(f"ffmpeg -hide_banner -loglevel error -y -i {video_path} -qscale:v 2 -r 25 {temp_folder}/%04d.jpg")

    files = sorted(os.listdir(temp_folder), key=lambda x: int(os.path.splitext(x)[0]))
    array = [cv2.imread(os.path.join(temp_folder, file)) for file in files]
    array = list(filter(lambda im: im is not None, array))

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=str(device))
    points = [fa.get_landmarks(I) for I in array]

    front256 = get_position(256)
    video = []
    for point, scene in zip(points, array):
        if point is not None:
            shape = np.array(point[0])
            shape = shape[17:]
            M = transformation_from_points(np.matrix(shape), np.matrix(front256))
            img = cv2.warpAffine(scene, M[:2], (256, 256))
            (x, y) = front256[-20:].mean(0).astype(np.int32)
            w = 160 // 2
            img = img[y - w // 2: y + w // 2, x - w: x + w, ...]
            img = cv2.resize(img, (128, 64))
            video.append(img)

    video = np.stack(video, axis=0).astype(np.float32)
    video = torch.FloatTensor(video.transpose(3, 0, 1, 2)) / 255.0
    return video, temp_folder

def predict_from_video(video_path):
    video_tensor, sample_path = preprocess_and_align(video_path)
    coords = generate_lip_coordinates(sample_path, detector, predictor)
    with torch.no_grad():
        pred = model(video_tensor[None, ...].to(device), coords[None, ...].to(device))
    output = ctc_decode(pred[0])
    return output[-1]
