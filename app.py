# # #from flask import Flask, request, jsonify, render_template
# # #from werkzeug.utils import secure_filename
# # #import os

# # #app = Flask(__name__)
# # #UPLOAD_FOLDER = 'uploads'
# # #os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # #app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # #@app.route('/')
# # #def index():
# #    # return render_template('videup.html')

# # #@app.route('/ajax-upload', methods=['POST'])
# # #def ajax_upload():
# #    # if 'video' not in request.files:
# #        # return jsonify({'error': 'No file part'}), 400

# #     #file = request.files['video']
# #     #if file.filename == '':
# #        # return jsonify({'error': 'No selected file'}), 400

# #     #filename = secure_filename(file.filename)
# #     #filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #     #file.save(filepath)

# #     # Dummy LSTM response (replace this with actual inference call)
# #     #generated_subtitles = "subtitles generated using the LSTM model!"

# #     #return jsonify({'subtitles': generated_subtitles})

# # #if __name__ == '__main__':
# #     #app.run(debug=True) 






# # #from flask import Flask, request, jsonify, render_template
# # #from werkzeug.utils import secure_filename
# # #import os

# # #app = Flask(__name__)

# # # Folder to save uploads (relative to this script's directory)
# # #UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
# # #os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # #app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # # Allowed video extensions
# # #ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'mpg'}

# # #def allowed_file(filename):
# #  #   return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # #@app.route('/')
# # #def index():
# #     # Make sure you have videup.html in a folder named 'templates' next to this script
# #  #   return render_template('videup.html')

# # #@app.route('/ajax-upload', methods=['POST'])
# # #def ajax_upload():
# #  #   if 'video' not in request.files:
# #   #      return jsonify({'error': 'No file part'}), 400

# #    # file = request.files['video']

# #     #if file.filename == '':
# #      #   return jsonify({'error': 'No selected file'}), 400

# #     #if not allowed_file(file.filename):
# #     #    return jsonify({'error': 'File type not allowed'}), 400

# #    # filename = secure_filename(file.filename)
# #    # save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #    # file.save(save_path)

# #     # Replace with your model inference call here:
# #    # generated_subtitles = "subtitles generated using the LSTM model!"

# #    # return jsonify({'success': True, 'filename': filename, 'subtitles': generated_subtitles})

# # #if __name__ == '__main__':
# #     #app.run(debug=True)








# # from flask import Flask, request, jsonify, render_template
# # from werkzeug.utils import secure_filename
# # import os
# # import cv2
# # import numpy as np
# # import dlib
# # from imutils import face_utils

# # # Flask setup
# # app = Flask(__name__)
# # BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# # UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# # PREPROCESS_FOLDER = os.path.join(BASE_DIR, 'preprocessed')
# # #MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, 'D:/LipReadingProject/model/shape_predictor_68_face_landmarks.dat', 'models', 'shape_predictor_68_face_landmarks.dat'))
# # MODEL_PATH = "D:/LipReadingProject/model/shape_predictor_68_face_landmarks.dat"


# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(PREPROCESS_FOLDER, exist_ok=True)
# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # # Check for model
# # assert os.path.exists(MODEL_PATH), f"❌ Landmark model not found at {MODEL_PATH}"

# # # Load dlib model
# # detector = dlib.get_frontal_face_detector()
# # predictor = dlib.shape_predictor(MODEL_PATH)

# # ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'mpg'}

# # def allowed_file(filename):
# #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # def preprocess_video(video_path, output_path):
# #     cap = cv2.VideoCapture(video_path)
# #     frames = []

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         faces = detector(gray)

# #         if faces:
# #             landmarks = predictor(gray, faces[0])
# #             landmarks = face_utils.shape_to_np(landmarks)
# #             lips = landmarks[48:68]
# #             x_min, y_min = np.min(lips, axis=0)
# #             x_max, y_max = np.max(lips, axis=0)

# #             pad = 10
# #             x_min, x_max = max(0, x_min - pad), min(gray.shape[1], x_max + pad)
# #             y_min, y_max = max(0, y_min - pad), min(gray.shape[0], y_max + pad)

# #             lip_crop = gray[y_min:y_max, x_min:x_max]
# #             resized = cv2.resize(lip_crop, (128, 128), interpolation=cv2.INTER_CUBIC)
# #             normalized = resized.astype(np.float32) / 255.0
# #             frames.append(normalized)

# #     cap.release()

# #     if frames:
# #         arr = np.array(frames)[..., np.newaxis]  # Shape: (T, 128, 128, 1)
# #         np.save(output_path, arr)
# #         return f"✅ Preprocessed video saved: {output_path}"
# #     else:
# #         return "⚠️ No valid frames found."

# # @app.route('/')
# # def index():
# #     return render_template('videup.html')

# # @app.route('/ajax-upload', methods=['POST'])
# # def ajax_upload():
# #     if 'video' not in request.files:
# #         return jsonify({'error': 'No video file provided'}), 400

# #     file = request.files['video']

# #     if file.filename == '':
# #         return jsonify({'error': 'No selected file'}), 400

# #     if file and allowed_file(file.filename):
# #         filename = secure_filename(file.filename)
# #         save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #         file.save(save_path)

# #         # Preprocess
# #         video_id = os.path.splitext(filename)[0]
# #         output_path = os.path.join(PREPROCESS_FOLDER, f"{video_id}.npy")
# #         preprocess_status = preprocess_video(save_path, output_path)

# #         return jsonify({
# #             'success': True,
# #             'filename': filename,
# #             'preprocessing': preprocess_status
# #         })

# #     return jsonify({'error': 'File type not allowed'}), 400

# # if __name__ == '__main__':
# #     app.run(debug=True)








# # from flask import Flask, request, jsonify, render_template
# # from werkzeug.utils import secure_filename
# # import os
# # import cv2
# # import numpy as np
# # import dlib
# # from imutils import face_utils

# # # --- Flask Setup ---
# # app = Flask(__name__)
# # BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# # UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# # PREPROCESS_FOLDER = os.path.join(BASE_DIR, 'preprocessed')
# # MODEL_PATH = r"D:\LipReadingProject\model\shape_predictor_68_face_landmarks.dat"

# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(PREPROCESS_FOLDER, exist_ok=True)
# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # # --- Dlib Setup ---
# # assert os.path.exists(MODEL_PATH), f"❌ Landmark model not found at {MODEL_PATH}"
# # detector = dlib.get_frontal_face_detector()
# # predictor = dlib.shape_predictor(MODEL_PATH)

# # ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'mpg'}

# # def allowed_file(filename):
# #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # # --- Preprocessing Function ---
# # def preprocess_video(video_path, output_path):
# #     cap = cv2.VideoCapture(video_path)
# #     frames = []

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         faces = detector(gray)

# #         if faces:
# #             landmarks = predictor(gray, faces[0])
# #             landmarks = face_utils.shape_to_np(landmarks)
# #             lips = landmarks[48:68]
# #             x_min, y_min = np.min(lips, axis=0)
# #             x_max, y_max = np.max(lips, axis=0)

# #             pad = 10
# #             x_min, x_max = max(0, x_min - pad), min(gray.shape[1], x_max + pad)
# #             y_min, y_max = max(0, y_min - pad), min(gray.shape[0], y_max + pad)

# #             lip_crop = gray[y_min:y_max, x_min:x_max]
# #             resized = cv2.resize(lip_crop, (128, 128), interpolation=cv2.INTER_CUBIC)
# #             normalized = resized.astype(np.float32) / 255.0
# #             frames.append(normalized)

# #     cap.release()

# #     if frames:
# #         arr = np.array(frames)[..., np.newaxis]  # Shape: (T, 128, 128, 1)
# #         np.save(output_path, arr)
# #         return f"✅ Preprocessed video saved to: {output_path}"
# #     else:
# #         return "⚠️ No valid frames found in video."

# # # --- Routes ---
# # @app.route('/')
# # def index():
# #     return render_template('videup.html')

# # @app.route('/ajax-upload', methods=['POST'])
# # def ajax_upload():
# #     if 'video' not in request.files:
# #         return jsonify({'error': 'No video file provided'}), 400

# #     file = request.files['video']

# #     if file.filename == '':
# #         return jsonify({'error': 'No selected file'}), 400

# #     if file and allowed_file(file.filename):
# #         filename = secure_filename(file.filename)
# #         save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #         file.save(save_path)

# #         video_id = os.path.splitext(filename)[0]
# #         output_path = os.path.join(PREPROCESS_FOLDER, f"{video_id}.npy")
# #         status = preprocess_video(save_path, output_path)

# #         return jsonify({
# #             'success': True,
# #             'filename': filename,
# #             'preprocessing': status
# #         })

# #     return jsonify({'error': 'File type not allowed'}), 400

# # if __name__ == '__main__':
# #     app.run(debug=True)










# from flask import Flask, request, jsonify, render_template
# from werkzeug.utils import secure_filename
# import os
# from predict_from_video import predict_from_video  # This will call the model

# app = Flask(__name__)
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# PREPROCESS_FOLDER = os.path.join(BASE_DIR, 'preprocessed')

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PREPROCESS_FOLDER, exist_ok=True)

# ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def index():
#     return render_template('videup.html')

# @app.route('/ajax-upload', methods=['POST'])
# def ajax_upload():
#     if 'video' not in request.files:
#         return jsonify({'error': 'No file'}), 400
#     file = request.files['video']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
#     if not allowed_file(file.filename):
#         return jsonify({'error': 'File type not allowed'}), 400

#     filename = secure_filename(file.filename)
#     save_path = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(save_path)

#     try:
#         predicted_text = predict_from_video(save_path)
#         return jsonify({'success': True, 'subtitles': predicted_text})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)






from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import dlib
from imutils import face_utils
from modelinf import predict_from_video  # <-- Your inference function

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PREPROCESS_FOLDER = os.path.join(BASE_DIR, 'preprocessed')
MODEL_PATH = MODEL_PATH = r"D:\LipReadingProject\New_model_code\LipCoordNet\LipCoordNet_Project\LipCoordNet\lip_coordinate_extraction\shape_predictor_68_face_landmarks_GTX.dat"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROCESS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check for model
assert os.path.exists(MODEL_PATH), f"❌ Landmark model not found at {MODEL_PATH}"

# Load dlib model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'mpg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if faces:
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
            frames.append(normalized)

    cap.release()

    if frames:
        arr = np.array(frames)[..., np.newaxis]  # Shape: (T, 128, 128, 1)
        np.save(output_path, arr)
        return arr
    else:
        return None

@app.route('/')
def index():
    return render_template('videup.html')

@app.route('/ajax-upload', methods=['POST'])
def ajax_upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Preprocess
        video_id = os.path.splitext(filename)[0]
        output_path = os.path.join(PREPROCESS_FOLDER, f"{video_id}.npy")
        preprocessed_data = preprocess_video(save_path, output_path)

        if preprocessed_data is None:
            return jsonify({'error': 'No valid frames found during preprocessing'}), 500

        # Predict from model
        subtitle = predict_from_video(preprocessed_data)

        return jsonify({
            'success': True,
            'filename': filename,
            'subtitle': subtitle
        })

    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
