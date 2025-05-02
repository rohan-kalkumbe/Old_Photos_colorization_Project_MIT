from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import os
from PIL import Image

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ✅ Correct prototxt filename
PROTOTXT_PATH = r"C:\Users\Adin\OneDrive\Desktop\MIT\old_photos_colorization\models\colorize.prototext"
MODEL_PATH = r"C:\Users\Adin\OneDrive\Desktop\MIT\old_photos_colorization\models\release.caffemodel"
POINTS_PATH = r"C:\Users\Adin\OneDrive\Desktop\MIT\old_photos_colorization\models\pts_in_hull.npy"


# Load model
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
pts = np.load(POINTS_PATH)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    image = cv2.imread(filepath)
    if image is None:
        return "Invalid image format", 400

    original_size = (image.shape[1], image.shape[0])
    lab = cv2.cvtColor(image.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
    L_original = lab[:, :, 0]  # Extract L channel (original size)
    L_input = cv2.resize(L_original, (224, 224)) - 50  # Resize for model
    net.setInput(cv2.dnn.blobFromImage(L_input))
    ab_base = net.forward()[0].transpose((1, 2, 0))
    ab_base = cv2.resize(ab_base, original_size)

    result_paths = []
    for i in range(8):
        ab = ab_base * (1 + (i - 4) * 0.1)
        lab_output = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(lab_output, cv2.COLOR_LAB2BGR)
        colorized = (np.clip(colorized, 0, 1) * 255).astype("uint8")
        result_path = os.path.join(RESULT_FOLDER, f"colorized_{i}_{file.filename}")
        cv2.imwrite(result_path, colorized)
        result_paths.append(f"/static/results/colorized_{i}_{file.filename}")

    return jsonify({"images": result_paths})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
