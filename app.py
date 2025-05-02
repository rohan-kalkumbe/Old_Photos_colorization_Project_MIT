from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
MODELS_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Paths to model files
PROTOTXT_PATH = os.path.join(MODELS_FOLDER, "colorize.prototext")
MODEL_PATH = os.path.join(MODELS_FOLDER, "release.caffemodel")
POINTS_PATH = os.path.join(MODELS_FOLDER, "pts_in_hull.npy")

# Load model
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
pts = np.load(POINTS_PATH)
pts = pts.transpose().reshape(2, 313, 1, 1)

# Assign cluster centers and prior using legacy .blobs assignment
class8_id = net.getLayerId("class8_ab")
conv8_id = net.getLayerId("conv8_313_rh")
net.getLayer(class8_id).blobs = [pts.astype("float32")]
net.getLayer(conv8_id).blobs = [np.full([1, 313], 2.606, dtype="float32")]

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
    L_original = lab[:, :, 0]
    L_input = cv2.resize(L_original, (224, 224)) - 50

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
