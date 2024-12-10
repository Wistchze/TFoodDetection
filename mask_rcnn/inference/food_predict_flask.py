# app.py

import os
import threading
import base64
from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib
from flask_cors import CORS  # Import CORS

# Suppress TensorFlow logging for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Define class names (Ensure this matches your trained model)
CLASS_NAMES = [
    "BG",
    "-",
    "18Friedegg",
    "Ayam",
    "Fried Rice",
    "Lalapan",
    "Sambal",
    "Tumis mie",
    "apple",
    "ayam-kentucky-dada",
    "ayam-kentucky-paha",
    "banana",
    "beef hamburger",
    "chicken-burger",
    "fried tofu",
    "indomie_goreng",
    "nasi_putih",
    "nugget",
    "omelet",
    "orange",
    "paha_ayam_goreng",
    "pisang",
    "rendang sapi",
    "rice",
    "sambal",
    "stir-fried kale",
    "tahu goreng",
    "tahu_goreng",
    "telur_dadar",
    "telur_rebus",
    "tempe goreng",
    "tempe_goreng",
    "tumis_kangkung"
]

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

# Enable CORS for all routes
CORS(app)

# Define the Mask R-CNN configuration
class InferenceConfig(Config):
    NAME = "foods_cfg"
    NUM_CLASSES = 33  # Update based on your dataset (including BG)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7

# Initialize a global threading lock
model_lock = threading.Lock()

# Define the Mask R-CNN Inference class
class MaskRCNNInference:
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls, model_path, class_names):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super(MaskRCNNInference, cls).__new__(cls)
                    cls._instance._initialize(model_path, class_names)
        return cls._instance

    def _initialize(self, model_path, class_names):
        self.class_names = class_names
        self.config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode='inference', config=self.config, model_dir=os.path.dirname(model_path))
        self.model.load_weights(model_path, by_name=True)
        print("Mask R-CNN model loaded successfully.")

    def infer(self, image_bytes):
        with model_lock:
            # Convert bytes data to NumPy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            # Decode image from NumPy array
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Unable to decode the image.')

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform detection
            results = self.model.detect([image_rgb], verbose=0)
            r = results[0]

            # Extract class names from class IDs
            detected_classes = [self.class_names[class_id] for class_id in r['class_ids']]

            # Annotate the image with bounding boxes and labels
            annotated_image = image.copy()
            for i in range(len(r['class_ids'])):
                class_id = r['class_ids'][i]
                score = r['scores'][i]
                if score < self.config.DETECTION_MIN_CONFIDENCE:
                    continue
                y1, x1, y2, x2 = r['rois'][i]
                label = self.class_names[class_id]
                # Draw rectangle
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the annotated image to JPEG format in memory
            _, buffer = cv2.imencode('.jpg', annotated_image)
            annotated_image_bytes = buffer.tobytes()
            # Encode the image bytes to base64
            annotated_image_base64 = base64.b64encode(annotated_image_bytes).decode('utf-8')

            return detected_classes, annotated_image_base64

# Initialize the Mask R-CNN model (Update the MODEL_PATH accordingly)
MODEL_PATH = 'logs/mrcnn_food_detection.h5'  # Replace with your actual model path

# Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f'Trained weights not found at {MODEL_PATH}. Please check the path.')

# Initialize the inference model
mask_rcnn = MaskRCNNInference(model_path=MODEL_PATH, class_names=CLASS_NAMES)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# New Route: /detect (POST Only) for API Usage
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400
    if file and allowed_file(file.filename):
        try:
            # Read the image bytes directly from the uploaded file
            image_bytes = file.read()
            # Perform inference
            detected_classes, annotated_image_base64 = mask_rcnn.infer(image_bytes)
            # Return JSON response
            return jsonify({
                'detected_classes': detected_classes,
                'annotated_image': annotated_image_base64
            }), 200
        except Exception as e:
            return jsonify({'error': f'An error occurred during inference: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg, bmp.'}), 400

# Run the Flask app
if __name__ == '__main__':
    # Run the app in single-threaded mode as per your decision
    app.run(host='0.0.0.0', port=5000, threaded=False)
