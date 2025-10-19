from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import os
import uuid
from ultralytics import YOLO

# Flask app
app = Flask(__name__)

# Directories
UPLOAD_FOLDER = os.path.join("static", "uploaded_images")
OUTPUT_FOLDER = os.path.join("static", "output")
MODEL_PATH = "best.pt"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file part.")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No selected file.")

        if file:
            # Save uploaded image
            file_ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}.{file_ext}"
            uploaded_path = os.path.join(UPLOAD_FOLDER, unique_filename)

            image = Image.open(file)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(uploaded_path)

            # Run YOLO model
            results = model(uploaded_path, project="static", name="output", save=True, exist_ok=True)

            # Get predictions
            classes = results[0].names
            labels = set([classes[int(cls)] for cls in results[0].boxes.cls])

            # Predicted image path
            predicted_image_path = os.path.join("static", "output", unique_filename)
            prediction_result = ", ".join(labels)

            return render_template(
                'index.html',
                uploaded_image=uploaded_path,
                predicted_image=predicted_image_path,
                prediction=prediction_result
            )

    return render_template('index.html')


# Serve uploaded and output files if needed (optional for debugging)
@app.route('/static/<path:filename>')
def serve_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
