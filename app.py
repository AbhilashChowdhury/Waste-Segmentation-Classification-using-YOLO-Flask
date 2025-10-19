from flask import Flask, render_template, request, send_from_directory, g
from PIL import Image
import os
import uuid
import time
import io
import base64
from ultralytics import YOLO

# Flask app setup
app = Flask(__name__)

# Directories
UPLOAD_FOLDER = os.path.join("static", "uploaded_images")
MODEL_PATH = "best.pt"

# Ensure directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model once per request
def get_model():
    if 'model' not in g:
        g.model = YOLO(MODEL_PATH)
    return g.model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file part.")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No selected file.")

        if file:
            # Generate unique filename
            file_ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}.{file_ext}"
            uploaded_path = os.path.join(UPLOAD_FOLDER, unique_filename)

            # Save the uploaded image
            image = Image.open(file)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(uploaded_path)

            # Time the prediction
            start_time = time.time()

            # Run YOLO prediction (no disk saving)
            model = get_model()
            results = model(uploaded_path)

            end_time = time.time()
            print(f"Prediction time: {end_time - start_time:.2f} seconds")

            # Get prediction labels
            classes = results[0].names
            labels = set([classes[int(cls)] for cls in results[0].boxes.cls])
            prediction_result = ", ".join(labels)

            # Plot predictions and convert to base64
            plotted_image = results[0].plot()
            img_pil = Image.fromarray(plotted_image)
            buffered = io.BytesIO()
            img_pil.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Final image to embed in HTML
            predicted_image_base64 = f"data:image/jpeg;base64,{img_str}"

            return render_template(
                'index.html',
                uploaded_image=uploaded_path,
                predicted_image=predicted_image_base64,
                prediction=prediction_result
            )

    return render_template('index.html')

# Serve static files (optional)
@app.route('/static/<path:filename>')
def serve_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
