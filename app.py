from flask import Flask, render_template, request
import torch
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Make sure uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model safely
model = None
model_path = "model/xception-b5690688.pth"

if os.path.exists(model_path):
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading model:", e)
else:
    print(f"Model file not found at {model_path}. The app will run but predictions will be placeholder.")

# Prediction function (placeholder if model is None)
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    
    # If model loaded, you can add preprocessing + real prediction here
    if model:
        # TODO: Add real preprocessing and prediction
        return "FAKE"  # placeholder until you implement
    else:
        return "FAKE"  # placeholder

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part in request!"
        file = request.files["image"]
        if file.filename == "":
            return "No file selected!"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"Saved file to {filepath}")

        result = predict_image(filepath)

        # Check if template exists
        try:
            return render_template("result.html", result=result)
        except Exception as e:
            return f"Error rendering template: {e}"

    try:
        return render_template("index.html")
    except Exception as e:
        return f"Error rendering template: {e}"

# Start app safely
if __name__ == "__main__":
    print("Starting Flask deepfake detector app...")
    app.run(debug=True)