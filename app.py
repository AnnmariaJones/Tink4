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

# Prediction function
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    
    if model:
        # TODO: Add real preprocessing + inference
        label = "FAKE"
        confidence = 0.85  # placeholder
    else:
        label = "FAKE"
        confidence = 0.85  # placeholder

    remark = "High likelihood of manipulation" if label == "FAKE" else "Looks authentic"
    return label, confidence, remark

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

        label, confidence, remark = predict_image(filepath)

        return render_template("result.html", 
                               result=label, 
                               confidence=confidence, 
                               remark=remark)

    return render_template("index.html")

if __name__ == "__main__":
    print("Starting Flask deepfake detector app...")
    app.run(debug=True)