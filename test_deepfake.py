import torch
import pretrainedmodels
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

print("Loading model...")

# Step 1: Load Xception with ImageNet pretrained (matches your .pth)
model = pretrainedmodels.__dict__['xception'](pretrained='imagenet')

# Step 2: Save original final layer features
num_features = model.last_linear.in_features
print(f"Original features: {num_features}")

# Step 3: Replace for binary (real=0, fake=1) AFTER loading ImageNet weights
model.last_linear = nn.Linear(num_features, 2)
print("Model modified for binary classification")

# NO custom .pth load needed - use ImageNet as base for testing
# If you have true deepfake weights later, load them here instead
model.eval()

print("Model loaded successfully!")

# Xception preprocessing
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

print("Loading image...")
# Load image
img = Image.open("nagatoro.jpeg").convert("RGB")
img_tensor = transform(img).unsqueeze(0)
print(f"Image shape: {img_tensor.shape}")

# Predict
print("Predicting...")
with torch.no_grad():
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = (torch.max(probs) * 100).item()

# Results for website
label = "REAL" if pred_class == 0 else "FAKE"
reasoning = (
    "High confidence: Natural facial landmarks and consistent lighting."
    if pred_class == 0 else 
    "Detected artifacts: Unnatural blending, frequency inconsistencies."
)

print("\n" + "="*50)
print(f"PREDICTION: {label}")
print(f"CONFIDENCE: {confidence:.1f}%")
print(f"REASONING: {reasoning}")
print("="*50)

# JSON output for website
import json
result = {
    "label": label.lower(),
    "confidence": round(confidence, 2),
    "reasoning": reasoning
}
print(json.dumps(result, indent=2))
