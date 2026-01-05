import torch
import torch.nn as nn
from torchvision import transforms
from flask import Flask, request, jsonify
from PIL import Image
import io

from modele_from_scratch import VGG16   

app = Flask(__name__)


device = torch.device("cpu")

# Classes of the dataset 
CLASSES = [
    "action",
    "animation",
    "comedy",
    "documentary",
    "drama",
    "fantasy",
    "horror",
    "romance",
    "science Fiction",  
    "thriller"
]

# Load the trained model
model = VGG16(num_classes=len(CLASSES))
model.load_state_dict(torch.load("genre_model.pth", map_location=device))
model.eval()

# Preprocessing consistent with training
transform = transforms.Compose([
    transforms.Resize((278, 185)),
    transforms.ToTensor()
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred_idx = outputs.argmax(1).item()

    return jsonify({"genre": CLASSES[pred_idx]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5075)
