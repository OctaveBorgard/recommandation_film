import torch
from torchvision import transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
import torch.nn.functional as F
from models import  BertClf
from transformers import  DistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast
import os
import json
from models.models_for_rag import search_hybrid_split
from models.models_for_rag import search_hybrid_api


app = Flask(__name__)


device = torch.device("cpu")

# Classes of the dataset 
CLASSES = ["action", "animation", "comedy", "documentary", "drama",
           "fantasy", "horror", "romance", "science Fiction", "thriller"]


movie_categories = ["horror","comedy","drama","action","documentary",
                    "romance","science Fiction","animation","thriller","fantasy"]

# Load the trained model
from models import efficient_net
classifier_poster = efficient_net(num_classes=len(CLASSES))
state = torch.load("exp/poster_classification/EfficientNet_4837/checkpoints/epoch_030_test_avg_loss_0.0259.pth", map_location=device)
classifier_poster.load_state_dict(state["model_state_dict"])
classifier_poster.eval()

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distilbert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                  num_labels=len(movie_categories),
                                                                  output_attentions=True,
                                                                  output_hidden_states=True)
classifier_plot = BertClf(distilbert)
state = torch.load("exp/plot_classification/bertcls_4846/checkpoints/epoch_065_test_avg_loss_0.0068.pth", map_location=device)
classifier_plot.load_state_dict(state["model_state_dict"])
classifier_plot.eval()

# Preprocessing consistent with training
transform = transforms.Compose([
    transforms.Resize((278, 185)),
    transforms.ToTensor()
])

@app.route("/predict", methods=["POST"])
def predict_poster():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = classifier_poster(img_tensor)
        pred_idx = outputs.argmax(1).item()

    return jsonify({"genre": CLASSES[pred_idx]})
# Out-of-Distribution detection by Maximum Softmax Probability (MSP) for Part 2
def msp_score(logits):
    """
    Maximum Softmax Probability (OOD score)
    """
    probs = F.softmax(logits, dim=1)
    score, _ = torch.max(probs, dim=1)
    return score.item()

@app.route("/validate-poster", methods=["POST"])
def validate_poster():
    # 1. It's a file ?
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # 2. Read image
    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

    # 3. Preprocess like when we training
    img_tensor = transform(img).unsqueeze(0)

    # 4. Run classifier_poster + calcul MSP score
    with torch.no_grad():
        logits = classifier_poster(img_tensor)
        confidence = msp_score(logits)

    # 5. >= threshold => poster ; < threshold => not   
    THRESHOLD = 0.3

    # 6. Return result
    return jsonify({
        "is_poster": bool(confidence >= THRESHOLD),
        "confidence": float(confidence),
        "method": "OOD detection using Maximum Softmax Probability"
    })

@app.route("/predict_plot", methods=["POST"])
def predict_plot():
    if "text" not in request.json:
        return jsonify({"error": "No text provided"}), 400

    text = request.json["text"]

    inputs = tokenizer(text, padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    input_ids = torch.tensor(input_ids).unsqueeze(0)  
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)  

    with torch.no_grad():
        outputs,_,_ = classifier_plot(input_ids, attention_mask)

    _, pred_idx = outputs.max(1)
    return jsonify({"genre": movie_categories[pred_idx.item()]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5075)
    
    
@app.route("/search", methods=["POST"])
def search_movies():
    if not request.is_json:
        return jsonify({"error": "Expected JSON body"}), 400

    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400

    results = search_hybrid_api(query)

    return jsonify({
        "query": query,
        "results": results
    })
