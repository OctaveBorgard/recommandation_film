import gradio as gr
import requests
import tempfile
import os
import io

# API_URL = "http://localhost:5075/predict"
API_URL = "http://api:5075/predict"  #API_URL  for predict
API_VALIDATE_URL = "http://api:5075/validate-poster"   # Partie 2

def predict_genre(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    response = requests.post(
        API_URL,
        files={"file":("input.png", img_byte_arr,"image/png")}
    )

    if response.status_code != 200:
        return "Error: API did not return a valid response."

    return response.json().get("genre", "Unknown")

def validate_poster(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    response = requests.post(
        API_VALIDATE_URL,
        files={"file": ("input.png", img_byte_arr, "image/png")}
    )

    if response.status_code != 200:
        return "Error calling validation API"

    result = response.json()
    return f"Is poster: {result['is_poster']} (confidence: {result['confidence']:.2f})"

with gr.Blocks() as app:
    gr.Markdown("# Movie Poster Tool")

    image_in = gr.Image(type="pil", label="Upload Poster")

    with gr.Row():
        btn_genre = gr.Button("Predict Genre")
        btn_validate = gr.Button("Validate Poster")

    genre_out = gr.Textbox(label="Predicted Genre")
    valid_out = gr.Textbox(label="Poster Validation")

    # Mỗi nút gọi 1 hàm, dùng chung 1 input ảnh
    btn_genre.click(fn=predict_genre, inputs=image_in, outputs=genre_out)
    btn_validate.click(fn=validate_poster, inputs=image_in, outputs=valid_out)

app.launch(server_name="0.0.0.0", server_port=7860)