import gradio as gr
import requests
import tempfile
import os
import io

# API_URL = "http://localhost:5075/predict"
API_URL = "http://api:5075/predict"

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

interface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Movie Poster Genre Classifier",
    description="Upload a movie poster and the API predicts the genre."
)

interface.launch(server_name="0.0.0.0", server_port=7860)


