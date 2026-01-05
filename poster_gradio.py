import gradio as gr
import requests
import tempfile
import os

API_URL = "http://localhost:5075/predict"

def predict_genre(image):
    # Create a temporary directory instead of a NamedTemporaryFile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "input.png")

        # Save the image manually
        image.save(tmp_path)

        # Open file for sending
        with open(tmp_path, "rb") as f:
            response = requests.post(API_URL, files={"file": f})

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

interface.launch(server_name="127.0.0.1", server_port=7860)

