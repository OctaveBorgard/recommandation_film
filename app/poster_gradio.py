import gradio as gr
import requests
import tempfile
import os
import io

# API_URL = "http://localhost:5075/predict"
API_URL = "http://api:5075/predict" #API_URL  for predict
API_VALIDATE_URL = "http://api:5075/validate-poster"   # Partie 2
API_PREDICT_PLOT_URL = "http://api:5075/predict_plot"
API_SEARCH_MOVIE_URL = "http://api:5075/search"

def predict_genre_poster(image):
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

def predict_genre_from_plot(plot):
    response = requests.post(
        API_PREDICT_PLOT_URL,
        json={"text": plot}
    )

    if response.status_code != 200:
        return "Error: API did not return a valid response."

    return response.json().get("genre", "Unknown")
def search_movies_nl(query):
    response = requests.post(
        API_SEARCH_MOVIE_URL,
        json={"query": query}
    )

    if response.status_code != 200:
        return "Error calling search API"

    data = response.json()
    results = data.get("results", [])

    if not results:
        return "No results found."

    # Format text output simply
    output = ""
    for i, res in enumerate(results, 1):
        output += (
            f"{i}. Source: {res['source']}\n"
            f"   Score: {res['score']:.2f}\n"
            f"   Plot: {res['plot'][:150]}...\n\n"
        )

    return output


with gr.Blocks() as app:
    gr.Markdown("# Movie Poster Tool")

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Upload Poster")

        with gr.Column(scale=1):
            genre_out = gr.Textbox(label="Predicted Genre")
            valid_out = gr.Textbox(label="Poster Validation")

            btn_genre = gr.Button("Predict Genre")
            btn_validate = gr.Button("Validate Poster")

    btn_genre.click(fn=predict_genre_poster, inputs=image_in, outputs=genre_out)
    btn_validate.click(fn=validate_poster, inputs=image_in, outputs=valid_out)

    with gr.Row():
        # add box to enter text:
        plot_in = gr.Textbox(lines=4, placeholder="Enter movie plot here...", label="Movie Plot")
        # add button to predict genre from plot
        with gr.Column():
            btn_plot_genre = gr.Button("Predict Genre from Plot")
            # add output box for genre prediction from plot
            plot_genre_out = gr.Textbox(label="Predicted Genre from Plot")

    btn_plot_genre.click(fn=predict_genre_from_plot, inputs=plot_in, outputs=plot_genre_out)
    with gr.Row():
        nl_query = gr.Textbox(
        lines=2,
        placeholder="Describe the movie you are looking for...",
        label="Natural Language Movie Search"
        )

    with gr.Row():
        btn_search_nl = gr.Button("Search Movies")
        nl_results = gr.Textbox(
        label="Search Results",
        lines=12
        )

    btn_search_nl.click(
         fn=search_movies_nl,
         inputs=nl_query,
        outputs=nl_results
        )

app.launch(server_name="0.0.0.0", server_port=7860)