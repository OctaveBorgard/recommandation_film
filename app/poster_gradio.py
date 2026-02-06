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
        return None, "Error calling search API"

    data = response.json()
    results = data.get("results", [])

    if not results:
        return None, "No results found."

    gallery_items = []
    text_details = ""

    for i, res in enumerate(results, 1):
        # 1. Préparation de l'image pour la galerie
        img_path = res['poster_path']
        # Si le chemin commence par /app/ dans le container API, 
        # assurez-vous qu'il est identique dans le container Gradio
        if os.path.exists(img_path):
            label = f"{i}. {res['source']} (Score: {res['score']:.2f})"
            gallery_items.append((img_path, label))
        
        # 2. Préparation du texte complet
        text_details += (
            f"--- Proposition {i} ({res['source'].upper()}) ---\n"
            f"Similarity: {res['score']:.2f}\n"
            f"SYNOPSIS: {res['plot']}\n\n"
        )

    return gallery_items, text_details


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

    btn_search_nl = gr.Button("Search Movies", variant="primary")

    with gr.Row():
        # Galerie pour afficher les 4 affiches
        nl_gallery = gr.Gallery(
            label="Movie Posters", 
            show_label=True, 
            elem_id="gallery", 
            columns=[2], 
            rows=[2], 
            object_fit="contain", 
            height="auto"
        )
        # Textbox pour les synopsis complets
        nl_results_text = gr.Textbox(
            label="Detailed Plots",
            lines=15,
            interactive=False
        )

    # Mise à jour du clic : deux sorties (Gallery + Textbox)
    btn_search_nl.click(
        fn=search_movies_nl,
        inputs=nl_query,
        outputs=[nl_gallery, nl_results_text]
    )

app.launch(server_name="0.0.0.0", server_port=7860)