import gradio as gr
import requests
import os
import io

# Configuration des URLs
API_BASE_URL = "http://api:5075"
API_URL = f"{API_BASE_URL}/predict"
API_VALIDATE_URL = f"{API_BASE_URL}/validate-poster"
API_PREDICT_PLOT_URL = f"{API_BASE_URL}/predict_plot"
API_SEARCH_MOVIE_URL = f"{API_BASE_URL}/search"
API_SEARCH_IMAGE_URL = f"{API_BASE_URL}/search-image"

# --- Fonctions (inchangées mais regroupées) ---

def predict_genre_poster(image):
    if image is None: return "Please upload an image"
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    response = requests.post(API_URL, files={"file":("input.png", img_byte_arr,"image/png")})
    return response.json().get("genre", "Unknown") if response.status_code == 200 else "Error"

def validate_poster(image):
    if image is None: return "Please upload an image"
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    response = requests.post(API_VALIDATE_URL, files={"file": ("input.png", img_byte_arr, "image/png")})
    if response.status_code != 200: return "Error calling validation API"
    result = response.json()
    return f"Poster detected" if result['is_poster'] else "Not a poster"

def predict_genre_from_plot(plot):
    if not plot: return "Please enter text"
    response = requests.post(API_PREDICT_PLOT_URL, json={"text": plot})
    return response.json().get("genre", "Unknown") if response.status_code == 200 else "Error"

def search_movies_nl(query):
    if not query: return None, "Please enter a query"
    response = requests.post(API_SEARCH_MOVIE_URL, json={"query": query})
    if response.status_code != 200: return None, "Error calling search API"
    
    results = response.json().get("results", [])
    if not results: return None, "No results found."

    gallery_items = []
    text_details = ""
    for i, res in enumerate(results, 1):
        img_path = res['poster_path']
        if os.path.exists(img_path):
            gallery_items.append((img_path, f"{res['source']} (Score: {res['score']:.2f})"))
        
        text_details += f"### {i}. {res['source'].upper()} (Match: {res['score']*100:.1f}%)\n{res['plot']}\n\n---\n"
    return gallery_items, text_details


def search_by_camera(image):
    if image is None: return None, "Please capture an image"
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    response = requests.post(
        API_SEARCH_IMAGE_URL, 
        files={"file": ("input.png", img_byte_arr, "image/png")}
    )
    
    if response.status_code != 200: return None, "Error calling API"
    
    results = response.json().get("results", [])
    if not results: return None, "No similar movie found."

    res = results[0]
    img_path = res['poster_path']
    details = f"### Match Found: {res['score']*100:.1f}%\n\n**Plot:** {res['plot']}"
    
    return img_path, details

# --- Interface Graphique ---

with gr.Blocks(theme=gr.themes.Soft(), title="Movie AI Suite") as app:
    gr.Markdown("# Movie Analysis & Search AI")
    gr.Markdown("Identify genres from posters or plots, and search movies using natural language.")

    with gr.Tabs():
        
        # ONGLET 1 : ANALYSE D'IMAGE
        with gr.TabItem("Poster Analysis"):
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="pil", label="Upload Movie Poster")
                    with gr.Row():
                        btn_validate = gr.Button("Is it a Poster?", variant="secondary")
                        btn_genre = gr.Button("Predict Genre", variant="primary")
                
                with gr.Column():
                    valid_out = gr.Label(label="Validation Result")
                    genre_out = gr.Textbox(label="Predicted Genre", placeholder="Result will appear here...")

        # ONGLET 2 : ANALYSE DE SYNOPSIS
        with gr.TabItem("Plot Analysis"):
            with gr.Row():
                with gr.Column():
                    plot_in = gr.Textbox(lines=8, placeholder="Paste the movie script or plot here...", label="Movie Plot")
                    btn_plot_genre = gr.Button("Analyze Plot Genre", variant="primary")
                with gr.Column():
                    plot_genre_out = gr.Label(label="Detected Genre")

        # ONGLET 3 : RECHERCHE SÉMANTIQUE
        with gr.TabItem("Smart Search"):
            nl_query = gr.Textbox(
                lines=2, 
                placeholder="Ex: A sci-fi movie set in space with a lonely robot...", 
                label="Search by Description"
            )
            btn_search_nl = gr.Button("Find Movies", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=2):
                    nl_gallery = gr.Gallery(
                        label="Matching Posters", 
                        columns=2, 
                        rows=2, 
                        object_fit="contain", 
                        height=600
                    )
                with gr.Column(scale=3):
                    nl_results_text = gr.Markdown(label="Detailed Synopses")
        # ONGLET 4 : RECHERCHE IMAGE
        with gr.TabItem("Camera Search"):
            gr.Markdown("### Take a photo of you and find wich movie you are !")
            with gr.Row():
                with gr.Column() as col1: # On définit col1 ici
                    cam_in = gr.Image(sources=["webcam"], type="pil", label="Camera")
                    btn_cam = gr.Button("Search from Photo", variant="primary")
                with gr.Column() as col2: # On définit col2 ici
                    cam_out_img = gr.Image(label="Found Poster")
                    cam_out_plot = gr.Markdown(label="Match Details")

    # --- Events ---
    btn_genre.click(fn=predict_genre_poster, inputs=image_in, outputs=genre_out)
    btn_validate.click(fn=validate_poster, inputs=image_in, outputs=valid_out)
    btn_plot_genre.click(fn=predict_genre_from_plot, inputs=plot_in, outputs=plot_genre_out)
    btn_search_nl.click(fn=search_movies_nl, inputs=nl_query, outputs=[nl_gallery, nl_results_text])
    btn_cam.click(fn=search_by_camera, inputs=cam_in, outputs=[cam_out_img, cam_out_plot])

app.launch(server_name="0.0.0.0", server_port=7860)
