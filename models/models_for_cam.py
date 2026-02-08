import torch
import clip
import json
import os
from PIL import Image
from annoy import AnnoyIndex
from pathlib import Path

BASE_POSTER_DIR = Path("/app/content/sorted_movie_posters_paligema")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
dim = 512 

POSTER_ROOT = "/app/content/sorted_movie_posters_paligema/"


u_image = AnnoyIndex(dim, 'angular')
u_image.load("/app/exp/image_index.ann")

with open("/app/exp/image_mapping.json", 'r') as f:
    mapping_image = json.load(f)

u_text = AnnoyIndex(dim, 'angular')
u_text.load("/app/exp/text_index.ann")

with open("/app/exp/text_mapping.json", 'r', encoding="utf-8") as f:
    mapping_text = json.load(f)


path_to_data = {os.path.basename(data['poster_path']): data for data in mapping_text.values()}

def search_by_image(input_image_path):
    """
    Prend une image en entrée et retourne le résultat le plus similaire 
    parmi la base de données d'images.
    """
    try:
        image = preprocess(Image.open(input_image_path)).unsqueeze(0).to(device)
    except Exception as e:
        return {"error": f"Impossible de charger l'image : {e}"}

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    query_vector = image_features.cpu().numpy()[0]


    idx, dist = u_image.get_nns_by_vector(query_vector, 1, include_distances=True)
    
    if not idx:
        return {"error": "Aucun résultat trouvé"}

    res_idx = idx[0]
    res_dist = dist[0]
    
    raw_path = mapping_image[str(res_idx)]
    clean_path = raw_path.replace('\\', '/')
    path_parts = clean_path.split('/')
    relative_filename = "/".join(path_parts[-2:])
    
    base_path = "/app/content/sorted_movie_posters_paligema"
    full_path = os.path.join(base_path, relative_filename)
    
    filename = os.path.basename(clean_path)
    extra_info = path_to_data.get(filename, {})

    return {
        "source": "image_query",
        "score": float((2 - res_dist**2) / 2),
        "poster_path": full_path,
        "plot": extra_info.get("plot", "Résumé non disponible")
    }