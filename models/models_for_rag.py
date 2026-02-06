from fileinput import filename
import torch
import clip
import json
import os
from PIL import Image
from annoy import AnnoyIndex
import matplotlib.pyplot as plt

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

def search_hybrid_split(query_text):
    # Encodage de la requête
    text_tokens = clip.tokenize([query_text], truncate=True).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    query_vector = text_features.cpu().numpy()[0]

    final_results = []

    # Recherche IMAGE
    ids_img, dists_img = u_image.get_nns_by_vector(query_vector, 2, include_distances=True)
    for idx, dist in zip(ids_img, dists_img):
        filename = Path(mapping_image[str(idx)]).name
        full_path = BASE_POSTER_DIR / filename

        extra_info = path_to_data.get(filename, {})
        
        final_results.append({
            "score": (2 - dist**2) / 2 * 100,
            "type": "IMAGE",
            "img_path": full_path,
            "plot": extra_info.get('plot', "Résumé non disponible")
        })

    # Recherche TEXTE
    ids_txt, dists_txt = u_text.get_nns_by_vector(query_vector, 2, include_distances=True)
    for idx, dist in zip(ids_txt, dists_txt):
        data = mapping_text[str(idx)]
        relative_path = data['poster_path'].lstrip('/')
        full_path = os.path.join(POSTER_ROOT, relative_path)

        
        final_results.append({
            "score": (2 - dist**2) / 2 * 100,
            "type": "TEXTE",
            "img_path": full_path,
            "plot": data.get('plot', "Résumé non disponible")
        })

    fig, axes = plt.subplots(1, 4, figsize=(22, 8))
    print(f"\n--- Résultats pour : '{query_text}' ---")

    for i, res in enumerate(final_results):
        print(f"{i+1}. [{res['type']}] Score: {res['score']:.2f}%")
        
        try:
            img = Image.open(full_path)


            axes[i].imshow(img)
            axes[i].set_title(f"Source: {res['type']}\n{res['score']:.1f}%", fontsize=11, fontweight='bold')
            
            plot_text = res['plot'][:80] + "..." if len(res['plot']) > 80 else res['plot']
            axes[i].set_xlabel(plot_text, fontsize=8, wrap=True)
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Erreur image\n{os.path.basename(res['img_path'])}", ha='center')
            print(f"   /!\\ Erreur sur : {res['img_path']}")
        
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Test
#search_hybrid_split("animated cartoons in Arabia")


def search_hybrid_api(query_text, top_k=2):
    # Encodage de la requête utilisateur
    text_tokens = clip.tokenize([query_text], truncate=True).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    query_vector = text_features.cpu().numpy()[0]
    results = []

    # Dossier racine tel que défini dans le Dockerfile
    base_path = "/app/content/sorted_movie_posters_paligema"

    # ---------- IMAGE SEARCH ----------
    ids_img, dists_img = u_image.get_nns_by_vector(
        query_vector, top_k, include_distances=True
    )

    for idx, dist in zip(ids_img, dists_img):
        raw_path = mapping_image[str(idx)]
        
        # 1. Conversion des \ en / (Windows -> Linux)
        clean_path = raw_path.replace('\\', '/')
        
        # 2. Extraction des deux derniers segments (ex: horror/69766.jpg)
        path_parts = clean_path.split('/')
        relative_filename = "/".join(path_parts[-2:]) 
        
        # 3. Reconstruction du chemin absolu pour Docker
        full_path = os.path.join(base_path, relative_filename)
        
        filename = os.path.basename(clean_path)
        extra_info = path_to_data.get(filename, {})

        results.append({
            "source": "image",
            "score": float((2 - dist**2) / 2),
            "poster_path": full_path,
            "plot": extra_info.get("plot", "Résumé non disponible")
        })

    # ---------- TEXT SEARCH ----------
    ids_txt, dists_txt = u_text.get_nns_by_vector(
        query_vector, top_k, include_distances=True
    )

    for idx, dist in zip(ids_txt, dists_txt):
        data = mapping_text[str(idx)]
        
        # Application du même traitement de nettoyage
        raw_path = data.get("poster_path", "")
        clean_path = raw_path.replace('\\', '/')
        path_parts = clean_path.split('/')
        relative_filename = "/".join(path_parts[-2:])
        full_path = os.path.join(base_path, relative_filename)

        results.append({
            "source": "text",
            "score": float((2 - dist**2) / 2),
            "poster_path": full_path,
            "plot": data.get("plot", "Résumé non disponible")
        })
    
    return results