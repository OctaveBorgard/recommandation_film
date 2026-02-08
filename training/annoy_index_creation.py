import os
import json
import torch
import clip
from PIL import Image
from annoy import AnnoyIndex
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# collecte des chemins d'images
# dossier racine des posters
root_dir = Path("content/sorted_movie_posters_paligema")

image_paths = []


for root, dirs, files in os.walk(root_dir):
    for file in files:
        # filtre uniquement les extensions d'images
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(root, file))

print(f"Number of images : {len(image_paths)}")

batch_size = 32
image_embeddings_list = []
print("Encoding...")
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    batch_images = []
    
    for path in batch_paths:
        img = preprocess(Image.open(path).convert("RGB"))
        batch_images.append(img)
    
    batch_tensor = torch.stack(batch_images).to(device)
    
    with torch.no_grad():
        embeddings = model.encode_image(batch_tensor)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        image_embeddings_list.append(embeddings.cpu())

all_embeddings = torch.cat(image_embeddings_list).numpy()

# 4. index Annoy
dim = all_embeddings.shape[1]
index = AnnoyIndex(dim, metric="angular")

mapping = {}

for i, emb in enumerate(all_embeddings):
    index.add_item(i, emb)
    mapping[i] = image_paths[i]

index.build(20)

index_path = "image_index.ann"
mapping_path = "image_mapping.json"

index.save(index_path)
with open(mapping_path, 'w') as f:
    json.dump(mapping, f)

print(f"Index saved {index_path}")
print(f"Mapping saved {mapping_path}")


# ----------------------Text--------------------------
import pandas as pd
csv_path = "content/movie_plots.csv"
df = pd.read_csv(csv_path)

def get_text_chunks(text, chunk_size=50, overlap=10):
    """Découpe un texte en morceaux de 'chunk_size' mots avec un 'overlap'."""
    words = text.split()
    chunks = []
    if len(words) <= chunk_size:
        return [text]
    
    current = 0
    while current < len(words):
        end = min(len(words), current + chunk_size)
        chunk = " ".join(words[current:end])
        chunks.append(chunk)
        if end == len(words):
            break
        current += chunk_size - overlap
    return chunks

# Préparation des données pour l'indexation
all_chunks = []
chunk_to_original_index = []

print("Preparation des chunks...")
for idx, row in df.iterrows():
    plot_chunks = get_text_chunks(str(row['movie_plot']), chunk_size=40, overlap=10)
    for c in plot_chunks:
        all_chunks.append(c)
        chunk_to_original_index.append(idx) # Garde la trace de la ligne originale

print(f"Nombre total de chunks à encoder : {len(all_chunks)}")

# Encodage par batch
batch_size = 32
text_embeddings_list = []

print("Encoding text chunks...")
for i in range(0, len(all_chunks), batch_size):
    batch_texts = all_chunks[i:i + batch_size]
    
    # truncate=True est gardé par sécurité pour CLIP (limite 77 tokens), 
    # mais le chunking en amont réduit grandement la perte d'info.
    text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings_list.append(text_features.cpu())

all_text_embeddings = torch.cat(text_embeddings_list).numpy()

# Création de l'index Annoy
dim = all_text_embeddings.shape[1]
text_index = AnnoyIndex(dim, metric="angular")
text_mapping = {}

for i, emb in enumerate(all_text_embeddings):
    text_index.add_item(i, emb)
    
    # On récupère les infos originales via l'index de référence
    original_row_idx = chunk_to_original_index[i]
    original_row = df.iloc[original_row_idx]
    
    # Le format reste STRICTEMENT identique à votre demande
    text_mapping[i] = {
        "poster_path": original_row['movie_poster_path'],
        "plot": original_row['movie_plot'], # On garde le plot complet pour l'affichage
        "category": original_row['movie_category']
    }

text_index.build(20)

# Sauvegarde
text_index_path = "text_index.ann"
text_mapping_path = "text_mapping.json"

text_index.save(text_index_path)
with open(text_mapping_path, 'w', encoding='utf-8') as f:
    json.dump(text_mapping, f, ensure_ascii=False, indent=4)

print(f"Index text saved : {text_index_path}")
print(f"Mapping text saved : {text_mapping_path}")