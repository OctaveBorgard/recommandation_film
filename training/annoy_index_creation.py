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

plots = df['movie_plot'].tolist()

print(f"Nombre de résumés à encoder : {len(plots)}")

batch_size = 32
text_embeddings_list = []

print("Encoding text plots...")
for i in range(0, len(plots), batch_size):
    batch_texts = plots[i:i + batch_size]
    
    text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings_list.append(text_features.cpu())

all_text_embeddings = torch.cat(text_embeddings_list).numpy()


dim = all_text_embeddings.shape[1]
text_index = AnnoyIndex(dim, metric="angular")


text_mapping = {}

for i, emb in enumerate(all_text_embeddings):
    text_index.add_item(i, emb)
    text_mapping[i] = {
        "poster_path": df.iloc[i]['movie_poster_path'],
        "plot": df.iloc[i]['movie_plot'],
        "category": df.iloc[i]['movie_category']
    }

text_index.build(20)

text_index_path = "text_index.ann"
text_mapping_path = "text_mapping.json"

text_index.save(text_index_path)
with open(text_mapping_path, 'w', encoding='utf-8') as f:
    json.dump(text_mapping, f, ensure_ascii=False, indent=4)

print(f"Index text saved : {text_index_path}")
print(f"Mapping text saved : {text_mapping_path}")