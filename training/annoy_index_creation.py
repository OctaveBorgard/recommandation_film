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
root_dir = Path("/content") # chemin à spécifier
image_paths = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        # filtre uniquement les extensions d'images
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(root, file))

print(f"Nombre d'images trouvées : {len(image_paths)}")

batch_size = 32
image_embeddings_list = []
print("Encodage des images en cours...")
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

# Création du dictionnaire de correspondance (Mapping)
# Index Annoy (int) -> Chemin de l'image (str)
mapping = {}

for i, emb in enumerate(all_embeddings):
    index.add_item(i, emb)
    mapping[i] = image_paths[i]

index.build(20)

# --- SAUVEGARDE ---
index_path = "image_index.ann"
mapping_path = "image_mapping.json"

index.save(index_path)
with open(mapping_path, 'w') as f:
    json.dump(mapping, f)

print(f"Index sauvegardé dans {index_path}")
print(f"Mapping sauvegardé dans {mapping_path}")