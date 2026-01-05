#%%
import requests

API_URL = "http://localhost:5075/predict"
IMAGE_PATH = r"content/sorted_movie_posters_paligema/action/110.jpg"
 

with open(IMAGE_PATH, "rb") as f:
    response = requests.post(API_URL, files={"file": f})

print("Status:", response.status_code)
print("Response:", response.json())

# %%
