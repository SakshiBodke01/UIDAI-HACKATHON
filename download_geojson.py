print("🚀 Script started")

import requests
import os

os.makedirs("assets", exist_ok=True)

url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
save_path = "assets/india_states.geojson"

response = requests.get(url)

print("Status code:", response.status_code)

if response.status_code == 200:
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"✅ GeoJSON file saved to {save_path}")
else:
    print(f"❌ Failed to download. Status code: {response.status_code}")
