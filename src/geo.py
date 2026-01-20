import os
import json
import requests

def load_geojson(path="assets/india_states.geojson"):
    """Load GeoJSON file, download if missing"""
    try:
        # If file doesn't exist, download it
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(path, "wb") as f:
                    f.write(response.content)
                print(f"✅ GeoJSON downloaded to {path}")
            else:
                print(f"❌ Failed to download GeoJSON. Status: {response.status_code}")
                return None

        # Load the file
        with open(path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
        
        print(f"✅ GeoJSON loaded successfully with {len(geojson_data.get('features', []))} features")
        return geojson_data
        
    except Exception as e:
        print(f"❌ Error loading GeoJSON: {e}")
        return None


def map_state_names(df, state_col='state'):
    """Normalize state names to match GeoJSON properties"""
    
    # Comprehensive state name mapping
    state_map = {
        # Abbreviated forms
        'Uttar Prad': 'Uttar Pradesh',
        'Saharanpu': 'Saharanpur',
        'Pratapgar': 'Pratapgarh',
        'Muzaffarn': 'Muzaffarnagar',
        'Rae Bareli': 'Raebareli',
        
        # Common variations
        'Telengana': 'Telangana',
        'Chattisgarh': 'Chhattisgarh',
        'Orissa': 'Odisha',
        'Pondicherry': 'Puducherry',
        
        # Union Territories
        'Delhi': 'NCT of Delhi',
        'Andaman': 'Andaman and Nicobar Islands',
        'Andaman & Nicobar': 'Andaman and Nicobar Islands',
        'Andaman And Nicobar Islands': 'Andaman and Nicobar Islands',
        'Dadra Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
        'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
        'Dadra And Nagar Haveli And Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
        'Daman': 'Dadra and Nagar Haveli and Daman and Diu',
        'Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
        
        # J&K variations
        'Jammu & Kashmir': 'Jammu and Kashmir',
        'Jammu And Kashmir': 'Jammu and Kashmir',
        'Ladakh': 'Ladakh'
    }
    
    # Normalize: strip whitespace and title case
    df[state_col] = df[state_col].astype(str).str.strip().str.title()
    
    # Apply mapping
    df[state_col] = df[state_col].replace(state_map)
    
    return df


def get_geojson_property_key(geojson):
    """
    Detect which property key to use for state names in GeoJSON
    Common options: ST_NM, NAME, name, state, State
    """
    if not geojson or 'features' not in geojson:
        return None
    
    # Check first feature for available properties
    if len(geojson['features']) > 0:
        props = geojson['features'][0].get('properties', {})
        
        # Priority order of property keys
        possible_keys = ['ST_NM', 'NAME', 'name', 'NAME_1', 'state', 'State']
        
        for key in possible_keys:
            if key in props:
                print(f"✅ Using GeoJSON property key: {key}")
                return f"properties.{key}"
    
    print("❌ Could not detect GeoJSON property key")
    return None