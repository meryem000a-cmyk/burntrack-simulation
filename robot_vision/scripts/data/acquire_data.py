import os
import requests
import time
from pygbif import occurrences as occ
from pyinaturalist import get_observations

def acquire_gbif_african_flora(taxon_key, output_dir, max_records=1500):
    """
    Downloads occurrences with images for a specific African taxon using pagination.
    """
    os.makedirs(output_dir, exist_ok=True)
    offset = 0
    limit = 300
    downloaded = 0

    while downloaded < max_records:
        res = occ.search(
            taxonKey=taxon_key, 
            continent='africa', 
            mediatype='StillImage', 
            limit=limit, 
            offset=offset
        )
        
        results = res.get('results', [])
        if not results:
            break
            
        for record in results:
            media = record.get('media', [])
            for m in media:
                if m.get('type') == 'StillImage' and 'identifier' in m:
                    img_url = m['identifier']
                    img_name = f"{record['key']}.jpg"
                    img_path = os.path.join(output_dir, img_name)
                    
                    if not os.path.exists(img_path):
                        try:
                            response = requests.get(img_url, timeout=10)
                            if response.status_code == 200:
                                with open(img_path, 'wb') as f:
                                    f.write(response.content)
                                downloaded += 1
                                if downloaded % 100 == 0:
                                    print(f"  Downloaded {downloaded}/{max_records} GBIF images...")
                        except requests.exceptions.RequestException:
                            continue
                
                if downloaded >= max_records:
                    break
        offset += limit

def acquire_inaturalist_flora(taxon_id, place_id, output_dir, max_records=1500):
    """
    Downloads research-grade occurrences from iNaturalist with pagination.
    """
    os.makedirs(output_dir, exist_ok=True)
    search_params = {
        "taxon_id": taxon_id,
        "place_id": place_id, 
        "quality_grade": "research",
        "has_photos": True,
        "per_page": 200
    }
    
    downloaded = 0
    page = 1
    
    while downloaded < max_records:
        search_params["page"] = page
        results = get_observations(**search_params).get("results", [])
        
        if not results:
            break
            
        for obs in results:
            for photo in obs.get("photos", []):
                url = photo.get("url").replace("square", "medium") 
                img_path = os.path.join(output_dir, f"inat_{obs['id']}.jpg")
                
                if not os.path.exists(img_path):
                    try:
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            with open(img_path, 'wb') as f:
                                f.write(response.content)
                            downloaded += 1
                            if downloaded % 100 == 0:
                                print(f"  Downloaded {downloaded}/{max_records} iNaturalist images...")
                    except Exception as e:
                        pass
                
                time.sleep(1.2)
                
                if downloaded >= max_records:
                    break
            if downloaded >= max_records:
                break
        page += 1

if __name__ == '__main__':
    # Define the output directory
    train_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'african_flora', 'images', 'train')
    print(f"Starting large-scale data acquisition. Images will be saved to: {train_dir}")
    
    # Defining a Pan-Biome diverse set of African Flora (17 Genera)
    # GBIF Keys (Genera)
    taxa_gbif = {
        'adansonia': 3152213,
        'acacia': 2978223,
        'vachellia': 8142432,
        'senegalia': 2978223,  # Using broader acacia key as fallback
        'combretum': 2986357,
        'brachystegia': 2952646,
        'colophospermum': 2974566,
        'ficus': 3097368,
        'khaya': 3190507,
        'macaranga': 3073879,
        'euphorbia': 11397237,
        'aloe': 2770879,
        'protea': 5428464,
        'erica': 2874415,
        'themeda': 2703466,
        'andropogon': 2706077,
        'tamarix': 2874694
    }
    
    # iNaturalist IDs (Genera)
    taxa_inat = {
        'adansonia': 81507,
        'acacia': 47452,
        'vachellia': 72418,
        'senegalia': 72356,
        'combretum': 81496,
        'brachystegia': 139468,
        'colophospermum': 428750,
        'ficus': 50999,
        'khaya': 126187,
        'macaranga': 133556,
        'euphorbia': 51822,
        'aloe': 71956,
        'protea': 129714,
        'erica': 55776,
        'themeda': 155853,
        'andropogon': 71966,
        'tamarix': 51305
    }
    
    # Increase records to 1500 for a production-level dataset
    target_records = 1500
    
    print("\n--- Fetching Diverse GBIF Data ---")
    for name, key in taxa_gbif.items():
        print(f"Fetching {name.capitalize()} images from GBIF...")
        acquire_gbif_african_flora(taxon_key=key, output_dir=os.path.join(train_dir, f'{name}_raw'), max_records=target_records)
        
    print("\n--- Fetching Diverse iNaturalist Data ---")
    for name, id_val in taxa_inat.items():
        print(f"Fetching {name.capitalize()} images from iNaturalist...")
        acquire_inaturalist_flora(taxon_id=id_val, place_id=97394, output_dir=os.path.join(train_dir, f'{name}_raw'), max_records=target_records)
    
    print("\nMassive data acquisition complete! You will need to manually curate these images into 'dry' and 'not_dry' folders/labels.")
