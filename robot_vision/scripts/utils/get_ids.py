import requests

genera = [
    "Adansonia", "Acacia", "Vachellia", "Senegalia", 
    "Combretum", "Brachystegia", "Colophospermum", 
    "Ficus", "Khaya", "Macaranga", 
    "Euphorbia", "Aloe", "Protea", "Erica", 
    "Themeda", "Andropogon", "Tamarix"
]

print("Fetching GBIF IDs...")
for genus in genera:
    resp = requests.get(f"https://api.gbif.org/v1/species/match?name={genus}&rank=GENUS")
    if resp.status_code == 200:
        data = resp.json()
        key = data.get('usageKey', 'NOT FOUND')
        print(f"'{genus.lower()}': {key},")
    else:
        print(f"Failed {genus}")

print("\nFetching iNaturalist IDs...")
for genus in genera:
    resp = requests.get(f"https://api.inaturalist.org/v1/taxa?q={genus}&rank=genus")
    if resp.status_code == 200:
        data = resp.json()
        results = data.get('results', [])
        if results:
            # Try to find exact match
            for r in results:
                if r['name'].lower() == genus.lower():
                    print(f"'{genus.lower()}': {r['id']},")
                    break
            else:
                print(f"'{genus.lower()}': {results[0]['id']},")
        else:
            print(f"'{genus.lower()}': NOT FOUND,")
    else:
        print(f"Failed {genus}")
