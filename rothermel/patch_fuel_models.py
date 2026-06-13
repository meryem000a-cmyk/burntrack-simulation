"""
patch_fuel_models.py
====================
Patch pour corriger les valeurs mx des fuel models africains.
Exécute : python patch_fuel_models.py
"""

import re

# Corrections : {code: nouveau_mx}
CORRECTIONS = {
    'AF_STEPPE': 20,
    'AF_SAHEL_GRASS': 20,
    'AF_SAHEL_WOODED': 22,
    'AF_SUDAN_GRASS': 20,
    'AF_SUDAN_WOODED': 22,
    'AF_CEREALES': 18,
    'AF_RANGE_DEGRADED': 15,
    'AF_STEPPE_DENSE': 22,
    'AF_GRASSLAND_FERTILE': 25,
    'AF_FYNBOS': 18,
    'AF_MIOMBO': 22,
    'AF_NAMA_KAROO': 18,
    'AF_THICKET': 25,
    'AF_DRY_FOREST': 22,
    'AF_SPINY_FOREST': 20,
    'AF_HUMID_GRASS': 25,
    'AF_RAVINALA': 25,
    'AF_ACACIA_SAVANNA': 22,
    'AF_RED_OAT_GRASS': 22,
    'AF_WHISTLING_THORN': 20,
    'AF_BLACK_COTTON': 25,
    'AF_MONTANE_GRASS': 30,
    'AF_MONTANE_FOREST': 35,
    'AF_SWAMP_GRASS': 40,
    'AF_PAPYRUS': 45,
}

def patch_file(filepath='fuel_models.py'):
    with open(filepath, 'r') as f:
        content = f.read()
    
    modified = False
    for code, new_mx in CORRECTIONS.items():
        # Pattern : code="AF_STEPPE", ..., mx=12, ...
        pattern = rf'(code="{code}"[^\n]*)mx=\d+'
        replacement = rf'\1mx={new_mx}'
        
        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            print(f"✅ {code}: mx corrigé → {new_mx}")
            content = new_content
            modified = True
        else:
            print(f"⚠️ {code}: non trouvé dans le fichier")
    
    if modified:
        # Backup
        import shutil
        shutil.copy(filepath, filepath + '.backup')
        print(f"\n💾 Backup créé : {filepath}.backup")
        
        # Écriture
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✅ Fichier corrigé : {filepath}")
    else:
        print("❌ Aucune modification effectuée")

if __name__ == "__main__":
    patch_file()