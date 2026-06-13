import os
import glob

# Correct 17-class mapping from data.yaml
correct_map = {
    'adansonia_not_dry': 0, 'adansonia_dry': 1,
    'acacia_not_dry': 2, 'acacia_dry': 3,
    'vachellia_not_dry': 4, 'vachellia_dry': 5,
    'senegalia_not_dry': 6, 'senegalia_dry': 7,
    'combretum_not_dry': 8, 'combretum_dry': 9,
    'brachystegia_not_dry': 10, 'brachystegia_dry': 11,
    'colophospermum_not_dry': 12, 'colophospermum_dry': 13,
    'ficus_not_dry': 14, 'ficus_dry': 15,
    'khaya_not_dry': 16, 'khaya_dry': 17,
    'macaranga_not_dry': 18, 'macaranga_dry': 19,
    'euphorbia_not_dry': 20, 'euphorbia_dry': 21,
    'aloe_not_dry': 22, 'aloe_dry': 23,
    'protea_not_dry': 24, 'protea_dry': 25,
    'erica_not_dry': 26, 'erica_dry': 27,
    'themeda_not_dry': 28, 'themeda_dry': 29,
    'andropogon_not_dry': 30, 'andropogon_dry': 31,
    'tamarix_not_dry': 32, 'tamarix_dry': 33
}

def fix_label_files(base_dir):
    label_files = glob.glob(os.path.join(base_dir, "labels/*/*.txt"))
    print(f"Discovered {len(label_files)} label files to verify and correct...")
    
    corrected_count = 0
    
    for lbl_path in label_files:
        filename = os.path.basename(lbl_path)
        # Find which species this belongs to by matching filename prefixes
        matched_species = None
        for k in correct_map.keys():
            sp_prefix = k.split('_')[0]
            if filename.startswith(f"{sp_prefix}_"):
                matched_species = sp_prefix
                break
        
        # Fallback for baobab duplicate (maps to adansonia)
        if filename.startswith("baobab_"):
            matched_species = "adansonia"
            
        if not matched_species:
            print(f"Warning: Could not match species for filename {filename}")
            continue
            
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
        if not lines:
            continue
            
        new_lines = []
        file_changed = False
        
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            old_class_id = int(parts[0])
            # Odd IDs represent dry, Even IDs represent not_dry in the alphabetical script run
            is_dry = (old_class_id % 2 != 0)
            dryness_suffix = "dry" if is_dry else "not_dry"
            
            correct_class_name = f"{matched_species}_{dryness_suffix}"
            correct_class_id = correct_map[correct_class_name]
            
            if old_class_id != correct_class_id:
                parts[0] = str(correct_class_id)
                file_changed = True
                
            new_lines.append(" ".join(parts) + "\n")
            
        if file_changed:
            with open(lbl_path, "w") as f:
                f.write("".join(new_lines))
            corrected_count += 1
            
    print(f"Correction complete! Recalibrated {corrected_count} label files to matching indices in data.yaml!")

if __name__ == "__main__":
    fix_label_files("datasets/yolo_flora")
