import json

# Définir les chemins des fichiers
input_path = 'DATASET'
output_path = 'dataset_export.json'

# Charger le contenu du fichier DATASET
with open(input_path, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# Exporter le contenu en JSON formaté
with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=2)

print(f"Export terminé : {output_path}")

