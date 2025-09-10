import json

# Nom du fichier d'entrée (ton fichier existant)
input_file = "dataset_export.json"

# Nom du fichier de sortie (format Alpaca)
output_file = "export_export_alpaca.json"

# Charger les données existantes
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Conversion au format Alpaca
alpaca_data = []
for entry in data:
    alpaca_entry = {
        "instruction": entry.get("user_input", "").strip(),
        "input": "",
        "output": entry.get("assistant_output", "").strip()
    }
    alpaca_data.append(alpaca_entry)

# Sauvegarder le fichier converti
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

print(f"✅ Conversion terminée ! Fichier enregistré sous : {output_file}")
