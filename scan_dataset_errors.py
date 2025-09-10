import re

def scan_dataset_errors(file_path):
    errors = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Check for JSON array encapsulation
    if not lines[0].strip().startswith('['):
        errors.append('Le fichier ne commence pas par un [ (crochet ouvrant de tableau JSON).')
    if not lines[-1].strip().endswith(']'):
        errors.append('Le fichier ne se termine pas par un ] (crochet fermant de tableau JSON).')

    # Check for missing commas between objects and isolated braces
    for i, line in enumerate(lines):
        # Detect isolated closing or opening braces
        if line.strip() == '{' or line.strip() == '}':
            errors.append(f'Ligne {i+1}: accolade isolée détectée.')
        # Detect missing comma between objects
        if i > 0 and lines[i-1].strip().endswith('}') and line.strip().startswith('{'):
            if not lines[i-1].strip().endswith('},'):
                errors.append(f'Ligne {i}: virgule manquante entre deux objets JSON.')
        # Detect trailing comma before closing array
        if line.strip() == '},' and i+1 < len(lines) and lines[i+1].strip() == ']':
            errors.append(f'Ligne {i+1}: virgule en trop avant la fermeture du tableau JSON.')

    # Try to parse as JSON
    import json
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f'Erreur JSONDecodeError: {e}')

    if not errors:
        print('Aucune erreur détectée. Le fichier semble être un JSON valide.')
    else:
        print('Erreurs détectées dans le fichier DATASET:')
        for err in errors:
            print(err)

if __name__ == '__main__':
    scan_dataset_errors('DATASET')

