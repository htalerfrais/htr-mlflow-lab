#!/usr/bin/env python3
"""
Script simple qui ajoute 99 à tous les indices (id) des lignes dans un fichier JSON.
1 -> 100, 2 -> 101, etc.
"""

import json
import sys
from pathlib import Path


def add_offset_to_ids(json_path, offset=99):
    """
    Ajoute un offset à tous les ids dans le fichier JSON.
    
    Args:
        json_path: Chemin vers le fichier JSON
        offset: Valeur à ajouter aux ids (par défaut 99)
    """
    # Lire le fichier JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Modifier tous les ids
    if 'samples' in data:
        for sample in data['samples']:
            if 'id' in sample:
                sample['id'] += offset
    
    # Sauvegarder le fichier modifié
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Offset de {offset} ajouté à tous les ids dans {json_path}")


if __name__ == "__main__":
    # Chemin par défaut
    default_path = Path(__file__).parent.parent / "data_local" / "perso_dataset" / "hector_pages_lines_3.2" / "tomerge" / "gt_hector_pages_lines.json"
    
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = default_path
    
    if not json_path.exists():
        print(f"Erreur: Le fichier {json_path} n'existe pas.")
        sys.exit(1)
    
    add_offset_to_ids(json_path)
