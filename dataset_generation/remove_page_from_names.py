#!/usr/bin/env python3
"""
Script qui garde uniquement la partie à partir de "ligne" ou "line" dans les noms de fichiers .jpg.
page_9_line_0181.jpg -> line_0181.jpg
"""

import re
import sys
from pathlib import Path


def remove_page_from_names(folder_path):
    """
    Cherche "ligne" ou "line" dans le nom du fichier et garde uniquement cette partie et ce qui suit.
    
    Args:
        folder_path: Chemin vers le dossier contenant les images
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Erreur: Le dossier {folder_path} n'existe pas.")
        sys.exit(1)
    
    # Pattern pour trouver "ligne" ou "line" (insensible à la casse)
    pattern = re.compile(r'.*?(ligne|line)', re.IGNORECASE)
    
    # Collecter tous les fichiers à renommer
    files_to_rename = []
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.jpg':
            old_name = file_path.name
            
            # Chercher "ligne" ou "line" dans le nom
            match = pattern.search(old_name)
            if match:
                # Trouver la position de "ligne" ou "line"
                start_pos = match.start(1)  # Position du groupe capturé (ligne ou line)
                # Garder tout ce qui suit à partir de cette position
                new_name = old_name[start_pos:]
                files_to_rename.append((file_path, new_name))
            else:
                print(f"⚠ Pas de 'ligne' ou 'line' trouvé dans {old_name}, on passe")
    
    # Renommer les fichiers
    renamed_count = 0
    for old_path, new_name in files_to_rename:
        new_path = old_path.parent / new_name
        if new_path.exists():
            print(f"⚠ Attention: {new_name} existe déjà, on passe {old_path.name}")
            continue
        old_path.rename(new_path)
        renamed_count += 1
        print(f"  {old_path.name} -> {new_name}")
    
    print(f"✓ {renamed_count} fichier(s) renommé(s)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = Path(sys.argv[1])
    else:
        print("Usage: python remove_page_from_names.py <chemin_vers_dossier>")
        print("Exemple: python remove_page_from_names.py data_local/perso_dataset/hector_pages_lines_3.2/tomerge/lines_out_sorted")
        sys.exit(1)
    
    remove_page_from_names(folder_path)
