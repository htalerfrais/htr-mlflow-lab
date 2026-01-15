#!/usr/bin/env python3
"""
Script qui ajoute 99 à tous les indices finaux des images dans un dossier.
page_*_line_0001.jpg -> page_*_line_0100.jpg
"""

import re
import sys
from pathlib import Path


def rename_images_with_offset(folder_path, offset=99):
    """
    Renomme les fichiers images en ajoutant un offset à l'indice de ligne.
    
    Args:
        folder_path: Chemin vers le dossier contenant les images
        offset: Valeur à ajouter aux indices (par défaut 99)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Erreur: Le dossier {folder_path} n'existe pas.")
        sys.exit(1)
    
    # Pattern pour matcher: page_X_line_YYYY.jpg
    pattern = re.compile(r'^page_(\d+)_line_(\d+)\.jpg$')
    
    # Collecter tous les fichiers à renommer
    files_to_rename = []
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.jpg':
            match = pattern.match(file_path.name)
            if match:
                page_num = match.group(1)
                line_num = int(match.group(2))
                new_line_num = line_num + offset
                new_name = f"page_{page_num}_line_{new_line_num:04d}.jpg"
                files_to_rename.append((file_path, new_name))
    
    # Trier par nom pour éviter les conflits lors du renommage
    files_to_rename.sort(key=lambda x: x[0].name, reverse=True)
    
    # Renommer les fichiers
    renamed_count = 0
    for old_path, new_name in files_to_rename:
        new_path = old_path.parent / new_name
        if new_path.exists():
            print(f"⚠ Attention: {new_name} existe déjà, on passe {old_path.name}")
            continue
        old_path.rename(new_path)
        renamed_count += 1
    
    print(f"✓ {renamed_count} fichier(s) renommé(s) avec un offset de {offset}")


if __name__ == "__main__":
    # Chemin par défaut
    default_path = Path(__file__).parent.parent / "data_local" / "perso_dataset" / "hector_pages_lines_3.2" / "tomerge" / "lines_out_sorted"
    
    if len(sys.argv) > 1:
        folder_path = Path(sys.argv[1])
    else:
        folder_path = default_path
    
    rename_images_with_offset(folder_path)
