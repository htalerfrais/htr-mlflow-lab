import os
import re

# have to modify the path to adapt to where the images are stored
path = "."

def get_sort_key(filename):
    """
    Extrait (numéro_page, numéro_ligne) pour un tri parfait.
    Exemple: 'page_2_line_89.png' -> (2, 89)
    """
    # On cherche tous les nombres dans le nom du fichier
    numbers = re.findall(r'\d+', filename)
    if len(numbers) >= 2:
        return (int(numbers[0]), int(numbers[1]))
    return (0, 0)

# 1. Lister les fichiers et les trier par Page puis par Ligne
files = [f for f in os.listdir(path) if "page_" in f and "line_" in f and f.endswith(".png")]
files.sort(key=get_sort_key)

# 2. Première passe : Renommer vers des noms temporaires
# On stocke aussi la structure 'page_X_' pour la réutiliser
temp_files = []
for index, filename in enumerate(files, start=1):
    temp_name = f"temp_process_{index}.tmp"
    
    # Extraire le préfixe de la page (ex: "page_2_")
    match_page = re.search(r'(page_\d+_)', filename)
    page_prefix = match_page.group(1) if match_page else "page_0_"
    
    os.rename(os.path.join(path, filename), os.path.join(path, temp_name))
    temp_files.append((temp_name, page_prefix, index))

# 3. Seconde passe : Attribution du nom final avec l'index global
for temp_name, page_prefix, global_index in temp_files:
    # On reconstruit le nom : Préfixe de la page d'origine + Index global
    new_name = f"{page_prefix}line_{global_index:04d}.png"
    
    os.rename(os.path.join(path, temp_name), os.path.join(path, new_name))
    print(f"Traité : {new_name}")

print(f"\nTerminé ! {len(files)} fichiers réindexés avec succès.")