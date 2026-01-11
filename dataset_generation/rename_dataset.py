import json
import os
import re
import shutil
from pathlib import Path

# ================= CONFIG =================
DATASET_FOLDER = "hector_pages_lines_3"

JSON_PATH = f"../data_local/perso_dataset/{DATASET_FOLDER}/gt_hector_pages_lines.json"
IMAGES_DIR = f"../data_local/perso_dataset/{DATASET_FOLDER}/lines_out"
OUTPUT_DIR = f"../data_local/perso_dataset/{DATASET_FOLDER}/lines_out_sorted"
OUTPUT_LINE_NUMBER_FORMAT = "04d"  # Format du numéro de ligne (04d = 4 chiffres avec zéros)
OUTPUT_FILE_EXTENSION = ".jpg"


def extract_line_id(filename):
    """Retourne l'id numérique extrait de 'line_XXXX'."""
    match = re.search(r"line_(\d+)", filename)
    return int(match.group(1)) if match else None


def process_images():
    """Réindexe les fichiers images en ignorant la partie 'page' dans leur nom."""
    folder = Path(IMAGES_DIR)
    target = Path(OUTPUT_DIR)
    target.mkdir(parents=True, exist_ok=True)

    files = sorted(
        (f for f in folder.glob(f"*{OUTPUT_FILE_EXTENSION}") if extract_line_id(f.name) is not None),
        key=lambda f: extract_line_id(f.name),
    )

    print(f"Reordering {len(files)} images...")
    for new_id, img in enumerate(files, start=1):
        page_match = re.search(r"page_(\d+)_", img.name)
        page = page_match.group(1) if page_match else "0"
        new_name = f"page_{page}_line_{new_id:{OUTPUT_LINE_NUMBER_FORMAT}}{OUTPUT_FILE_EXTENSION}"
        destination = target / new_name
        if destination.exists():
            destination.unlink()
        shutil.copy2(img, destination)

    print(f"Images rewritten in {OUTPUT_DIR}")


def process_json():
    """Réindexe les IDs du JSON pour qu'ils soient consécutifs."""
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    samples.sort(key=lambda s: s.get("id", 0))

    for new_id, sample in enumerate(samples, start=1):
        sample["id"] = new_id

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"JSON ids rewritten, total {len(samples)} entries")


def main():
    process_images()
    process_json()


if __name__ == "__main__":
    main()
