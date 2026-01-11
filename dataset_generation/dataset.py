import cv2
import numpy as np
import glob
import os
from PIL import Image


# ================= CONFIG =================
IMAGE_GLOB = "../data_local/perso_dataset/hector_pages_lines_2/raw_pages/*.jpeg"
OUT_DIR = "../data_local/perso_dataset/hector_pages_lines_2/lines_out"

# Canvas settings
CANVAS_H = 128
CANVAS_BACKGROUND = 255  # Valeur du blanc dans le canvas

# Margins
VERT_MARGIN = 150
HORZ_MARGIN = 110

# Line detection thresholds
MIN_LINE_HEIGHT = 15
MIN_BLACK_PIXELS = 50  # Nombre minimum de pixels noirs requis pour qu'une ligne soit valide
LINE_DETECTION_THRESHOLD = 0.12  # Seuil de détection de ligne (ratio du maximum)
SMOOTH_KERNEL_SIZE = 21  # Taille du kernel de lissage pour la détection de lignes

# Page detection
PAGE_DETECTION_BLUR_SIZE = (7, 7)  # Taille du kernel de blur pour la détection de page
PAGE_PADDING = 15  # Padding autour de la page détectée

# Shadow removal
SHADOW_REMOVAL_BLUR_SIZE = (61, 61)  # Taille du kernel de blur pour l'élimination des ombres
SHADOW_REMOVAL_SCALE = 255  # Scale pour la division lors de l'élimination des ombres
NORMALIZE_MIN = 0  # Valeur minimale pour la normalisation
NORMALIZE_MAX = 255  # Valeur maximale pour la normalisation

# Binarization
BINARIZATION_THRESHOLD_MIN = 0  # Valeur minimale pour le threshold (OTSU utilise 0)
BINARIZATION_THRESHOLD_MAX = 255  # Valeur maximale pour le threshold

# Side crop
SIDE_CROP_RATIO = 0   # hard remove paper edges

# Output format
OUTPUT_FILE_EXTENSION = ".jpg"  # Extension des fichiers de sortie
OUTPUT_LINE_NUMBER_FORMAT = "04d"  # Format du numéro de ligne (04d = 4 chiffres avec zéros)


os.makedirs(OUT_DIR, exist_ok=True)


# ================= HELPERS =================
def to_canvas(img, H=None):
    if H is None:
        H = CANVAS_H
    h, w = img.shape
    if h > H:
        scale = H / h
        img = cv2.resize(img, (int(w * scale), H), cv2.INTER_CUBIC)
        h = H
    canvas = np.full((H, img.shape[1]), CANVAS_BACKGROUND, np.uint8)
    y0 = (H - h) // 2
    canvas[y0:y0+h, :] = img
    return canvas


# ================= MAIN LOOP =================
counter = 0
global_line_id = 0

images = list(glob.glob(IMAGE_GLOB))
print(f"Found {len(images)} images matching '{IMAGE_GLOB}'")

for img_path in images:
    name = os.path.splitext(os.path.basename(img_path))[0]
    absolute_path = os.path.abspath(img_path)

    absolute_path = absolute_path.replace("\\", "/")
    print(f"Processing {absolute_path}")
    if not os.path.exists(absolute_path):
        print(f"⚠️  File does not exist: {absolute_path}, skipping...")
        continue
    print(f"Image exists: {absolute_path}")

    try:
        gray = np.array(Image.open(absolute_path).convert('L'))
    except Exception as e:
        print(f"⚠️  Failed to read {absolute_path}: {e}")
        continue

    # -------- SHADOW REMOVAL --------
    bg = cv2.GaussianBlur(gray, SHADOW_REMOVAL_BLUR_SIZE, 0)
    norm = cv2.divide(gray, bg, scale=SHADOW_REMOVAL_SCALE)
    norm = cv2.normalize(norm, None, NORMALIZE_MIN, NORMALIZE_MAX, cv2.NORM_MINMAX)

    # -------- BINARIZE --------
    bw = cv2.threshold(
        norm, BINARIZATION_THRESHOLD_MIN, BINARIZATION_THRESHOLD_MAX, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # -------- LINE DETECTION --------
    ink = np.sum(bw == 0, axis=1)
    kernel = np.ones(SMOOTH_KERNEL_SIZE) / SMOOTH_KERNEL_SIZE
    ink_smooth = np.convolve(ink, kernel, mode="same")

    th = LINE_DETECTION_THRESHOLD * np.max(ink_smooth)
    mask = ink_smooth > th

    lines = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= MIN_LINE_HEIGHT:
                lines.append((start, i))
            start = None
    if start is not None:
        lines.append((start, len(mask)))

    # -------- EXTRACT LINES --------
    if len(lines) > 0:
        print(f"  {name}: detected {len(lines)} lines")
    
    for li, (y1, y2) in enumerate(lines):
        y1 = max(0, y1 - VERT_MARGIN)
        y2 = min(bw.shape[0], y2 + VERT_MARGIN)

        line = bw[y1:y2, :]

        ys, xs = np.where(line == 0)
        if len(xs) < MIN_BLACK_PIXELS:
            continue

        x1 = max(0, xs.min() - HORZ_MARGIN)
        x2 = min(line.shape[1], xs.max() + HORZ_MARGIN)
        line = line[:, x1:x2]

        line = to_canvas(line, CANVAS_H)

        out_name = f"{name}_line_{global_line_id:{OUTPUT_LINE_NUMBER_FORMAT}}{OUTPUT_FILE_EXTENSION}"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), line)
        global_line_id += 1


print(f"✅ Saved {global_line_id} lines into {OUT_DIR}")
