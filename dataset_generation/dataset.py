import cv2
import numpy as np
import glob
import os
from PIL import Image


# ================= CONFIG =================
IMAGE_GLOB = "../data_local/perso_dataset/hector_pages_lines_2/raw_pages/*.jpeg"
OUT_DIR = "../data_local/perso_dataset/hector_pages_lines_2/lines_out"

CANVAS_H = 128
VERT_MARGIN = 110
HORZ_MARGIN = 110
MIN_LINE_HEIGHT = 15
MIN_BLACK_PIXELS = 50  # Nombre minimum de pixels noirs requis pour qu'une ligne soit valide
SIDE_CROP_RATIO = 0   # hard remove paper edges


os.makedirs(OUT_DIR, exist_ok=True)


# ================= HELPERS =================
def to_canvas(img, H=128):
    h, w = img.shape
    if h > H:
        scale = H / h
        img = cv2.resize(img, (int(w * scale), H), cv2.INTER_CUBIC)
        h = H
    canvas = np.full((H, img.shape[1]), 255, np.uint8)
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

    # -------- PAGE DETECTION --------
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    page_bw = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    cnts, _ = cv2.findContours(
        255 - page_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(cnts) == 0:
        print(f"⚠️  No contours found in {absolute_path}, skipping...")
        continue
    page = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(page)

    pad = 15
    gray = gray[y+pad:y+h-pad, x+pad:x+w-pad]

    # -------- HARD SIDE CROP --------
    H, W = gray.shape
    cut = int(SIDE_CROP_RATIO * W)
    gray = gray[:, cut:W-cut]

    # -------- SHADOW REMOVAL --------
    bg = cv2.GaussianBlur(gray, (61, 61), 0)
    norm = cv2.divide(gray, bg, scale=255)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)

    # -------- BINARIZE --------
    bw = cv2.threshold(
        norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # -------- LINE DETECTION --------
    ink = np.sum(bw == 0, axis=1)
    kernel = np.ones(31) / 31
    ink_smooth = np.convolve(ink, kernel, mode="same")

    th = 0.15 * np.max(ink_smooth)
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

        out_name = f"{name}_line_{global_line_id:04d}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), line)
        global_line_id += 1


print(f"✅ Saved {global_line_id} lines into {OUT_DIR}")
