#!/usr/bin/env python3
"""Crop binarized handwritten line images to remove white space."""

from PIL import Image
import numpy as np
from pathlib import Path


def crop_whitespace(image_path, output_path):
    """Crop image to remove white space on all sides."""
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Find pixels with ink (assuming white is 255)
    has_ink = img_array < 255
    
    if not np.any(has_ink):
        # Empty image, save as is
        img.save(output_path)
        return
    
    # Find columns with ink (horizontal)
    cols_with_ink = np.any(has_ink, axis=0)
    left = np.argmax(cols_with_ink)
    right = len(cols_with_ink) - np.argmax(cols_with_ink[::-1])
    
    # Find rows with ink (vertical)
    rows_with_ink = np.any(has_ink, axis=1)
    top = np.argmax(rows_with_ink)
    bottom = len(rows_with_ink) - np.argmax(rows_with_ink[::-1])
    
    # Crop and save
    cropped = img.crop((left, top, right, bottom))
    cropped.save(output_path)


def main():
    input_dir = Path("data_local/perso_dataset/hector_500_cropped/lines_out_sorted")
    output_dir = Path("data_local/perso_dataset/hector_500_cropped/lines_out_sorted_cropped")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all images
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    
    for img_path in image_files:
        output_path = output_dir / img_path.name
        crop_whitespace(img_path, output_path)
        print(f"Processed: {img_path.name}")
    
    print(f"\nDone! Processed {len(image_files)} images.")


if __name__ == "__main__":
    main()
