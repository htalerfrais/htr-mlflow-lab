from __future__ import annotations

import glob
import json
import logging
import os
from typing import List, Tuple

from src.data.base import DataImporter, Sample, DatasetInfo


logger = logging.getLogger(__name__)


class LocalLineImporter(DataImporter):
    """Import line images and ground truth stored locally on disk."""

    def __init__(
        self,
          images_dir: str | None = None,
          ground_truth_path: str | None = None,
          image_template: str = "*line_{id:04d}.jpg"
    ) -> None:
        if images_dir is None:
            raise ValueError("'images_dir' must be provided in the configuration")
        if ground_truth_path is None:
            raise ValueError("'ground_truth_path' must be provided in the configuration")
        self._images_dir = images_dir
        self._ground_truth_path = ground_truth_path
        self._image_template = image_template

    def import_data(self, split: str = "validation") -> Tuple[List[Sample], DatasetInfo]:  # noqa: ARG002
        try:
            with open(self._ground_truth_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception as exc:  # pragma: no cover - config errors
            raise RuntimeError(f"Failed to load ground truth file: {self._ground_truth_path}") from exc

        dataset_name = payload.get("dataset_name", "local_dataset")
        samples_payload = payload.get("samples", []) # get sub dict of key "samples" in dataset 

        samples: List[Sample] = []
        missing_images: List[str] = []

        for entry in samples_payload:
            identifier = entry.get("id")
            ground_truth = entry.get("ground_truth")

            if identifier is None or ground_truth is None:
                logger.warning("Skipping malformed entry without 'id' or 'ground_truth': %s", entry)
                continue
            
            # Format the pattern to search for images (e.g., "page_*_line_0002.png")
            image_pattern = self._image_template.format(id=identifier)
            image_path_pattern = os.path.join(self._images_dir, image_pattern)
            
            # Use glob to find matching image files (handles page_*_line_XXXX.png pattern)
            matching_images = glob.glob(image_path_pattern)
            
            if not matching_images:
                missing_images.append(image_path_pattern)
                continue
            
            if len(matching_images) > 1:
                logger.warning(
                    "Multiple images found for id %s: %s. Using the first one: %s",
                    identifier,
                    matching_images,
                    matching_images[0]
                )
            
            # Use the first matching image
            image_path = matching_images[0]
            samples.append((image_path, ground_truth))

        if missing_images:
            raise FileNotFoundError(
                "Missing image files for the following entries: " + ", ".join(missing_images)
            )

        dataset_info: DatasetInfo = {
            "dataset_name": dataset_name,
            "dataset_root": self._images_dir,
            "ground_truth_path": self._ground_truth_path,
            "num_samples": len(samples),
        }

        logger.info(
            "Imported %s sample(s) from local dataset '%s' located at %s",
            len(samples),
            dataset_name,
            self._images_dir,
        )

        return samples, dataset_info


class LocalLineTextImporter(DataImporter):
    """Import line images and ground truth from a simple text file format.
    
    This importer handles datasets where:
    - Images are stored in a 'lines' directory with various extensions (jpg, png, etc.)
    - Ground truth is stored in a text file with format: 'image_name.ext : ground_truth_text'
    """

    def __init__(
        self,
        lines_dir: str | None = None,
        ground_truth_txt_path: str | None = None,
        display_name: str = "local_text_dataset"
    ) -> None:
        if lines_dir is None:
            raise ValueError("'lines_dir' must be provided in the configuration")
        if ground_truth_txt_path is None:
            raise ValueError("'ground_truth_txt_path' must be provided in the configuration")
        self._lines_dir = lines_dir
        self._ground_truth_txt_path = ground_truth_txt_path
        self._dataset_name = display_name

    def import_data(self, split: str = "validation") -> Tuple[List[Sample], DatasetInfo]:  # noqa: ARG002
        try:
            with open(self._ground_truth_txt_path, "r", encoding="utf-8") as fp:
                lines = fp.readlines()
        except Exception as exc:  # pragma: no cover - config errors
            raise RuntimeError(f"Failed to load ground truth file: {self._ground_truth_txt_path}") from exc

        samples: List[Sample] = []
        missing_images: List[str] = []
        skipped_lines: List[str] = []

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse the format: "image_name.ext : ground_truth_text"
            if " : " not in line:
                skipped_lines.append(f"Line {line_num}: missing separator ' : ' - {line[:50]}")
                continue
            
            parts = line.split(" : ", 1)  # Split only on the first occurrence
            if len(parts) != 2:
                skipped_lines.append(f"Line {line_num}: invalid format - {line[:50]}")
                continue
            
            image_name = parts[0].strip()
            ground_truth = parts[1].strip()
            
            if not image_name or not ground_truth:
                skipped_lines.append(f"Line {line_num}: empty image name or ground truth - {line[:50]}")
                continue
            
            # Build the full image path
            image_path = os.path.join(self._lines_dir, image_name)
            
            # Check if the image exists
            if not os.path.isfile(image_path):
                missing_images.append(image_path)
                continue
            
            samples.append((image_path, ground_truth))

        # Log warnings for skipped lines
        if skipped_lines:
            logger.warning(
                "Skipped %d malformed line(s) in ground truth file:\n%s",
                len(skipped_lines),
                "\n".join(skipped_lines[:10])  # Show first 10 only
            )

        # Raise error if images are missing
        if missing_images:
            raise FileNotFoundError(
                f"Missing {len(missing_images)} image file(s). First 10 missing:\n" 
                + "\n".join(missing_images[:10])
            )

        dataset_info: DatasetInfo = {
            "dataset_name": self._dataset_name,
            "dataset_root": self._lines_dir,
            "ground_truth_path": self._ground_truth_txt_path,
            "num_samples": len(samples),
        }

        logger.info(
            "Imported %s sample(s) from local text dataset '%s' located at %s",
            len(samples),
            self._dataset_name,
            self._lines_dir,
        )

        return samples, dataset_info


