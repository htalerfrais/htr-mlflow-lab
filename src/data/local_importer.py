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
          images_dir: str = "data_local/perso_dataset/hector_pages_lines_2/lines_out_sorted",
          ground_truth_path: str = "data_local/perso_dataset/hector_pages_lines_2/gt_hector_pages_lines.json",
          image_template: str = "page_*_line_{id:04d}.jpg"
    ) -> None:
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
