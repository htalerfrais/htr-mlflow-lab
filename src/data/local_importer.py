from __future__ import annotations

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
          images_dir: str = "data_local/pero_dataset/dataset_test_1/lines_test_1_10first",
          ground_truth_path: str = "data_local/pero_dataset/dataset_test_1/gt_test_1_10first.json",
          image_template: str = "line_{id}.jpg"
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
            
            image_name = self._image_template.format(id=identifier) #associate the right line image to the gt text
            image_path = os.path.join(self._images_dir, image_name) 

            if not os.path.exists(image_path):
                missing_images.append(image_path)
                continue

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
