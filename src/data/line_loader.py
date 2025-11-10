# src/data_loaders/line_loader.py
"""IAM dataset loader implementation."""

import logging
import os
import tempfile
from typing import List, Tuple

from datasets import load_dataset

from src.data.base import DataLoader, Sample, DatasetInfo


logger = logging.getLogger(__name__)


class IAMLineLoader(DataLoader):
    """Load IAM line-level samples from Hugging Face."""

    def __init__(self, dataset_name: str = "Teklia/IAM-line", config_name: str = "default") -> None:
        self._dataset_name = dataset_name
        self._config_name = config_name

    def load_data(self, split: str = "validation") -> Tuple[List[Sample], DatasetInfo]:
        try:
            dataset = load_dataset(self._dataset_name, self._config_name)

            temp_dir = tempfile.mkdtemp()

            samples: List[Sample] = []
            data_split = dataset[split]

            for i, sample in enumerate(data_split):
                image = sample["image"]
                text = sample["text"]

                image_path = os.path.join(temp_dir, f"sample_{i}.png")
                image.save(image_path)

                samples.append((image_path, text))

            logger.info("Loaded %s sample(s) from %s (%s split)", len(samples), self._dataset_name, split)

            dataset_info: DatasetInfo = {
                "dataset_name": self._dataset_name,
                "dataset_split": split,
                "num_samples": len(samples),
                "dataset_version": self._config_name,
            }

            return samples, dataset_info

        except Exception as exc:
            raise Exception(f"Failed to load IAM dataset: {exc}") from exc

    def get_name(self) -> str:
        return "IAM-line"
