"""Line-to-text pipeline implementation."""

import logging
from typing import Dict, Any

from src.data_loaders.base import DataLoader
from src.models.base import OCRModel
from src.pipelines.base import Pipeline
from src.utils.metrics import calculate_cer, calculate_wer


logger = logging.getLogger(__name__)


class LineToTextPipeline(Pipeline):
    """Run OCR on line-level images and compute accuracy metrics."""

    def __init__(self, data_loader: DataLoader, model: OCRModel) -> None:
        self._data_loader = data_loader
        self._model = model

    def get_name(self) -> str:
        return "line_to_text"

    def run(self) -> Dict[str, Any]:
        logger.info("Starting line-to-text pipeline with model %s", self._model.get_name())

        try:
            samples, dataset_info = self._data_loader.load_data()
        except Exception as exc:
            logger.error("Failed to load data: %s", exc)
            raise

        num_samples = len(samples)
        if num_samples == 0:
            raise ValueError("No samples available for pipeline execution")

        total_cer = 0.0
        total_wer = 0.0

        for index, (image_path, ground_truth) in enumerate(samples, start=1):
            logger.info("Processing sample %s/%s: %s", index, num_samples, image_path)

            try:
                prediction = self._model.predict(image_path)

                cer = calculate_cer(ground_truth, prediction)
                wer = calculate_wer(ground_truth, prediction)

                total_cer += cer
                total_wer += wer

                logger.info("Sample %s - CER: %.4f, WER: %.4f", index, cer, wer)
                logger.info("Ground truth: '%s'", ground_truth)
                logger.info("Prediction: '%s'", prediction)

            except Exception as exc:
                logger.error("Failed to process sample %s: %s", index, exc)
                raise

        final_cer = total_cer / num_samples
        final_wer = total_wer / num_samples

        logger.info(
            "Pipeline completed - Final CER: %.4f, Final WER: %.4f",
            final_cer,
            final_wer,
        )

        return {
            "final_cer": final_cer,
            "final_wer": final_wer,
            "dataset_info": dataset_info,
        }
