from __future__ import annotations

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.models.base import OCRModel, ImageInput


DEFAULT_CHARSET = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'-")


class ONNXModel(OCRModel):
    """Wrapper around ONNX Runtime for OCR inference."""

    def __init__(
        self,
        onnx_path: str = "models_local/model-crnn1.onnx",
        charset: list[str] | None = None,
        device: str | None = None,
        input_size: tuple[int, int] | None = (96, 1408),
    ) -> None:
        self._onnx_path = onnx_path
        self._charset = charset if charset is not None else DEFAULT_CHARSET
        self._device = device or "cpu"
        self._input_size = input_size
        
        if self._device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def predict(self, image: ImageInput) -> str:
        pil_image = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        
        if self._input_size is not None:
            pil_image = pil_image.resize((self._input_size[1], self._input_size[0]))
        
        # Convert to numpy array (C, H, W) format with batch dimension
        img_array = np.array(pil_image).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        outputs = self._session.run([self._output_name], {self._input_name: img_array})
        predictions = outputs[0]
        text = self._greedy_decode_ctc(predictions)
        
        return text.strip()
    
    def _greedy_decode_ctc(self, predictions: np.ndarray) -> str:
        """Greedy CTC decoding: collapse repetitions and remove blank token."""
        # Get best prediction for each timestep
        pred_indices = np.argmax(predictions[0], axis=-1)
        
        decoded_indices = []
        previous = -1
        for idx in pred_indices:
            if idx != 0 and idx != previous:
                decoded_indices.append(idx)
            previous = idx
        
        # Map indices to characters using charset
        text = ''.join(self._charset[idx - 1] for idx in decoded_indices if 1 <= idx <= len(self._charset))
        return text

    def get_name(self) -> str:
        return f"ONNXModel(model={self._onnx_path})"

