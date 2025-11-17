from __future__ import annotations

from itertools import groupby

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.models.base import OCRModel, ImageInput


DEFAULT_CHARSET = list('\'3.FR20JWIe8CyBowxTV5rgOYQ,ipPcqDGnMAK(Eb6)fH:"9LlUt;jsz m4&1#kZ-adNhvu7!S?')


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
    
    def _resize_keep_aspect_ratio(self, image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        target_h, target_w = target_size
        img_w, img_h = image.size
        
        scale = min(target_w / img_w, target_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_image = Image.new('RGB', (target_w, target_h), (0, 0, 0))
        new_image.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        
        return new_image

    def predict(self, image: ImageInput) -> str:
        pil_image = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        
        if self._input_size is not None:
            pil_image = self._resize_keep_aspect_ratio(pil_image, self._input_size)
        
        img_array = np.array(pil_image).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        outputs = self._session.run([self._output_name], {self._input_name: img_array})
        predictions = outputs[0]
        text = self._greedy_decode_ctc(predictions)
        
        return text.strip()
    
    def _greedy_decode_ctc(self, predictions: np.ndarray) -> str:
        argmax_preds = np.argmax(predictions, axis=-1)
        grouped_preds = [[k for k, _ in groupby(preds)] for preds in argmax_preds]
        texts = ["".join([self._charset[k] for k in group if k < len(self._charset)]) for group in grouped_preds]
        return texts[0]

    def get_name(self) -> str:
        return f"ONNXModel(model={self._onnx_path})"

