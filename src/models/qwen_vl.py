from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor

from src.models.base import OCRModel, ImageInput


class QwenVLModel(OCRModel):
    """
    Minimal wrapper around Qwen/Qwen2.5-VL-* vision-language models for OCR.

    This wrapper formats a simple multimodal chat prompt (image + instruction)
    and returns the generated text.
    """

    def __init__(
        self,
        pretrained_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str | None = None,
        device_map: str | dict | None = "auto",
        torch_dtype: str | None = "auto",
        max_new_tokens: int = 256,
        prompt: str = "Read and transcribe all text in the image exactly. Output only the transcription.",
    ) -> None:
        self._pretrained_model_name = pretrained_model_name
        self._max_new_tokens = max_new_tokens
        self._prompt = prompt

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._device_map = device_map
        self._torch_dtype = torch_dtype

        self._processor = AutoProcessor.from_pretrained(pretrained_model_name)

        # Prefer the model class when available (transformers versions vary).
        model = None
        load_kwargs: dict = {}

        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype if torch_dtype == "auto" else getattr(torch, torch_dtype)

        if device_map is not None:
            load_kwargs["device_map"] = device_map

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained_model_name,
                **load_kwargs,
            )
            print("Qwen2_5_VLForConditionalGeneration loaded")
        except Exception:
            print("Qwen2_5_VLForConditionalGeneration not found")
            # Fallback for older/newer transformers where the specific class name differs.
            try:
                from transformers import AutoModelForVision2Seq

                model = AutoModelForVision2Seq.from_pretrained(
                    pretrained_model_name,
                    **load_kwargs,
                )
                print("AutoModelForVision2Seq loaded")
            except Exception:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name,
                    **load_kwargs,
                )
                print("AutoModelForCausalLM loaded")
        self._model = model

        # If we didn't use device_map, place the whole model on the selected device.
        if self._device_map is None:
            self._model.to(self._device)

        self._model.eval()

    def predict(self, image: ImageInput) -> str:
        pil_image = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]

        # Qwen-VL processors typically implement apply_chat_template().
        if hasattr(self._processor, "apply_chat_template"):
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
            )
        else:
            # Fallback: just pass plain text + image.
            inputs = self._processor(
                text=[self._prompt],
                images=[pil_image],
                return_tensors="pt",
            )

        # Move tensors to the right device (no-op for device_map="auto" models if already aligned).
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        # If we provided input_ids, strip the prompt tokens to keep only newly generated text.
        if "input_ids" in inputs:
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = generated_ids[:, prompt_len:]

        out = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return out.strip()

    def get_name(self) -> str:
        return f"QwenVL(pretrained={self._pretrained_model_name})"
