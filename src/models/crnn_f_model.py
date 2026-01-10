# CRNN model trained on RIMES French dataset by Fatima
# Based on CRNNv2 architecture from fatima_model.ipynb

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from .base import OCRModel, ImageInput


# ===== French vocabulary (92 characters + CTC blank) =====
VOCAB = list(
    "abcdefghijklmnopqrstuvwxyzàâéèêëîôûùüç"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÀÂÉÈÊËÎÔÛÙÜÇ"
    "0123456789.,!? '"
)
IDX2CHAR = {i + 1: ch for i, ch in enumerate(VOCAB)}
N_CLASSES = len(VOCAB) + 1  # +1 for CTC blank (index 0)


# ===== CRNNv2 Architecture =====
class CRNNv2(nn.Module):
    """
    Slightly deeper & faster CRNN:
    - more conv blocks (better features)
    - more downsampling in height (smaller LSTM input)
    - same time axis reduction (W // 4) so training loop stays the same
    """
    def __init__(self, img_height: int, n_channels: int, n_classes: int):
        super(CRNNv2, self).__init__()

        self.cnn = nn.Sequential(
            # Block 1: 128xW -> 64xW/2
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64xW/2 -> 32xW/4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32xW/4 -> 16xW/4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),

            # Block 4: 16xW/4 -> 8xW/4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),
        )

        # after these pools, height = img_height / 16  (128 -> 8)
        conv_height = img_height // 16
        self.conv_height = conv_height
        self.rnn_input_size = 256 * conv_height

        self.dropout = nn.Dropout(0.3)

        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,
        )

        self.fc = nn.Linear(256 * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W]
        conv = self.cnn(x)           # [B,C,H',W'] with H'=8, W'=W//4
        conv = self.dropout(conv)
        b, c, h, w = conv.size()

        conv = conv.permute(0, 3, 1, 2)   # [B,W',C,H']
        conv = conv.reshape(b, w, c * h)  # [B,T, C*H] = [B,T, rnn_input_size]

        rnn_out, _ = self.rnn(conv)      # [B,T,512]
        out = self.fc(rnn_out)           # [B,T,n_classes]
        out = out.log_softmax(2)         # CTC expects log-probs

        return out


# ===== Preprocessing for variable width images =====
class ResizeNormalizeVariable:
    """Resize to fixed height with variable width, then normalize for grayscale."""

    def __init__(self, img_height: int = 128):
        self.img_height = img_height
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Resize keeping aspect ratio
        w, h = img.size
        new_h = self.img_height
        new_w = int(w * (new_h / h))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # To tensor and normalize with (0.5,), (0.5,)
        img_tensor = self.to_tensor(img)
        img_tensor = (img_tensor - 0.5) / 0.5
        return img_tensor


# ===== OCRModel wrapper =====
class CRNNFModel(OCRModel):
    """CRNN model trained on RIMES French dataset by Fatima."""

    def __init__(
        self,
        model_path: str = "models_local/best_model_metrics.pt",
        img_height: int = 128,
        n_channels: int = 1,
        device: str = 'cpu'
    ):
        """
        Initialize the CRNN-F model for French OCR.

        Args:
            model_path: Path to the checkpoint file (.pt)
            img_height: Height of input images (must be multiple of 16)
            n_channels: Number of input channels (1 for grayscale)
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.img_height = img_height
        self.idx2char = IDX2CHAR

        # Build model
        self.model = CRNNv2(img_height, n_channels, N_CLASSES)

        # Load checkpoint (structure: {"model_state_dict": ..., "epoch": ..., ...})
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Internal transform (grayscale, resize, normalize)
        self.transform = ResizeNormalizeVariable(img_height)

    def predict(self, image: ImageInput) -> str:
        """
        Run OCR inference on an image.

        Args:
            image: Either a path to an image file or a PIL Image object

        Returns:
            str: The recognized text from the image
        """
        # Convert to grayscale
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        elif image.mode != 'L':
            image = image.convert('L')

        # Apply transform and add batch dimension
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)  # (B, T, C) - batch_first format

        # Decode (using Fatima's decode_prediction logic)
        return self._decode_prediction(output)[0]

    def _decode_prediction(self, output: torch.Tensor) -> list:
        """
        CTC greedy decoding for (B, T, C) format (format of the output tensor of the model)

        Args:
            output: (batch, T, n_classes) log-softmax scores

        Returns:
            list: Decoded strings for each sample in batch
        """
        _, pred_idxs = torch.max(output, dim=2)  # [B,T]

        results = []
        for b in range(pred_idxs.size(0)):
            preds = pred_idxs[b].cpu().tolist()
            prev = None
            chars = []
            for idx in preds:
                if idx != prev and idx != 0:  # 0 is CTC blank
                    chars.append(self.idx2char.get(idx, ""))
                prev = idx
            results.append("".join(chars))
        return results

    def get_name(self) -> str:
        """Return a human-readable model name."""
        return "CRNN-F (RIMES French)"
