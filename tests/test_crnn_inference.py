"""Minimal test script for CRNN inference."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.onnx_model import ONNXModel

# Initialize model
model = ONNXModel()

# Run inference
image_path = "data_local/pero_dataset/dataset_sizes/Hector.jpg"
prediction = model.predict(image_path)

print(prediction)

