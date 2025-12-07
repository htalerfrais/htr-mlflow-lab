"""Inspect CRNN PyTorch model architecture."""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

model_path = "models_local/crnn.pth"

print(f"Loading model from: {model_path}")
weights = torch.load(model_path, map_location='cpu', weights_only=False)

print(f"\nType: {type(weights)}")

if isinstance(weights, dict):
    print(f"\nKeys ({len(weights.keys())} layers):")
    for key in weights.keys():
        print(f"  {key}: {weights[key].shape}")
else:
    print("\nC'est un modèle complet (pas un state_dict)")

