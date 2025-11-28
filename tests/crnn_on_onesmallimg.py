"""Test minimaliste du modèle CRNN sur une seule image."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
import matplotlib.pyplot as plt
from src.models.crnn_model import CRNNModel, resizeNormalize

# ===== CONFIGURATION =====
IMAGE_PATH = "data_local\pero_dataset\dataset_sizes\Hector.jpg"
GROUND_TRUTH = "Hector"
MODEL_PATH = "models_local/crnn.pth" 

img_width = 100

# ===== CHARGEMENT =====
print("🔧 Chargement du modèle...")
model = CRNNModel(
    model_path=MODEL_PATH,
    img_height=32,
    img_width=img_width,  # Plus large pour une meilleure qualité
    n_hidden=256,
    device='cpu'
)

print(f"📷 Chargement de l'image: {IMAGE_PATH}")
img_original = Image.open(IMAGE_PATH).convert('L')

# ===== PRÉDICTION =====
print("🔮 Prédiction en cours...")
prediction = model.predict(img_original)

# ===== VISUALISATION =====
transform = resizeNormalize((img_width, 32))
tensor = transform(img_original)
img_resized = img_original.resize((img_width, 32), Image.BILINEAR)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].imshow(img_original, cmap='gray')
axes[0].set_title(f'Original ({img_original.size[0]}x{img_original.size[1]})')
axes[0].axis('off')

axes[1].imshow(img_resized, cmap='gray')
axes[1].set_title(f'Après resize ({img_width}x32)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('tests/crnn_test_result.png', dpi=100)
print("💾 Visualisation sauvegardée: tests/crnn_test_result.png")

# ===== RÉSULTATS =====
print("\n" + "="*60)
print("📊 RÉSULTATS")
print("="*60)
print(f"Ground truth : {GROUND_TRUTH}")
print(f"Prédiction   : {prediction}")
print(f"Match        : {'✅' if prediction.lower() == GROUND_TRUTH.lower() else '❌'}")
print("="*60)

