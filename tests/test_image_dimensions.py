"""Test des dimensions d'image et visualisation de la normalisation CRNN."""

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.crnn_model import resizeNormalize, CRNN

print("="*70)
print("🧪 TEST DES DIMENSIONS D'IMAGE POUR LE MODÈLE CRNN")
print("="*70)

# Charger le modèle pour vérifier ses paramètres
print("\n1️⃣ Analyse des poids du modèle...")
weights = torch.load('models_local/crnn.pth', map_location='cpu', weights_only=False)

# Extraire les paramètres
nc = weights['cnn.conv0.weight'].shape[1]
nh = weights['rnn.0.rnn.weight_hh_l0'].shape[1]
nclass = weights['rnn.1.embedding.bias'].shape[0]

print(f"   ✅ n_channels: {nc} ({'grayscale' if nc == 1 else 'RGB'})")
print(f"   ✅ n_hidden: {nh}")
print(f"   ✅ n_class: {nclass} (alphabet: {nclass - 1} caractères)")

# Charger une image de test
test_image_path = "data_local/pero_dataset/dataset_test_1/lines_test_1_10first/line_1.jpg"
print(f"\n2️⃣ Chargement de l'image test: {test_image_path}")

if not os.path.exists(test_image_path):
    print(f"   ❌ Image non trouvée: {test_image_path}")
    print("   💡 Veuillez fournir le chemin d'une image de test")
    sys.exit(1)

img_original = Image.open(test_image_path)
print(f"   📐 Dimensions originales: {img_original.size} (width x height)")
print(f"   🎨 Mode: {img_original.mode}")

# Convertir en niveaux de gris si nécessaire
if img_original.mode != 'L':
    img_gray = img_original.convert('L')
    print(f"   🔄 Convertie en grayscale (mode L)")
else:
    img_gray = img_original

# Tester différentes dimensions
print(f"\n3️⃣ Test de différentes dimensions de redimensionnement:")
print("   " + "-"*60)

test_dimensions = [
    (100, 32),   # Standard CRNN
    (280, 32),   # Plus large
    (160, 32),   # Moyen
    (200, 64),   # Plus haute
]

for img_width, img_height in test_dimensions:
    try:
        # Vérifier que img_height est multiple de 16
        if img_height % 16 != 0:
            print(f"   ⚠️  ({img_width}, {img_height}): img_height doit être multiple de 16 - SKIP")
            continue
        
        transform = resizeNormalize((img_width, img_height))
        tensor = transform(img_gray)
        
        # Calculer la hauteur finale après CNN
        # Le CNN CRNN a 4 pooling layers qui réduisent la hauteur
        final_h = img_height // 16  # 4 pooling de 2x2 = 2^4 = 16
        
        print(f"   ✅ ({img_width:3d}, {img_height:2d}) → tensor: {tuple(tensor.shape)} → hauteur CNN finale: {final_h}")
        
        if final_h != 1:
            print(f"      ⚠️  ATTENTION: hauteur finale = {final_h}, devrait être 1!")
            
    except Exception as e:
        print(f"   ❌ ({img_width}, {img_height}): Erreur - {e}")

# Visualisation de la normalisation
print(f"\n4️⃣ Visualisation de l'image normalisée (32x100):")
transform_standard = resizeNormalize((100, 32))
tensor_standard = transform_standard(img_gray)

# Créer une figure avec 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Image originale
axes[0].imshow(img_original, cmap='gray' if img_original.mode == 'L' else None)
axes[0].set_title(f'Image Originale\n{img_original.size[0]}x{img_original.size[1]} px')
axes[0].axis('off')

# Image redimensionnée (avant normalisation)
img_resized = img_gray.resize((100, 32), Image.BILINEAR)
axes[1].imshow(img_resized, cmap='gray')
axes[1].set_title('Après Resize\n100x32 px')
axes[1].axis('off')

# Image normalisée (dénormalisée pour affichage)
# La normalisation fait: img.sub_(0.5).div_(0.5) => (img - 0.5) / 0.5
# Pour inverser: img_denorm = (img_norm * 0.5) + 0.5
img_normalized = tensor_standard.numpy().squeeze()
img_denormalized = (img_normalized * 0.5) + 0.5
img_denormalized = np.clip(img_denormalized, 0, 1)  # Clip pour [0, 1]

axes[2].imshow(img_denormalized, cmap='gray')
axes[2].set_title('Après Normalisation\n(dénormalisée pour affichage)')
axes[2].axis('off')

plt.tight_layout()
output_path = 'tests/image_normalization_test.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   💾 Image sauvegardée: {output_path}")

# Afficher les statistiques du tensor normalisé
print(f"\n5️⃣ Statistiques du tensor normalisé:")
print(f"   Shape: {tensor_standard.shape}")
print(f"   Min: {tensor_standard.min():.3f}")
print(f"   Max: {tensor_standard.max():.3f}")
print(f"   Mean: {tensor_standard.mean():.3f}")
print(f"   Std: {tensor_standard.std():.3f}")

# Vérifier la compatibilité avec le modèle
print(f"\n6️⃣ Test de compatibilité avec le modèle CRNN:")
try:
    # Créer le modèle
    model = CRNN(imgH=32, nc=1, nclass=37, nh=256)
    model.load_state_dict(weights)
    model.eval()
    
    # Ajouter la dimension batch
    input_tensor = tensor_standard.unsqueeze(0)
    print(f"   Input shape: {tuple(input_tensor.shape)} (batch, channels, height, width)")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"   Output shape: {tuple(output.shape)} (sequence_length, batch, n_class)")
    print(f"   ✅ Le modèle fonctionne correctement!")
    
    # Afficher quelques prédictions brutes
    probs = torch.nn.functional.softmax(output, dim=2)
    max_probs, max_indices = probs.max(2)
    
    print(f"\n   Premières prédictions brutes (top 10 timesteps):")
    for i in range(min(10, output.shape[0])):
        class_idx = max_indices[i, 0].item()
        prob = max_probs[i, 0].item()
        print(f"   Timestep {i:2d}: class {class_idx:2d} (prob: {prob:.3f})")
    
except Exception as e:
    print(f"   ❌ Erreur lors du test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ Tests terminés!")
print("="*70)

