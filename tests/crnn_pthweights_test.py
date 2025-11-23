import torch

# Charger le fichier
weights = torch.load('models_local/crnn.pth', map_location='cpu', weights_only=False)

# Voir la structure
print(type(weights))
print(weights.keys() if isinstance(weights, dict) else "C'est un modèle complet")

# Si c'est un dict, voir les layers
if isinstance(weights, dict):
    for key in weights.keys():
        print(f"{key}: {weights[key].shape}")