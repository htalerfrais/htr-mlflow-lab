# TrOCR Fine-Tuning Guide

## 🎯 Quick Start

**Lancer le fine-tuning:**
```powershell
# Activate venv
& .venv\Scripts\Activate.ps1

# Run fine-tuning
python src/fine_tuning/run_fine_tuning.py configs/fine_tuning/trocr_iam_finetune.yaml
```

**Ou sur WSL:**
```bash
conda activate htr-mlflow
python src/fine_tuning/run_fine_tuning.py configs/fine_tuning/trocr_iam_finetune.yaml
```

---

## 📁 Structure Créée

```
src/fine_tuning/
├── base_trainer.py       # Classe abstraite pour fine-tuning
├── trocr_trainer.py      # Fine-tuning TrOCR
├── utils.py              # Data collator, metrics (CER/WER)
└── run_fine_tuning.py    # Script principal

configs/fine_tuning/
└── trocr_iam_finetune.yaml  # Config par défaut
```

---

## ⚙️ Configuration

**Éditer `configs/fine_tuning/trocr_iam_finetune.yaml`:**

### Modèle
```yaml
model:
  pretrained_model_name: "microsoft/trocr-base-handwritten"
  # Ou: "microsoft/trocr-small-handwritten"
  # Ou: "microsoft/trocr-large-handwritten"
```

### Dataset
```yaml
dataset:
  name: "Teklia/IAM-line"
  max_samples: 100  # Pour test rapide, commenter pour full dataset
```

### Training
```yaml
training:
  num_train_epochs: 5        # Nombre d'epochs
  per_device_train_batch_size: 8  # Batch size (réduire si OOM)
  learning_rate: 5e-5        # Learning rate
  fp16: true                 # Mixed precision (GPU only)
```

### Output
```yaml
output:
  dir: "./models/fine_tuned_trocr_iam"  # Où sauver le modèle
```

---

## 🚀 Utilisation

### Test Rapide (100 samples)
```yaml
# Dans le config
dataset:
  max_samples: 100
```
```powershell
python src/fine_tuning/run_fine_tuning.py configs/fine_tuning/trocr_iam_finetune.yaml
```

### Full Training (tout le dataset)
```yaml
# Commenter max_samples dans le config
dataset:
  # max_samples: 100
```

### Avec GPU
```yaml
training:
  fp16: true
  per_device_train_batch_size: 16  # Plus grand batch si assez de VRAM
```

### Sans GPU (CPU)
```yaml
training:
  fp16: false
  per_device_train_batch_size: 4  # Plus petit batch
  num_train_epochs: 2  # Moins d'epochs (plus lent sur CPU)
```

---

## 📊 Résultats

**Pendant le training:**
- Logs dans console (loss, CER, WER par epoch)
- Checkpoints sauvés dans `output.dir`
- Métriques loggées dans MLflow

**Après le training:**
- Modèle fine-tuné dans `./models/fine_tuned_trocr_iam/`
- Best model chargé automatiquement
- Métriques finales affichées (CER, WER)

**MLflow:**
```
http://13.60.230.97:5000
Experiment: TrOCR-FineTuning
```

---

## 🔧 Utiliser le Modèle Fine-Tuné

**Créer une config pour inférence:**
```yaml
# configs/trocr_finetuned_iam.yaml
model: "trocr"
model_params:
  pretrained_model_name: "./models/fine_tuned_trocr_iam"  # Chemin local
  device: null  # Auto-detect GPU
  max_new_tokens: 256

dataset: "teklia/iam-line"
params:
  split: "test"
```

**Run inference:**
```powershell
python src/experiments/run_experiment.py configs/trocr_finetuned_iam.yaml
```

---

## 🎛️ Hyperparamètres Importants

### Learning Rate
- **5e-5** (défaut): Bon point de départ
- **1e-4**: Plus agressif, convergence rapide
- **1e-5**: Plus prudent, meilleure généralisation

### Batch Size
- **8** (défaut): Bon compromis
- **16+**: Si VRAM > 12GB
- **4**: Si OOM errors

### Epochs
- **5** (défaut): Standard pour IAM
- **10**: Pour meilleure convergence
- **3**: Test rapide

### FP16 (Mixed Precision)
- **true**: GPU avec Tensor Cores (RTX, A100, etc.)
- **false**: CPU ou GPU sans Tensor Cores

---

## 📈 Monitoring

**Pendant le training:**
```
Epoch 1/5: 100%|████████| 1000/1000 [10:30<00:00,  1.58it/s]
Eval CER: 0.0523, WER: 0.1234
Saving checkpoint to ./models/fine_tuned_trocr_iam/checkpoint-1000
```

**Métriques importantes:**
- **CER** (Character Error Rate): Plus bas = mieux
- **WER** (Word Error Rate): Plus bas = mieux
- **Loss**: Doit descendre progressivement

---

## 🐛 Troubleshooting

### OOM (Out of Memory)
```yaml
training:
  per_device_train_batch_size: 4  # Réduire
  gradient_accumulation_steps: 2  # Accumuler gradients
```

### Convergence lente
```yaml
training:
  learning_rate: 1e-4  # Augmenter
  warmup_steps: 1000   # Plus de warmup
```

### Overfitting
```yaml
training:
  weight_decay: 0.1    # Plus de régularisation
  num_train_epochs: 3  # Moins d'epochs
```

### GPU pas détecté
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```
Si False:
- Vérifier drivers NVIDIA
- Réinstaller PyTorch avec CUDA
- Utiliser WSL si Windows

---

## 📚 Exemples Avancés

### Early Stopping
```yaml
training:
  early_stopping_patience: 3
  early_stopping_threshold: 0.001
```

### Custom Dataset (tes propres images)
```python
# Dans trocr_trainer.py, adapter prepare_dataset()
# pour charger tes données locales
```

### Fine-tune sur subset spécifique
```yaml
dataset:
  train_split: "train[:1000]"  # Premiers 1000 samples
  eval_split: "validation[:100]"
```

---

## ✅ Checklist

Avant de lancer:
- [ ] GPU disponible (ou fp16: false pour CPU)
- [ ] Assez de VRAM (8GB+ recommandé)
- [ ] MLflow server accessible
- [ ] Dataset téléchargé (auto si Teklia/IAM-line)
- [ ] Config ajustée selon tes besoins

Après le training:
- [ ] CER/WER améliorés vs modèle de base?
- [ ] Modèle sauvé dans output.dir
- [ ] Métriques loggées dans MLflow
- [ ] Tester sur test set

---

**Temps estimés:**
- Test (100 samples, 5 epochs): ~5-10 min (GPU) / ~30-60 min (CPU)
- Full IAM (~10k samples, 5 epochs): ~2-3h (GPU) / ~10-15h (CPU)

**Bon fine-tuning! 🚀**
