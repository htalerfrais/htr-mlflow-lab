"""Minimalist script to inspect TrOCR model modules for LoRA fine-tuning."""

from transformers import VisionEncoderDecoderModel

# Load the same model as in fine-tuning script
model = VisionEncoderDecoderModel.from_pretrained("agomberto/trocr-large-handwritten-fr")

print("=" * 80)
print("TrOCR Model Modules Inspection")
print("=" * 80)
print()

# Get all module names
all_modules = [name for name, _ in model.named_modules()]

print(f"Total number of modules: {len(all_modules)}")
print()

# Filter for linear layers (potential LoRA targets)
linear_modules = []
for name, module in model.named_modules():
    if hasattr(module, 'weight') and len(module.weight.shape) >= 2:
        # Check if it's a linear layer (has weight matrix)
        linear_modules.append(name)

print("=" * 80)
print("LINEAR LAYERS (potential LoRA targets):")
print("=" * 80)
for module_name in sorted(linear_modules):
    print(f"  {module_name}")

print()
print("=" * 80)
print("ENCODER LINEAR LAYERS (Vision):")
print("=" * 80)
encoder_modules = [m for m in linear_modules if "encoder" in m.lower() and "decoder" not in m.lower()]
for module_name in sorted(encoder_modules):
    print(f"  {module_name}")

print()
print("=" * 80)
print("DECODER LINEAR LAYERS (Text):")
print("=" * 80)
decoder_modules = [m for m in linear_modules if "decoder" in m.lower()]
for module_name in sorted(decoder_modules):
    print(f"  {module_name}")

print()
print("=" * 80)
print("RECOMMENDED TARGET MODULES:")
print("=" * 80)
print("Based on the inspection above, look for:")
print("  - Encoder: query, key, value, output.dense, intermediate.dense")
print("  - Decoder: q_proj, k_proj, v_proj, out_proj, fc1, fc2")
print("=" * 80)
