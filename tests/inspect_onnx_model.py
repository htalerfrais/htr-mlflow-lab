"""Inspect ONNX model architecture."""

import onnx
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

onnx_path = "models_local/model-crnn1.onnx"

print(f"Loading ONNX model from: {onnx_path}")
model = onnx.load(onnx_path)

print(f"\nModel inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[dim.dim_value for dim in inp.type.tensor_type.shape.dim]}")

print(f"\nModel outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[dim.dim_value for dim in out.type.tensor_type.shape.dim]}")

print(f"\nNodes ({len(model.graph.node)}):")
for i, node in enumerate(model.graph.node[:10]):  # Show first 10 nodes
    print(f"  {i+1}. {node.name}: {node.op_type}")
if len(model.graph.node) > 10:
    print(f"  ... and {len(model.graph.node) - 10} more nodes")

