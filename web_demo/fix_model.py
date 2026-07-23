#!/usr/bin/env python3
from pathlib import Path

# Read both files
model_path = Path("models/model.onnx")
data_path = Path("models/model.onnx.data")

# Just copy data into a temp file, then rename
# (This is a hack - real solution needs onnx module)
with open(model_path, "rb") as f:
    model_content = f.read()
with open(data_path, "rb") as f:
    data_content = f.read()

# Write as single combined file (not proper ONNX, but ort can sometimes read it)
# Actually, let's just keep them separate and ensure server serves them correctly
print(f"Model: {len(model_content) / 1024:.1f} KB")
print(f"Data: {len(data_content) / 1024:.1f} KB")
print("Keeping as separate files - ONNX Runtime Web handles this")
