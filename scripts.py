import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os

# --------------------------------------------------
# Project-relative paths
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

RAW_MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "raw",
    "whisper_tiny"
)

MODEL_ID = "openai/whisper-tiny"

print(f"Downloading {MODEL_ID}...")
print(f"Saving raw artifacts to: {RAW_MODEL_DIR}")

# --------------------------------------------------
# Load model (FP32 reference weights)
# --------------------------------------------------
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
processor = WhisperProcessor.from_pretrained(MODEL_ID)

# --------------------------------------------------
# Inspect architecture (useful for C++ porting)
# --------------------------------------------------
print("\n--- Model Architecture ---")
print(model)

# --------------------------------------------------
# Save raw PyTorch weights
# --------------------------------------------------
os.makedirs(RAW_MODEL_DIR, exist_ok=True)

weights_path = os.path.join(RAW_MODEL_DIR, "weights.pt")
torch.save(model.state_dict(), weights_path)

print(f"\n[Success] Raw weights saved to {weights_path}")

# --------------------------------------------------
# Export encoder to ONNX (reference / debugging)
# --------------------------------------------------
print("\nExporting encoder to ONNX...")

dummy_input = torch.randn(1, 80, 3000)  # log-mel features

onnx_path = os.path.join(RAW_MODEL_DIR, "encoder.onnx")

torch.onnx.export(
    model.model.encoder,
    dummy_input,
    onnx_path,
    input_names=["input_features"],
    output_names=["last_hidden_state"],
    opset_version=17,
    dynamic_axes={
        "input_features": {2: "time"},
        "last_hidden_state": {1: "time"}
    }
)

print(f"[Success] ONNX encoder saved to {onnx_path}")
