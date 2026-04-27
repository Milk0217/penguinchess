#!/usr/bin/env python3
"""Export AlphaZeroResNet to ONNX format for Rust-side inference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
from penguinchess.ai.alphazero_net import detect_net_arch

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
ALPHAZERO_DIR = MODELS_DIR / "alphazero"

# 从命令行参数选择导出目标，默认导出 best 检查点
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None,
                    help="Model path to export (default: auto-detect alphazero_resnet_large_best.pth)")
args = parser.parse_args()

# 自动检测并导出模型
if args.model:
    best_candidates = [Path(args.model)]
else:
    best_candidates = list(ALPHAZERO_DIR.glob("alphazero_resnet_large_best.pth"))
    if not best_candidates:
        best_candidates = list(ALPHAZERO_DIR.glob("alphazero_resnet_best.pth"))
    if not best_candidates:
        best_candidates = list(ALPHAZERO_DIR.glob("alphazero_resnet_iter_*.pth"))

if not best_candidates:
    print("Error: No ResNet model found")
    sys.exit(1)

# 取最新的
model_path = best_candidates[-1]
print(f"Loading model from: {model_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(model_path, map_location="cpu", weights_only=True)
NetClass = detect_net_arch(state)
net = NetClass().to("cpu")
net.load_state_dict(state)
net.eval()
print(f"Model: {NetClass.__name__} ({net.arch_name})")

# Create dummy input
dummy = torch.randn(1, 206)

# Export
onnx_path = ALPHAZERO_DIR / f"alphazero_{net.arch_name}_best.onnx"
torch.onnx.export(
    net,
    dummy,
    str(onnx_path),
    input_names=["obs"],
    output_names=["policy_logits", "value"],
    dynamic_axes={
        "obs": {0: "batch"},
        "policy_logits": {0: "batch"},
        "value": {0: "batch"},
    },
    opset_version=17,
)
print(f"Exported: {onnx_path}")

# Validate
import onnxruntime as ort
sess = ort.InferenceSession(str(onnx_path))
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]
print(f"  Input: {input_name} {sess.get_inputs()[0].shape}")
print(f"  Outputs: {output_names}")
for o in sess.get_outputs():
    print(f"    {o.name}: {o.shape}")

batch = torch.randn(32, 206).numpy()
results = sess.run(output_names, {input_name: batch})
print(f"  Inference OK: logits={results[0].shape}, values={results[1].shape}")

print("Done!")
