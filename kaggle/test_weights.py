"""Verify weight size compatibility."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch, numpy as np
from penguinchess.rust_ffi import ffi_ab_create
from penguinchess.ai.nnue import NNUE

sd = torch.load("models/ab_nnue/nnue_gen_2.pt", map_location="cpu", weights_only=False)
sd = sd.get("model_state", sd) if isinstance(sd, dict) and "model_state" in sd else sd

ft_key = [k for k in sd if "ft.weight" in k][0]
fc1_key = [k for k in sd if "fc1.weight" in k][0]
print(f"Teacher: ft={sd[ft_key].shape} fc1={sd[fc1_key].shape}")

ft = sd[ft_key].cpu().numpy().T.ravel()
fb = sd["ft.bias"].cpu().numpy()
f1 = sd["fc1.weight"].cpu().numpy().T.ravel()
f1b = sd["fc1.bias"].cpu().numpy()
f2 = sd["fc2.weight"].cpu().numpy().T.ravel()
f2b = sd["fc2.bias"].cpu().numpy()
f3 = sd["fc3.weight"].cpu().numpy().T.ravel()
f3b = sd["fc3.bias"].cpu().numpy()
flat = np.concatenate([ft, fb, f1, f1b, f2, f2b, f3, f3b])
print(f"Flat arr size: {len(flat)}")

cfg = json.dumps({"max_depth": 4, "tt_size": 65536, "null_move": True})
h = ffi_ab_create(cfg)
try:
    h.set_weights({k: v.cpu() for k, v in sd.items()})
    print("set_weights: OK")
except Exception as ex:
    print(f"FAILED: {ex}")
