"""Verify weight transpose correctness between PyTorch and Rust.
Check if Rust AB search uses correct weights."""

import sys, json, time, torch, numpy as np
sys.path.insert(0, '.')
from penguinchess.rust_ffi import get_engine, ffi_ab_create, create_string_buffer, c_char_p, c_int32
from penguinchess.ai.nnue import NNUE

eng = get_engine()
model = NNUE(ft_dim=64, hidden_dim=256)
sd = torch.load('models/ab_nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
sd = sd['model_state'] if 'model_state' in sd else sd
model.load_state_dict(sd, strict=False)
model.eval()

# Python forward
state_json = json.dumps({
    'board': {'cells': [{'coord': {'q': -3 + i % 7, 'r': -3 + i // 7, 's': 0}, 'state': 'active', 'points': 1} for i in range(60)]},
    'pieces': [{'id': 4, 'alive': True, 'hex_idx': 0, 'hex_value': 1}, {'id': 5, 'alive': True, 'hex_idx': 1, 'hex_value': 1}],
    'scores': [0, 0], 'phase': 'placement', 'current_player': 0, 'placement_count': 0, 'episode_steps': 0, 'terminated': False
})

# To get the Rust NNUE eval, we call ffi_ab_search at depth 1 and check the score
cfg = json.dumps({'max_depth': 1, 'tt_size': 256, 'nnue_order_depth': 0, 'null_move': False, 'lmr_moves': 999})
handle = ffi_ab_create(cfg)
handle.set_weights({k: v.cpu() for k, v in model.state_dict().items()})

buf = create_string_buffer(4096)
for _ in range(5):
    eng._lib.ffi_ab_search(handle._handle, c_char_p(state_json.encode()), c_int32(1), buf, c_int32(4096))

# Parse result
result = json.loads(buf.value.decode())
print(f'Rust AB depth 1: score={result.get("score", "N/A"):.4f}', flush=True)

# Python forward
from penguinchess.ai.sparse_features import state_to_features
from penguinchess.core import PenguinChessCore
core = PenguinChessCore().reset(seed=42)
state_json_py = json.loads(state_json)
# Reconstruct core manually
sparse, dense = state_to_features([state_json])[0] if hasattr(state_json, '__iter__') else (state_to_features.__code__, None)

# Actually just use manual sparse/dense
hexes_fixed = np.zeros(60, dtype=np.float32)
for i in range(60):
    hexes_fixed[i] = 1.0 / 3.0
meta = np.array([0.0, 0.0, 0.0, 1.0/3, 1.0/3, 0.0], dtype=np.float32)
dense_t = torch.from_numpy(np.concatenate([hexes_fixed, meta])).float()

# Forward through Python NNUE (manual)
ft_w = model.ft_weight_gather  # (360, 64)
ft_b = model.ft_bias  # (64,)
stm_idx = [0]  # piece 4 at hex 0, P1
nstm_idx = [1]  # piece 5 at hex 1, P2
if stm_idx:
    idx_t = torch.tensor(stm_idx, dtype=torch.long)
    stm_acc = ft_b + ft_w[idx_t].sum(dim=0)
else:
    stm_acc = ft_b.clone()
if nstm_idx:
    idx_t = torch.tensor(nstm_idx, dtype=torch.long)
    nstm_acc = ft_b + ft_w[idx_t].sum(dim=0)
else:
    nstm_acc = ft_b.clone()

with torch.no_grad():
    x = torch.cat([stm_acc.clamp(0, 127), nstm_acc.clamp(0, 127), dense_t], dim=-1)
    v = torch.tanh(model.fc3(model.fc2(model.fc1(x).relu()).relu()))
    py_score = v.item()

print(f'Python NNUE:  {py_score:.4f}', flush=True)
print(f'Rust NNUE:    {result.get("score", 0):.4f}', flush=True)
print(f'Node count:   {result.get("nodes_searched", 0)}', flush=True)
print(f'Match: {abs(py_score - result.get("score", 0)) < 0.1}', flush=True)
