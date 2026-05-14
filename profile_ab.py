"""Profile AB search speed."""
import sys, json, time, torch, numpy as np
sys.path.insert(0, '.')
from penguinchess.rust_ffi import get_engine, ffi_ab_create, create_string_buffer, c_char_p, c_int32

eng = get_engine()
from penguinchess.ai.nnue import NNUE

model = NNUE(ft_dim=64, hidden_dim=256)
sd = torch.load('models/ab_nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
sd = sd['model_state'] if 'model_state' in sd else sd
model.load_state_dict(sd, strict=False)

cfg = json.dumps({'max_depth': 6, 'tt_size': 65536})
handle = ffi_ab_create(cfg)
handle.set_weights({k: v.cpu() for k, v in model.state_dict().items()})

# Test depth 2 search
state = json.dumps({
    'board': {'cells': [{'coord': {'q': -3 + i % 7, 'r': -3 + i // 7, 's': 0}, 'state': 'active', 'points': 1} for i in range(60)]},
    'pieces': [{'id': 4, 'alive': True, 'hex_idx': 0, 'hex_value': 1}, {'id': 5, 'alive': True, 'hex_idx': 1, 'hex_value': 1}],
    'scores': [0, 0], 'phase': 'placement', 'current_player': 0, 'placement_count': 0, 'episode_steps': 0, 'terminated': False
})

buf = create_string_buffer(4096)
t0 = time.time()
for _ in range(10):
    eng._lib.ffi_ab_search(handle._handle, c_char_p(state.encode()), c_int32(2), buf, c_int32(4096))
t1 = time.time()
print(f'depth 2: {(t1-t0)/10*1000:.1f}ms', flush=True)

t0 = time.time()
for _ in range(10):
    eng._lib.ffi_ab_search(handle._handle, c_char_p(state.encode()), c_int32(6), buf, c_int32(4096))
t1 = time.time()
print(f'depth 6: {(t1-t0)/10*1000:.1f}ms', flush=True)
print('DONE', flush=True)
