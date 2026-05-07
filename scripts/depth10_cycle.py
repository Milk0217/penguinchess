"""Depth 10 self-play data gen + Rust training."""
import sys, time, json, torch
sys.path.insert(0, '.')
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import ffi_ab_create, ffi_ab_generate_selfplay_data, get_engine
from penguinchess.rust_ffi import POINTER, c_float, c_int32, c_char_p, create_string_buffer
import numpy as np

m = NNUE()
sd = torch.load('models/nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
m.load_state_dict(sd, strict=False)
d = {k: v.cpu() for k, v in m.state_dict().items()}

cfg_j = json.dumps({'max_depth': 10, 'tt_size': 65536, 'lmr_moves': 3,
                     'lmr_depth': 1, 'nnue_order_depth': 2, 'null_move': True,
                     'num_threads': 1, 'reuse': False})
h = ffi_ab_create(cfg_j)
h.set_weights(d)

t0 = time.time()
cnt = ffi_ab_generate_selfplay_data(h, num_games=100, workers=4,
                                     output_path='models/nnue_trained/sp_d10.bin')
print(f'Data: {cnt} pos in {time.time()-t0:.0f}s')
del h

# Rust training
ft_w = m.state_dict()['ft.weight'].cpu().numpy().T.ravel()
ft_b = m.state_dict()['ft.bias'].cpu().numpy().ravel()
fc1_w = m.state_dict()['fc1.weight'].cpu().numpy().ravel()
fc1_b = m.state_dict()['fc1.bias'].cpu().numpy().ravel()
fc2_w = m.state_dict()['fc2.weight'].cpu().numpy().ravel()
fc2_b = m.state_dict()['fc2.bias'].cpu().numpy().ravel()
fc3_w = m.state_dict()['fc3.weight'].cpu().numpy().ravel()
fc3_b = m.state_dict()['fc3.bias'].cpu().numpy().ravel()
flat = np.concatenate([ft_w, ft_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b]).astype(np.float32)

eng = get_engine()
ptr = flat.ctypes.data_as(POINTER(c_float))
out = create_string_buffer(4096)
t0 = time.time()
rc = eng._lib.ffi_nnue_train(
    ptr, c_int32(len(flat)),
    c_char_p(b'models/nnue_trained/sp_d10.bin'),
    c_char_p(json.dumps({'lr': 3e-4, 'wd': 1e-4, 'batch_size': 4096, 'epochs': 30}).encode()),
    out, c_int32(4096))
print(f'Train: {time.time()-t0:.0f}s  rc={rc}')
print(f'Result: {json.loads(out.value.decode()) if out.value else {}}')
