"""Depth 10 with adaptive depth + 8 workers."""
import sys, time, json, torch, numpy as np
sys.path.insert(0, '.')
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import ffi_ab_create, ffi_ab_generate_selfplay_data, get_engine
from penguinchess.rust_ffi import POINTER, c_float, c_int32, c_char_p, create_string_buffer

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

print('Generating 500 games (adaptive depth, 8 workers)...', flush=True)
t0 = time.time()
cnt = ffi_ab_generate_selfplay_data(h, num_games=500, workers=8,
                                     output_path='models/nnue_trained/sp_adaptive.bin')
print(f'  {cnt} pos in {time.time()-t0:.0f}s', flush=True)
del h

# Training
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
    c_char_p(b'models/nnue_trained/sp_adaptive.bin'),
    c_char_p(json.dumps({'lr': 3e-4, 'wd': 1e-4, 'batch_size': 4096, 'epochs': 30}).encode()),
    out, c_int32(4096))
print(f'Train: {time.time()-t0:.0f}s', flush=True)
print(f'Result: {json.loads(out.value.decode()) if out.value else {}}', flush=True)

# Save model
off = 0; FD, HD, HD2 = 64, 256, 128; idim = FD*2+66
fw = flat[off:off+360*FD].reshape(360, FD).T; off += 360*FD
fb = flat[off:off+FD]; off += FD
c1w = flat[off:off+HD*idim].reshape(HD, idim); off += HD*idim
c1b = flat[off:off+HD]; off += HD
c2w = flat[off:off+HD2*HD].reshape(HD2, HD); off += HD2*HD
c2b = flat[off:off+HD2]; off += HD2
c3w = flat[off:off+HD2].reshape(1, HD2); c3b = flat[off:off+1]
m2 = NNUE()
m2.ft.weight.data = torch.from_numpy(fw); m2.ft.bias.data = torch.from_numpy(fb)
m2.fc1.weight.data = torch.from_numpy(c1w); m2.fc1.bias.data = torch.from_numpy(c1b)
m2.fc2.weight.data = torch.from_numpy(c2w); m2.fc2.bias.data = torch.from_numpy(c2b)
m2.fc3.weight.data = torch.from_numpy(c3w); m2.fc3.bias.data = torch.from_numpy(c3b)
torch.save({'model_state': m2.state_dict()}, 'models/nnue_trained/adaptive_trained.pt')
print('Saved!', flush=True)
