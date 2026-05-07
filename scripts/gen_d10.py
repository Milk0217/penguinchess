"""Gen from scratch on depth 10 data."""
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
h = ffi_ab_create(json.dumps({'max_depth': 10, 'tt_size': 65536, 'lmr_moves': 3,
                               'lmr_depth': 1, 'nnue_order_depth': 2, 'null_move': True}))
h.set_weights(d)
print('500 games, depth 10, 8 workers...', flush=True)
t0 = time.time()
cnt = ffi_ab_generate_selfplay_data(h, num_games=500, workers=8, output_path='models/nnue_trained/sp_d10.bin')
print(f'  {cnt} pos in {time.time()-t0:.0f}s', flush=True)
del h

m2 = NNUE()
w = lambda k: m2.state_dict()[k].cpu().numpy()
flat = np.concatenate([w('ft.weight').T.ravel(), w('ft.bias').ravel(),
                       w('fc1.weight').ravel(), w('fc1.bias').ravel(),
                       w('fc2.weight').ravel(), w('fc2.bias').ravel(),
                       w('fc3.weight').ravel(), w('fc3.bias').ravel()]).astype(np.float32)
eng = get_engine(); ptr = flat.ctypes.data_as(POINTER(c_float))
t0 = time.time()
eng._lib.ffi_nnue_train(ptr, c_int32(len(flat)),
    c_char_p(b'models/nnue_trained/sp_d10.bin'),
    c_char_p(json.dumps({'lr': 3e-4, 'wd': 1e-4, 'batch_size': 4096, 'epochs': 50}).encode()),
    create_string_buffer(4096), 4096)
print(f'Train: {time.time()-t0:.0f}s', flush=True)

off = 0; FD, HD, HD2 = 64, 256, 128; idim = FD*2+66
m3 = NNUE()
m3.ft.weight.data = torch.from_numpy(flat[off:off+360*FD].reshape(FD, 360)); off += 360*FD
m3.ft.bias.data = torch.from_numpy(flat[off:off+FD]); off += FD
m3.fc1.weight.data = torch.from_numpy(flat[off:off+HD*idim].reshape(HD, idim)); off += HD*idim
m3.fc1.bias.data = torch.from_numpy(flat[off:off+HD]); off += HD
m3.fc2.weight.data = torch.from_numpy(flat[off:off+HD2*HD].reshape(HD2, HD)); off += HD2*HD
m3.fc2.bias.data = torch.from_numpy(flat[off:off+HD2]); off += HD2
m3.fc3.weight.data = torch.from_numpy(flat[off:off+HD2].reshape(1, HD2)); off += HD2
m3.fc3.bias.data = torch.from_numpy(flat[off:off+1])
torch.save({'model_state': m3.state_dict()}, 'models/nnue_trained/gen_d10.pt')
print('Saved gen_d10.pt!')
