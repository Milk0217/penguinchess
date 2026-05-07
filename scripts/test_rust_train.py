"""Test Rust NNUE training via FFI."""
import sys, json, time, numpy as np, ctypes
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from penguinchess.rust_ffi import get_engine, ffi_ab_create, ffi_ab_generate_selfplay_data
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import POINTER, c_float, c_int32, c_char_p, create_string_buffer

# Generate self-play data (small set)
model = NNUE()
sd = {k:v.cpu() for k,v in model.state_dict().items()}
handle = ffi_ab_create(json.dumps({"max_depth":2,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2}))
handle.set_weights(sd)

data_path = 'models/nnue_trained/test_train.bin'
Path(data_path).parent.mkdir(exist_ok=True)
print('Generating 100 self-play games...')
t0 = time.time()
count = ffi_ab_generate_selfplay_data(handle, num_games=100, workers=4, output_path=data_path)
print(f'  {count} positions in {time.time()-t0:.0f}s')

# Call Rust training via FFI
engine = get_engine()
m2 = NNUE()
sd2 = {k:v.cpu() for k,v in m2.state_dict().items()}
import torch
ft_w = sd2['ft.weight'].cpu().numpy().T.ravel()
ft_b = sd2['ft.bias'].cpu().numpy().ravel()
fc1_w = sd2['fc1.weight'].cpu().numpy().ravel()
fc1_b = sd2['fc1.bias'].cpu().numpy().ravel()
fc2_w = sd2['fc2.weight'].cpu().numpy().ravel()
fc2_b = sd2['fc2.bias'].cpu().numpy().ravel()
fc3_w = sd2['fc3.weight'].cpu().numpy().ravel()
fc3_b = sd2['fc3.bias'].cpu().numpy().ravel()
flat = np.concatenate([ft_w, ft_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b]).astype(np.float32)
print(f'Flat array: {len(flat)} floats')

ptr = flat.ctypes.data_as(POINTER(c_float))
cfg = json.dumps({"lr":3e-4,"wd":1e-4,"batch_size":4096,"epochs":20,"max_norm":1.0})
out = create_string_buffer(4096)
print(f'Training {count} positions, 20 epochs...')
t0 = time.time()
rc = engine._lib.ffi_nnue_train(
    ptr, c_int32(len(flat)),
    c_char_p(data_path.encode('utf-8')),
    c_char_p(cfg.encode('utf-8')),
    out, c_int32(4096))
train_s = time.time() - t0
result = json.loads(out.value.decode('utf-8')) if out.value else {}
print(f'Result: {result}')
print(f'Training time: {train_s:.0f}s (rc={rc})')

# Verify weights changed
print(f'First 5 weights after: {flat[:5]}')

# Save model from flat array
# Read back the NNUE model weights from flat
ft_dim = 64; fc1_dim = 256; fc2_dim = 128
off = 0
ft_w = flat[off:off+360*ft_dim].reshape(360, ft_dim).T; off += 360*ft_dim
ft_b = flat[off:off+ft_dim]; off += ft_dim
fc1_w = flat[off:off+fc1_dim*(ft_dim*2+66)].reshape(fc1_dim, ft_dim*2+66); off += fc1_dim*(ft_dim*2+66)
fc1_b = flat[off:off+fc1_dim]; off += fc1_dim
fc2_w = flat[off:off+fc2_dim*fc1_dim].reshape(fc2_dim, fc1_dim); off += fc2_dim*fc1_dim
fc2_b = flat[off:off+fc2_dim]; off += fc2_dim
fc3_w = flat[off:off+fc2_dim].reshape(1, fc2_dim)
fc3_b = flat[off:off+1]

m3 = NNUE()
m3.ft.weight.data = torch.from_numpy(ft_w)
m3.ft.bias.data = torch.from_numpy(ft_b)
m3.fc1.weight.data = torch.from_numpy(fc1_w)
m3.fc1.bias.data = torch.from_numpy(fc1_b)
m3.fc2.weight.data = torch.from_numpy(fc2_w)
m3.fc2.bias.data = torch.from_numpy(fc2_b)
m3.fc3.weight.data = torch.from_numpy(fc3_w)
m3.fc3.bias.data = torch.from_numpy(fc3_b)
torch.save({'model_state':m3.state_dict(),'val_loss':result.get('best_loss',0)},
           'models/nnue_trained/rust_trained.pt')
print(f'Saved to models/nnue_trained/rust_trained.pt')
print(f'Done!')
