"""Debug large NNUE weight loading"""
import json, sys
sys.path.insert(0, '.')
from penguinchess.ai.nnue import NNUE, FT_DIM, HIDDEN_DIM
import numpy as np

model = NNUE()
sd = {k: v.cpu() for k, v in model.state_dict().items()}

ft = sd['ft.weight'].cpu().numpy()
fc1 = sd['fc1.weight'].cpu().numpy()
fc2 = sd['fc2.weight'].cpu().numpy()
fc3 = sd['fc3.weight'].cpu().numpy()

print(f'PyTorch shapes: ft={ft.shape} fc1={fc1.shape} fc2={fc2.shape} fc3={fc3.shape}')
print(f'FT_DIM={FT_DIM} HIDDEN_DIM={HIDDEN_DIM}')
print(f'360*ft_dim = {360*FT_DIM}')

# Sizes
ft_sz = 360 * FT_DIM
ft_b_sz = FT_DIM
fc1_sz = fc1.shape[0] * fc1.shape[1]
fc1_b_sz = fc1.shape[0]
fc2_sz = fc2.shape[0] * fc2.shape[1]
fc2_b_sz = fc2.shape[0]
fc3_sz = fc3.shape[0] * fc3.shape[1]
fc3_b_sz = fc3.shape[0]
total = ft_sz + ft_b_sz + fc1_sz + fc1_b_sz + fc2_sz + fc2_b_sz + fc3_sz + fc3_b_sz
print(f'Sizes: ft={ft_sz}+{ft_b_sz} fc1={fc1_sz}+{fc1_b_sz} fc2={fc2_sz}+{fc2_b_sz} fc3={fc3_sz}+{fc3_b_sz} = {total}')

ft_w = sd['ft.weight'].cpu().numpy().T.ravel()
ft_b = sd['ft.bias'].cpu().numpy().ravel()
fc1_w = sd['fc1.weight'].cpu().numpy().ravel()
fc1_b = sd['fc1.bias'].cpu().numpy().ravel()
fc2_w = sd['fc2.weight'].cpu().numpy().ravel()
fc2_b = sd['fc2.bias'].cpu().numpy().ravel()
fc3_w = sd['fc3.weight'].cpu().numpy().ravel()
fc3_b = sd['fc3.bias'].cpu().numpy().ravel()

flat = np.concatenate([ft_w, ft_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b])
print(f'Flat array: {len(flat)} floats')

# Rust total_floats()
rust_total = 360*FT_DIM + FT_DIM + HIDDEN_DIM*(FT_DIM*2+66) + HIDDEN_DIM + 256*HIDDEN_DIM + 256 + 256 + 1
# Wait, Rust FC2_DIM=256, FC1_DIM=512=HIDDEN_DIM
# total_floats = 360*FT_DIM + FT_DIM + FC1_DIM*INPUT_DIM + FC1_DIM + FC2_DIM*FC1_DIM + FC2_DIM + 1*FC2_DIM + 1
fc1_d = HIDDEN_DIM
fc2_d = 256
input_d = FT_DIM*2+66  # 322
total_floats = 360*FT_DIM + FT_DIM + fc1_d*input_d + fc1_d + fc2_d*fc1_d + fc2_d + 1*fc2_d + 1
print(f'Rust total_floats: {total_floats}')
print(f'Match: {len(flat) == total_floats}')
from penguinchess.rust_ffi import ffi_ab_create
cfg = '{"max_depth":1,"tt_size":256,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2}'
h = ffi_ab_create(cfg)
ok = h.set_weights(sd)
print(f'set_weights: {ok}')
h.free()
