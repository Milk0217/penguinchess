"""Debug: check ffi_ab_set_weights error"""
import json, sys, ctypes
sys.path.insert(0, '.')
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import get_engine, POINTER, c_float, c_int32, create_string_buffer
import numpy as np

model = NNUE()
sd = {k: v.cpu() for k, v in model.state_dict().items()}
ft_w = sd['ft.weight'].cpu().numpy().T.ravel()
ft_b = sd['ft.bias'].cpu().numpy().ravel()
fc1_w = sd['fc1.weight'].cpu().numpy().ravel()
fc1_b = sd['fc1.bias'].cpu().numpy().ravel()
fc2_w = sd['fc2.weight'].cpu().numpy().ravel()
fc2_b = sd['fc2.bias'].cpu().numpy().ravel()
fc3_w = sd['fc3.weight'].cpu().numpy().ravel()
fc3_b = sd['fc3.bias'].cpu().numpy().ravel()
flat = np.concatenate([ft_w, ft_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b]).astype(np.float32)

eng = get_engine()
cfg = '{"max_depth":1,"tt_size":256,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2}'
buf = create_string_buffer(1024)
rc1 = eng._lib.ffi_ab_create(c_void_p if hasattr(ctypes, 'c_void_p') else ctypes.c_void_p)(None)  # skip this
# Actually use the Python wrapper properly
from penguinchess.rust_ffi import ffi_ab_create
h = ffi_ab_create(cfg)

ptr = flat.ctypes.data_as(POINTER(c_float))
out = create_string_buffer(4096)
rc = eng._lib.ffi_ab_set_weights(
    c_int32(h._handle), ptr, c_int32(len(flat)), out, c_int32(4096))
result = json.loads(out.value.decode('utf-8')) if out.value else {}
print(f'rc={rc}, result={result}')
h.free()
