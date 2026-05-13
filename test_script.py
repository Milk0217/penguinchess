"""Test Rust-native data generation (AB search labels from random games)."""
import time, torch, sys
from penguinchess.rust_ffi import ffi_ab_create, ffi_ab_generate_random_data
from penguinchess.ai.nnue import NNUE

m = NNUE()
sd = torch.load('models/nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
m.load_state_dict(sd)

cfg = '{"max_depth":4,"tt_size":262144,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2}'
handle = ffi_ab_create(cfg)
handle.set_weights(m.state_dict())

t0 = time.time()
count = ffi_ab_generate_random_data(handle, num_games=500, workers=4, output_path='data_nnue.bin')
elapsed = time.time() - t0
print(f'Generated {count} positions in {elapsed:.0f}s ({elapsed/max(count,1)*1e3:.1f}ms/pos, {elapsed/500:.1f}s/game)')
handle.free()
