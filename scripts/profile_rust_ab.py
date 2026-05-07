"""Profile the Rust NNUE evaluation speed."""
import json, time
import torch
import numpy as np
from penguinchess.rust_ffi import ffi_ab_create, get_engine
from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE

# Get state
core = PenguinChessCore(seed=42).reset(seed=42)
for _ in range(6):
    core.step(core.get_legal_actions()[0])

def core_to_state_json(core):
    state_map = {'active': 'active', 'occupied': 'occupied', 'used': 'used', 'eliminated': 'eliminated'}
    hexes = [{
        'coord': {'q': h.q, 'r': h.r, 's': h.s},
        'state': state_map.get(h.state, 'active'),
        'points': h.points if h.state == 'active' else 0,
    } for h in core.hexes]
    pieces = [{
        'id': p.id, 'alive': p.alive,
        'hex_idx': core.hexes.index(p.hex) if p.alive and p.hex else None,
        'hex_value': p.hex_value,
    } for p in core.pieces]
    phase = 'placement' if core.phase == 'placement' else 'movement'
    return json.dumps({
        'board': {'cells': hexes, 'hex_map': {}, 'neighbors': []},
        'pieces': pieces, 'scores': core.players_scores, 'phase': phase,
        'current_player': core.current_player, 'placement_count': core._placement_count,
        'episode_steps': core._episode_steps, 'terminated': False, 'last_action': None,
    })

state_json = core_to_state_json(core)

# Create handle
handle = ffi_ab_create(json.dumps({'max_depth': 1, 'time_limit_ms': 0, 'tt_size': 256, 'lmr_moves': 3, 'lmr_depth': 1}))
model = NNUE()
sd = torch.load('models/nnue/nnue_gen_1.pt', map_location='cpu', weights_only=False)
model.load_state_dict(sd['model_state'])
handle.set_weights(model.state_dict())

# Profile: search at depth 1 (1 node, but does 19 evals in order_moves + search)
t0 = time.time()
for _ in range(5):
    r = handle.search(state_json, max_depth=1)
t = (time.time() - t0) / 5
print(f"Depth 1 avg: {t*1000:.1f}ms")

# Profile: just the search overhead (state deserialization + init)
# Create handle that has empty weights (no good evaluations but still shows overhead)
h2 = ffi_ab_create(json.dumps({'max_depth': 1, 'time_limit_ms': 0, 'tt_size': 256, 'lmr_moves': 3, 'lmr_depth': 1}))
t0 = time.time()
for _ in range(20):
    r = h2.search(state_json, max_depth=1)
t = (time.time() - t0) / 20
print(f"Depth 1 (empty weights): {t*1000:.1f}ms")

h2.free()
handle.free()
print("Done")
