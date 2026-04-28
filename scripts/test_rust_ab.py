"""Quick test of Rust Alpha-Beta search via FFI."""
import json, time
import torch
from penguinchess.core import PenguinChessCore
from penguinchess.rust_ffi import ffi_ab_create
from penguinchess.ai.nnue import NNUE

def core_to_game_state_json(core):
    state_map = {'active': 'active', 'occupied': 'occupied', 'used': 'used', 'eliminated': 'eliminated'}
    hexes = [{
        'coord': {'q': h.q, 'r': h.r, 's': h.s},
        'state': state_map.get(h.state, 'Active'),
        'points': h.points if h.state == 'active' else 0,
    } for h in core.hexes]
    pieces = [{
        'id': p.id,
        'alive': p.alive,
        'hex_idx': core.hexes.index(p.hex) if p.alive and p.hex else None,
        'hex_value': p.hex_value,
    } for p in core.pieces]
    phase = 'placement' if core.phase == 'placement' else 'movement'
    return json.dumps({
        'board': {'cells': hexes, 'hex_map': {}, 'neighbors': []},
        'pieces': pieces,
        'scores': core.players_scores,
        'phase': phase,
        'current_player': core.current_player,
        'placement_count': core._placement_count,
        'episode_steps': core._episode_steps,
        'terminated': getattr(core, '_terminated', False),
        'last_action': None,
    })

print("Creating search handle...")
handle = ffi_ab_create(json.dumps({
    'max_depth': 3, 'time_limit_ms': 0, 'tt_size': 1 << 16, 'lmr_moves': 3, 'lmr_depth': 1
}))
print(f"Handle: {handle._handle}")

print("Loading NNUE weights...")
model = NNUE()
sd = torch.load('models/nnue/nnue_gen_1.pt', map_location='cpu', weights_only=False)
model.load_state_dict(sd['model_state'])
ok = handle.set_weights(model.state_dict())
print(f"Weights set: {ok}")

print("Creating game position...")
core = PenguinChessCore(seed=42).reset(seed=42)
for _ in range(6):
    legal = core.get_legal_actions()
    core.step(legal[0])

state_json = core_to_game_state_json(core)
print(f"Phase: {core.phase}, legal moves: {len(core.get_legal_actions())}")

for depth in [1, 2, 3, 4]:
    t0 = time.time()
    result = handle.search(state_json, max_depth=depth)
    elapsed = time.time() - t0
    print(f"Depth {depth}: action={result.get('best_action')}, score={result.get('score', '?'):.3f}, "
          f"nodes={result.get('nodes_searched')}, time={elapsed*1000:.0f}ms, "
          f"depth_reached={result.get('depth_reached')}")
    if 'error' in result:
        print(f"  ERROR: {result['error']}")
        break

handle.free()
print("Done!")
