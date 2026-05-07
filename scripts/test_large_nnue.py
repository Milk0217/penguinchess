"""Smoke test: large NNUE model weights load into AB search"""
import json, time, sys
sys.path.insert(0, '.')
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import ffi_ab_create
from penguinchess.core import PenguinChessCore

model = NNUE()
print(f'NNUE: ft={model.ft.weight.shape} fc1={model.fc1.weight.shape} fc2={model.fc2.weight.shape} fc3={model.fc3.weight.shape}')
sd = {k: v.cpu() for k, v in model.state_dict().items()}

cfg = '{"max_depth":4,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2}'
h = ffi_ab_create(cfg)
ok = h.set_weights(sd)
print(f'Weights set: {ok}')

core = PenguinChessCore(seed=42).reset(seed=42)
for _ in range(6): core.step(core.get_legal_actions()[0])
smap = {'active': 'active', 'occupied': 'occupied', 'used': 'used', 'eliminated': 'eliminated'}
hx = [{'coord': {'q': x.q, 'r': x.r, 's': x.s}, 'state': smap.get(x.state, 'active'), 'points': x.points if x.state == 'active' else 0} for x in core.hexes]
px = [{'id': p.id, 'alive': p.alive, 'hex_idx': core.hexes.index(p.hex) if p.alive and p.hex else None, 'hex_value': p.hex_value} for p in core.pieces]
sj = json.dumps({'board': {'cells': hx, 'hex_map': {}, 'neighbors': []}, 'pieces': px, 'scores': core.players_scores, 'phase': 'movement', 'current_player': core.current_player, 'placement_count': core._placement_count, 'episode_steps': core._episode_steps, 'terminated': False, 'last_action': None})

t0 = time.time()
r = h.search(sj, max_depth=4)
t = time.time() - t0
print(f'AB depth 4: {r.get("nodes_searched")} nodes, {t*1000:.0f}ms')
h.free()
print('OK!')
