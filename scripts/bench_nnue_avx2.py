"""Mini benchmark for AVX2 NNUE."""
import json, time, torch, sys
from penguinchess.core import PenguinChessCore
from penguinchess.rust_ffi import ffi_ab_create
from penguinchess.ai.nnue import NNUE

core = PenguinChessCore(seed=42).reset(seed=42)
for _ in range(6): core.step(core.get_legal_actions()[0])
model = NNUE()
sd = torch.load('models/nnue/nnue_gen_1.pt', map_location='cpu', weights_only=False)
model.load_state_dict(sd['model_state'])

h = ffi_ab_create(json.dumps({'max_depth':1,'time_limit_ms':0,'tt_size':256,'lmr_moves':3,'lmr_depth':1}))
ok = h.set_weights(model.state_dict())
print('Weights set:', ok); sys.stdout.flush()

smap = {'active': 'active', 'occupied': 'occupied', 'used': 'used', 'eliminated': 'eliminated'}
hexes = [{'coord':{'q':h.q,'r':h.r,'s':h.s},'state':smap.get(h.state,'active'),'points':h.points if h.state=='active' else 0} for h in core.hexes]
pieces = [{'id':p.id,'alive':p.alive,'hex_idx':core.hexes.index(p.hex) if p.alive and p.hex else None,'hex_value':p.hex_value} for p in core.pieces]
state_json = json.dumps({'board':{'cells':hexes,'hex_map':{},'neighbors':[]},'pieces':pieces,'scores':core.players_scores,'phase':'movement','current_player':core.current_player,'placement_count':core._placement_count,'episode_steps':core._episode_steps,'terminated':False,'last_action':None})

for depth in [1,2]:
    t0 = time.time()
    result = h.search(state_json, max_depth=depth)
    elapsed = time.time() - t0
    print(f'Depth {depth}: {result.get("nodes_searched")} nodes, {elapsed*1000:.0f}ms')
h.free()
