"""Quick test of Rust Alpha-Beta search via FFI (Rust NNUE, no callback)."""
import json, time
import torch
from penguinchess.core import PenguinChessCore
from penguinchess.rust_ffi import ffi_ab_create
from penguinchess.ai.nnue import NNUE

def core_to_game_state_json(core):
    state_map = {'active': 'active', 'occupied': 'occupied', 'used': 'used', 'eliminated': 'eliminated'}
    hexes = [{'coord':{'q':h.q,'r':h.r,'s':h.s},'state':state_map.get(h.state,'active'),'points':h.points if h.state=='active' else 0} for h in core.hexes]
    pieces = [{'id':p.id,'alive':p.alive,'hex_idx':core.hexes.index(p.hex) if p.alive and p.hex else None,'hex_value':p.hex_value} for p in core.pieces]
    phase = 'placement' if core.phase == 'placement' else 'movement'
    return json.dumps({'board':{'cells':hexes,'hex_map':{},'neighbors':[]},'pieces':pieces,'scores':core.players_scores,'phase':phase,'current_player':core.current_player,'placement_count':core._placement_count,'episode_steps':core._episode_steps,'terminated':False,'last_action':None})

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

print("Loading NNUE model and Rust weights...")
model = NNUE()
sd = torch.load('models/nnue/nnue_gen_1.pt', map_location='cpu', weights_only=False)
model.load_state_dict(sd['model_state'])

print("Creating search handle (Rust CPU mode)...")
handle = ffi_ab_create('{"max_depth":10,"time_limit_ms":0,"tt_size":262144,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2}')
ok = handle.set_weights(model.state_dict())
print(f"Weights set: {ok}")

core = PenguinChessCore(seed=42).reset(seed=42)
for _ in range(6): core.step(core.get_legal_actions()[0])

state_json = core_to_game_state_json(core)
print(f"Phase: {core.phase}, legal: {len(core.get_legal_actions())}")

for depth in [2, 3, 4, 5]:
    t0 = time.time()
    result = handle.search(state_json, max_depth=depth)
    elapsed = time.time() - t0
    nps = result['nodes_searched'] / elapsed if 'nodes_searched' in result else 0
    print(f"Depth {depth}: {result.get('nodes_searched')} nodes, {elapsed*1000:.0f}ms, {nps:.0f} nps")
    if 'error' in result:
        print(f"  ERROR: {result['error']}")
        break

handle.free()
print("Done!")
