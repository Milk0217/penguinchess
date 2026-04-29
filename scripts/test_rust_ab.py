"""Benchmark Rust Alpha-Beta search with null-move + Lazy SMP."""
import json, time
import torch
from penguinchess.core import PenguinChessCore
from penguinchess.rust_ffi import ffi_ab_create
from penguinchess.ai.nnue import NNUE

def core_to_game_state_json(core):
    sm = {'active': 'active', 'occupied': 'occupied', 'used': 'used', 'eliminated': 'eliminated'}
    h = [{'coord':{'q':x.q,'r':x.r,'s':x.s},'state':sm.get(x.state,'active'),'points':x.points if x.state=='active' else 0} for x in core.hexes]
    p = [{'id':x.id,'alive':x.alive,'hex_idx':core.hexes.index(x.hex) if x.alive and x.hex else None,'hex_value':x.hex_value} for x in core.pieces]
    ph = 'placement' if core.phase == 'placement' else 'movement'
    return json.dumps({'board':{'cells':h,'hex_map':{},'neighbors':[]},'pieces':p,'scores':core.players_scores,'phase':ph,'current_player':core.current_player,'placement_count':core._placement_count,'episode_steps':core._episode_steps,'terminated':False,'last_action':None})

print("Loading model...")
model = NNUE()
sd = torch.load('models/nnue/nnue_gen_1.pt', map_location='cpu', weights_only=False)
model.load_state_dict(sd['model_state'])

core = PenguinChessCore(seed=42).reset(seed=42)
for _ in range(6): core.step(core.get_legal_actions()[0])
sj = core_to_game_state_json(core)

# Single-thread benchmark
handle = ffi_ab_create('{"max_depth":10,"tt_size":262144,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2}')
handle.set_weights(model.state_dict())
print(f"Single thread: {len(core.get_legal_actions())} legal moves")
for d in [2,3,4,5,6]:
    t0=time.time(); r=handle.search(sj,max_depth=d); t=time.time()-t0
    print(f"  D{d}: {r.get('nodes_searched')}n {t*1000:.0f}ms")
handle.free()

# Lazy SMP benchmark (deeper)
for nthr in [4]:
    h = ffi_ab_create(f'{{"max_depth":12,"tt_size":262144,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2,"num_threads":{nthr}}}')
    h.set_weights(model.state_dict())
    print(f"\nLazy SMP {nthr} threads:")
    for d in [6,7,8,9,10]:
        t0=time.time(); r=h.search(sj,max_depth=d); t=time.time()-t0
        print(f"  D{d}: move={r.get('best_action')} score={r.get('score'):+.3f} {t*1000:.0f}ms")
        if t>30: print(f"    (slow, stopping)"); break
    h.free()
