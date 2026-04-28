"""Quick eval of self-distilled NNUE vs Random"""
import json, time, torch, numpy as np
from penguinchess.core import PenguinChessCore
from penguinchess.rust_ffi import ffi_ab_create
from penguinchess.ai.nnue import NNUE

model = NNUE()
sd = torch.load('models/nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
loss = sd.get('val_loss', '?')
print(f'Loaded: nnue_gen_2.pt  val_loss={loss:.4f}')

handle = ffi_ab_create(json.dumps({'max_depth':2,'time_limit_ms':0,'tt_size':262144,'lmr_moves':3,'lmr_depth':1,'nnue_order_depth':2}))
handle.set_weights(model.state_dict())

def core_to_json(core):
    smap = {'active':'active','occupied':'occupied','used':'used','eliminated':'eliminated'}
    hexes = [{'coord':{'q':h.q,'r':h.r,'s':h.s},'state':smap.get(h.state,'active'),'points':h.points if h.state=='active' else 0} for h in core.hexes]
    pieces = [{'id':p.id,'alive':p.alive,'hex_idx':core.hexes.index(p.hex) if p.alive and p.hex else None,'hex_value':p.hex_value} for p in core.pieces]
    return json.dumps({'board':{'cells':hexes,'hex_map':{},'neighbors':[]},'pieces':pieces,'scores':core.players_scores,'phase':'placement' if core.phase=='placement' else 'movement','current_player':core.current_player,'placement_count':core._placement_count,'episode_steps':core._episode_steps,'terminated':False,'last_action':None})

print('Eval vs Random (Rust AB depth 2, 100 games)...')
wins, draws, losses = 0, 0, 0
for g in range(100):
    core = PenguinChessCore(seed=g*1000).reset(seed=g*1000)
    while True:
        legal = core.get_legal_actions()
        if not legal: break
        if core.current_player == 0:
            result = handle.search(core_to_json(core), max_depth=2)
            action = result.get('best_action', legal[0])
        else:
            action = legal[np.random.randint(len(legal))]
        core.step(action)
        if core.phase == 'gameover': break
    s1, s2 = core.players_scores
    if s1 > s2: wins += 1
    elif s2 > s1: losses += 1
    else: draws += 1
    if (g+1) % 50 == 0: print(f'  [{g+1}/100] W{wins} L{losses} D{draws}')

print(f'Result: {wins:.0%} win, {draws:.0%} draw, {losses:.0%} loss')
handle.free()
