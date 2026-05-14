"""Evaluate large NNUE model vs Random using AB search (depth 4)."""
import sys, json, time, torch, random as rnd
sys.path.insert(0, '.')
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import ffi_ab_create, get_engine
from penguinchess.core import PenguinChessCore

model = NNUE()
sd = torch.load('models/nnue/nnue_gen_4.pt', map_location='cpu', weights_only=False)
sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
model.load_state_dict(sd, strict=False)
model.eval()
print(f'Model: {sum(p.numel() for p in model.parameters()):,} params', flush=True)

cfg = json.dumps({'max_depth': 4, 'tt_size': 65536, 'nnue_order_depth': 2, 'null_move': True})
handle = ffi_ab_create(cfg)
handle.set_weights({k: v.cpu() for k, v in model.state_dict().items()})

def ab_action(core, legal):
    """Use AB search to pick best action."""
    hx = [{'coord':{'q':x.q,'r':x.r,'s':x.s},'state':'active' if x.state=='active' else 'occupied' if x.state=='occupied' else x.state,'points':x.points if x.state in ('active','occupied') else 0} for x in core.hexes]
    px = [{'id':p.id,'alive':p.alive,'hex_idx':core.hexes.index(p.hex) if p.alive and p.hex else None,'hex_value':p.hex_value} for p in core.pieces]
    sj = json.dumps({'board':{'cells':hx},'pieces':px,'scores':core.players_scores,'phase':core.phase,'current_player':core.current_player,'placement_count':core._placement_count,'episode_steps':core._episode_steps,'terminated':core._terminated})
    r = handle.search(sj, max_depth=4)
    if isinstance(r, dict) and 'best_action' in r:
        a = r['best_action']
        if a in legal: return a
    return max(legal, key=lambda x: core.hexes[x].points if x < len(core.hexes) and core.hexes[x].state == 'active' else 0) if legal else 0

w = d = l = 0
t0 = time.time()
for g in range(100):
    core = PenguinChessCore(seed=g*997+42).reset(seed=g*997+42)
    for _ in range(6):
        leg = core.get_legal_actions()
        if leg: core.step(rnd.choice(leg))
    while not core._terminated and core._episode_steps < 200:
        leg = core.get_legal_actions()
        if not leg: break
        if core.current_player == 0:
            action = ab_action(core, leg)
        else:
            action = rnd.choice(leg)
        core.step(action)
    s1, s2 = core.players_scores
    if s1 > s2: w+=1; print('W', end='', flush=True)
    elif s1==s2: d+=1; print('D', end='', flush=True)
    else: l+=1; print('L', end='', flush=True)

elapsed = time.time() - t0
print(f'\nFINAL: W={w} L={l} D={d} ({100*w//100}% vs Random) {elapsed:.0f}s', flush=True)
