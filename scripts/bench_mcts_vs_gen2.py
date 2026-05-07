"""Head-to-head: MCTS+NNUE best model vs gen_2 vs Random"""
import json, random, torch, time, sys
sys.path.insert(0,'.')
from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue_mcts import NNUEMCTSModel
from penguinchess.rust_ffi import get_engine, RustStatefulGame, NNUEMCTSNative, ffi_ab_create

def core_to_json(core):
    smap = {'active':'active','occupied':'occupied','used':'used','eliminated':'eliminated'}
    h = [{'coord':{'q':x.q,'r':x.r,'s':x.s},'state':smap.get(x.state,'active'),'points':x.points if x.state=='active' else 0} for x in core.hexes]
    p = [{'id':x.id,'alive':x.alive,'hex_idx':core.hexes.index(x.hex) if x.alive and x.hex else None,'hex_value':x.hex_value} for x in core.pieces]
    ph = 'placement' if core.phase=='placement' else 'movement'
    return json.dumps({'board':{'cells':h,'hex_map':{},'neighbors':[]},'pieces':p,'scores':core.players_scores,'phase':ph,'current_player':core.current_player,'placement_count':core._placement_count,'episode_steps':core._episode_steps,'terminated':False,'last_action':None})

device = 'cuda' if torch.cuda.is_available() else 'cpu'
engine = get_engine()

# Load MCTS+NNUE best model
mcts_model = NNUEMCTSModel().to(device).eval()
sd = torch.load('models/nnue_mcts/nnue_mcts_best.pt', map_location='cpu', weights_only=False)
sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
mcts_model.load_state_dict(sd, strict=False)
native_sd = {k: v.cpu() for k, v in mcts_model.state_dict().items()}
native_mcts = NNUEMCTSNative(native_sd)
print(f'MCTS+NNUE model loaded (WR={sd.get("win_rate",0)*100:.0f}%)')

# Load gen_2 (AB+NNUE)
gen2_model = __import__('penguinchess.ai.nnue', fromlist=['NNUE']).NNUE()
sd2 = torch.load('models/nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
sd2 = sd2.get('model_state', sd2) if isinstance(sd2, dict) and 'model_state' in sd2 else sd2
gen2_model.load_state_dict(sd2)
ab_cfg = '{"max_depth":4,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2}'
ab_handle = ffi_ab_create(ab_cfg)
ab_handle.set_weights(gen2_model.state_dict())
print(f'gen_2 (AB depth 4) loaded')

def play_vs_mcts(seed, model_handle, num_sims):
    """MCTS+NNUE vs gen_2 AB search"""
    core = PenguinChessCore(seed=seed).reset(seed=seed)
    rgame = RustStatefulGame(engine, seed)
    for _ in range(6):
        leg = core.get_legal_actions()
        if leg: a = random.choice(leg); core.step(a); rgame.step(a)
    while not core._terminated and core._episode_steps < 200:
        legal = core.get_legal_actions()
        if not legal: break
        if core.current_player == 0:
            ac = model_handle.search(rgame.handle, num_simulations=num_sims, c_puct=1.4)
            action = max((int(k) for k in ac.keys()), default=random.choice(legal)) if ac else random.choice(legal)
        else:
            r = ab_handle.search(core_to_json(core), max_depth=4)
            action = r.get('best_action', 0)
        if action in legal: core.step(action); rgame.step(action)
        elif legal: core.step(legal[0]); rgame.step(legal[0])
    s1, s2 = core.players_scores
    if s1 > s2: return 1
    elif s1 == s2: return 0
    return -1

def play_vs_random(seed, model_handle, num_sims):
    """MCTS+NNUE vs Random"""
    core = PenguinChessCore(seed=seed).reset(seed=seed)
    rgame = RustStatefulGame(engine, seed)
    for _ in range(6):
        leg = core.get_legal_actions()
        if leg: a = random.choice(leg); core.step(a); rgame.step(a)
    while not core._terminated and core._episode_steps < 200:
        legal = core.get_legal_actions()
        if not legal: break
        if core.current_player == 0:
            ac = model_handle.search(rgame.handle, num_simulations=num_sims, c_puct=1.4)
            action = max((int(k) for k in ac.keys()), default=random.choice(legal)) if ac else random.choice(legal)
        else:
            action = random.choice(legal)
        if action in legal: core.step(action); rgame.step(action)
        elif legal: core.step(legal[0]); rgame.step(legal[0])
    s1, s2 = core.players_scores
    if s1 > s2: return 1
    elif s1 == s2: return 0
    return -1

N = 100
print(f'\nMCTS+NNUE vs gen_2 (AB depth 4) — {N} games...')
t0 = time.time()
w = d = l = 0
for g in range(N):
    r = play_vs_mcts(g * 9973 + 42, native_mcts, 100)
    if r > 0: w += 1
    elif r == 0: d += 1
    else: l += 1
    if (g+1) % 25 == 0: print(f'  [{g+1}/{N}] W:{w} L:{l} D:{d}')
t = time.time() - t0
pct = w/N*100
print(f'\nvs gen_2: {w}/{N} wins ({pct:.0f}%)  {d} draws  {l} losses  ({t:.0f}s)')

print(f'\nMCTS+NNUE vs Random — {N} games...')
t0 = time.time()
w2 = d2 = l2 = 0
for g in range(N):
    r = play_vs_random(g * 9973 + 999, native_mcts, 100)
    if r > 0: w2 += 1
    elif r == 0: d2 += 1
    else: l2 += 1
    if (g+1) % 25 == 0: print(f'  [{g+1}/{N}] W:{w2} L:{l2} D:{d2}')
t = time.time() - t0
pct2 = w2/N*100
print(f'\nvs Random: {w2}/{N} wins ({pct2:.0f}%)  {d2} draws  {l2} losses  ({t:.0f}s)')

native_mcts.free()
ab_handle.free()
