"""Benchmark large MCTS+NNUE (242K) vs Random + gen_2"""
import json, random, torch, time, sys
sys.path.insert(0,'.')
from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue_mcts import NNUEMCTSModel
from penguinchess.rust_ffi import get_engine, RustStatefulGame, NNUEMCTSNative, ffi_ab_create
from penguinchess.ai.nnue import NNUE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
engine = get_engine()

model = NNUEMCTSModel(hidden_dim=512, ft_dim=128).to(device).eval()
ckpt = torch.load('models/nnue_mcts_large/nnue_mcts_best.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
model.load_state_dict(sd, strict=False)
native_sd = {k: v.cpu() for k, v in model.state_dict().items()}
native = NNUEMCTSNative(native_sd)
print(f'Large model loaded, best WR={sd.get("win_rate",0)*100:.0f}%')

# gen_2
gen2 = NNUE()
sd2 = torch.load('models/nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
sd2 = sd2.get('model_state', sd2) if isinstance(sd2, dict) and 'model_state' in sd2 else sd2
gen2.load_state_dict(sd2)
ab = ffi_ab_create('{"max_depth":4,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2}')
ab.set_weights(gen2.state_dict())

def play(seed, opponent='random'):
    core = PenguinChessCore(seed=seed).reset(seed=seed)
    rgame = RustStatefulGame(engine, seed)
    for _ in range(6):
        leg = core.get_legal_actions()
        if leg: a = random.choice(leg); core.step(a); rgame.step(a)
    while not core._terminated and core._episode_steps < 200:
        legal = core.get_legal_actions()
        if not legal: break
        if core.current_player == 0:
            ac = native.search(rgame.handle, num_simulations=100, c_puct=1.4)
            action = max((int(k) for k in ac.keys()), default=random.choice(legal)) if ac else random.choice(legal)
        elif opponent == 'gen2':
            r = ab.search(json.dumps({'board':{'cells':[{'coord':{'q':x.q,'r':x.r,'s':x.s},'state':x.state,'points':x.points if x.state=='active' else 0} for x in core.hexes],'hex_map':{},'neighbors':[]},'pieces':[{'id':p.id,'alive':p.alive,'hex_idx':core.hexes.index(p.hex) if p.alive and p.hex else None,'hex_value':p.hex_value} for p in core.pieces],'scores':core.players_scores,'phase':core.phase,'current_player':core.current_player,'placement_count':core._placement_count,'episode_steps':core._episode_steps,'terminated':False,'last_action':None}), max_depth=4)
            action = r.get('best_action', 0)
        else:
            action = random.choice(legal)
        if action in legal: core.step(action); rgame.step(action)
        elif legal: core.step(legal[0]); rgame.step(legal[0])
    s1,s2 = core.players_scores
    return 1 if s1>s2 else (-1 if s2>s1 else 0)

for opp_name, opp_type in [('Random', 'random'), ('gen_2', 'gen2')]:
    N=100; w=d=l=0; t0=time.time()
    print(f'\nvs {opp_name} — {N} games...', end=' ', flush=True)
    for g in range(N):
        r = play(g*9973+42+(0 if opp_type=='random' else 1), opp_type)
        if r>0: w+=1
        elif r==0: d+=1
        else: l+=1
        if (g+1)%25==0: print(f'[{g+1}]W:{w}L:{l}D:{d}', end=' ', flush=True)
    print(f'\n  Result: {w}/{N} wins ({w/N*100:.0f}%)  {d} draws  {time.time()-t0:.0f}s')

native.free(); ab.free()
