"""Eval MCTS+NNUE large model vs gen_2 + Random"""
import json, random, torch, time, sys
sys.path.insert(0,'.')
from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue_mcts import NNUEMCTSModel
from penguinchess.rust_ffi import get_engine, RustStatefulGame, NNUEMCTSNative

device = 'cuda' if torch.cuda.is_available() else 'cpu'
engine = get_engine()

# Load large MCTS+NNUE model
hd, fd = 512, 128
model = NNUEMCTSModel(hidden_dim=hd, ft_dim=fd).to(device).eval()
ckpt = torch.load('models/nnue_mcts_large/nnue_mcts_best.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
model.load_state_dict(sd, strict=False)
native_sd = {k: v.cpu() for k, v in model.state_dict().items()}
native = NNUEMCTSNative(native_sd)
print(f'Large model loaded (242K params, best WR={sd.get("win_rate",0)*100:.0f}%)')

N = 100

print(f'\nvs Random — {N} games...')
t0 = time.time()
w = d = l = 0
for g in range(N):
    core = PenguinChessCore(seed=g*9973+42).reset(seed=g*9973+42)
    rgame = RustStatefulGame(engine, g*9973+42)
    for _ in range(6):
        leg = core.get_legal_actions()
        if leg: a = random.choice(leg); core.step(a); rgame.step(a)
    while not core._terminated and core._episode_steps < 200:
        legal = core.get_legal_actions()
        if not legal: break
        if core.current_player == 0:
            ac = native.search(rgame.handle, num_simulations=100, c_puct=1.4)
            action = max((int(k) for k in ac.keys()), default=random.choice(legal)) if ac else random.choice(legal)
        else:
            action = random.choice(legal)
        if action in legal: core.step(action); rgame.step(action)
        elif legal: core.step(legal[0]); rgame.step(legal[0])
    s1,s2 = core.players_scores
    if s1>s2: w+=1
    elif s1==s2: d+=1
    else: l+=1
    if (g+1)%25==0: print(f'  [{g+1}/{N}] W:{w} L:{l} D:{d}')
print(f'\nvs Random: {w}/{N} wins ({w/N*100:.0f}%)  {d} draws  {l} losses  ({time.time()-t0:.0f}s)')

native.free()
