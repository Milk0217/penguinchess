"""Evaluate AB+NNUE vs PPO."""
import sys, json, torch, numpy as np, random as rnd
sys.path.insert(0, '.')
from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import ffi_ab_create
from stable_baselines3 import PPO
import gymnasium as gym
import penguinchess.env

env = gym.make('PenguinChess-v0')
ppo = PPO.load('models/ppo_penguinchess_gen_10.zip', device='cpu')

m = NNUE()
try:
    sd = torch.load('models/nnue_trained/d6_best.pt', map_location='cpu', weights_only=False)
except:
    sd = torch.load('models/nnue_trained/scratch.pt', map_location='cpu', weights_only=False)
sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
m.load_state_dict(sd, strict=False)

cfg = json.dumps({"max_depth":6,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2,"null_move":True})
h = ffi_ab_create(cfg)
h.set_weights({k: v.cpu() for k, v in m.state_dict().items()})

sm = {'active': 'active'}

def ab_move(core):
    hx = [{'coord': {'q': x.q, 'r': x.r, 's': x.s}, 'state': sm.get(x.state, 'active'),
           'points': x.points if x.state == 'active' else 0} for x in core.hexes]
    px = [{'id': p.id, 'alive': p.alive,
           'hex_idx': core.hexes.index(p.hex) if p.alive and p.hex else None,
           'hex_value': p.hex_value} for p in core.pieces]
    r = h.search(json.dumps({'board': {'cells': hx}, 'pieces': px,
                              'scores': core.players_scores, 'phase': core.phase,
                              'current_player': core.current_player,
                              'placement_count': core._placement_count,
                              'episode_steps': core._episode_steps, 'terminated': False}),
                 max_depth=6)
    return r.get('best_action', 0)

def get_obs(core):
    flat = []
    for cell in core.hexes:
        val = cell.points if cell.state == 'active' else 0
        flat.extend([cell.q, cell.r, cell.s, val])
    for p in core.pieces:
        if p.alive:
            flat.extend([p.id, core.hexes.index(p.hex) if p.hex else -1, p.hex_value])
        else:
            flat.extend([-1, -1, 0])
    flat.extend([core.current_player, 1 if core.phase == 'movement' else 0] + list(core.players_scores))
    return np.array(flat[:206], dtype=np.float32)

aw = al = ad = 0; pw = pl = pd = 0; n = 50
for g in range(n):
    env.reset(seed=g*9973+42)
    core = env.unwrapped.core
    for _ in range(6):
        l = core.get_legal_actions()
        if l: core.step(rnd.choice(l))
    while not core._terminated and core._episode_steps < 200:
        l = core.get_legal_actions()
        if not l: break
        if core.current_player == 0:
            a = ab_move(core)
        else:
            a, _ = ppo.predict(get_obs(core), deterministic=True)
            a = int(a.item()) if isinstance(a, np.ndarray) else int(a)
        if a in l: core.step(a)
        elif l: core.step(l[0])
    s1, s2 = core.players_scores
    if s1 > s2: aw += 1; pl += 1
    elif s1 == s2: ad += 1; pd += 1
    else: al += 1; pw += 1
    print(f'  g{g+1}: AB+NNUE={aw} PPO={pw}', flush=True)

print(f'\nAB+NNUE (depth 6) vs PPO gen10:', flush=True)
print(f'  AB+NNUE wins: {aw}/{n} ({aw/n*100:.0f}%)  Draws: {ad}', flush=True)
print(f'  PPO wins:     {pw}/{n} ({pw/n*100:.0f}%)', flush=True)
