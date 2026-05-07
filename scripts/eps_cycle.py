"""Epsilon-greedy self-play data gen + Rust training."""
import sys, time, json, torch, numpy as np, random as rnd
sys.path.insert(0, '.')
from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import ffi_ab_create, ffi_ab_generate_selfplay_data, get_engine
from penguinchess.rust_ffi import POINTER, c_float, c_int32, c_char_p, create_string_buffer

# Use gen_2 weights with epsilon=0.2 for diverse data
m = NNUE()
sd = torch.load('models/nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
m.load_state_dict(sd, strict=False)
d = {k: v.cpu() for k, v in m.state_dict().items()}
h = ffi_ab_create(json.dumps({'max_depth': 4, 'tt_size': 65536, 'lmr_moves': 3,
                               'lmr_depth': 1, 'nnue_order_depth': 2, 'null_move': True,
                               'epsilon': 0.2}))
h.set_weights(d)
print('2000 games, epsilon=0.2, depth 4, 8 workers...', flush=True)
t0 = time.time()
cnt = ffi_ab_generate_selfplay_data(h, num_games=2000, workers=8, output_path='models/nnue_trained/sp_eps.bin')
print(f'  {cnt} pos in {time.time()-t0:.0f}s', flush=True)
del h

# Train from scratch
m2 = NNUE()
w = lambda k: m2.state_dict()[k].cpu().numpy()
flat = np.concatenate([w('ft.weight').T.ravel(), w('ft.bias').ravel(),
                       w('fc1.weight').ravel(), w('fc1.bias').ravel(),
                       w('fc2.weight').ravel(), w('fc2.bias').ravel(),
                       w('fc3.weight').ravel(), w('fc3.bias').ravel()]).astype(np.float32)
eng = get_engine()
ptr = flat.ctypes.data_as(POINTER(c_float))
t0 = time.time()
eng._lib.ffi_nnue_train(ptr, c_int32(len(flat)),
    c_char_p(b'models/nnue_trained/sp_eps.bin'),
    c_char_p(json.dumps({'lr': 3e-4, 'wd': 1e-4, 'batch_size': 4096, 'epochs': 50}).encode()),
    create_string_buffer(4096), 4096)
print(f'Train: {time.time()-t0:.0f}s', flush=True)

# Save
off = 0; FD, HD, HD2 = 64, 256, 128; idim = FD*2+66
m3 = NNUE()
m3.ft.weight.data = torch.from_numpy(flat[off:off+360*FD].reshape(FD, 360)); off += 360*FD
m3.ft.bias.data = torch.from_numpy(flat[off:off+FD]); off += FD
m3.fc1.weight.data = torch.from_numpy(flat[off:off+HD*idim].reshape(HD, idim)); off += HD*idim
m3.fc1.bias.data = torch.from_numpy(flat[off:off+HD]); off += HD
m3.fc2.weight.data = torch.from_numpy(flat[off:off+HD2*HD].reshape(HD2, HD)); off += HD2*HD
m3.fc2.bias.data = torch.from_numpy(flat[off:off+HD2]); off += HD2
m3.fc3.weight.data = torch.from_numpy(flat[off:off+HD2].reshape(1, HD2)); off += HD2
m3.fc3.bias.data = torch.from_numpy(flat[off:off+1])
torch.save({'model_state': m3.state_dict()}, 'models/nnue_trained/eps.pt')
print('Saved!', flush=True)

# Eval
h2 = ffi_ab_create(json.dumps({'max_depth': 4, 'tt_size': 65536}))
h2.set_weights({k: v.cpu() for k, v in m3.state_dict().items()})
sm = {'active': 'active', 'occupied': 'occupied', 'used': 'used', 'eliminated': 'eliminated'}
w = d = l = 0
for g in range(100):
    c = PenguinChessCore(seed=g*9973+42).reset(seed=g*9973+42)
    for _ in range(6):
        leg = c.get_legal_actions()
        if leg: c.step(rnd.choice(leg))
    while not c._terminated and c._episode_steps < 200:
        leg = c.get_legal_actions()
        if not leg: break
        if c.current_player == 0:
            hx = [{'coord': {'q': x.q, 'r': x.r, 's': x.s}, 'state': sm.get(x.state, 'active'),
                   'points': x.points if x.state == 'active' else 0} for x in c.hexes]
            px = [{'id': p.id, 'alive': p.alive,
                   'hex_idx': c.hexes.index(p.hex) if p.alive and p.hex else None,
                   'hex_value': p.hex_value} for p in c.pieces]
            sj = json.dumps({'board': {'cells': hx}, 'pieces': px, 'scores': c.players_scores,
                             'phase': 'movement', 'current_player': c.current_player,
                             'placement_count': c._placement_count,
                             'episode_steps': c._episode_steps, 'terminated': False})
            ac = h2.search(sj, max_depth=4).get('best_action', 0)
        else:
            ac = rnd.choice(leg)
        if ac in leg: c.step(ac)
        elif leg: c.step(leg[0])
    s1, s2 = c.players_scores
    if s1 > s2: w += 1
    elif s1 == s2: d += 1
    else: l += 1
print(f'Eps vs Random (AB d4): {w}/100 ({w}%)')
