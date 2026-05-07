"""Full AB+NNUE training cycle: load gen_2 → Rust self-play data gen → Rust train → save + eval."""
import sys, time, json, numpy as np, torch, ctypes
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import (
    get_engine, ffi_ab_create, ffi_ab_generate_selfplay_data,
    POINTER, c_float, c_int32, c_char_p, create_string_buffer)

N_GAMES = 500
DEPTH = 4  # AB search depth for labels
EPOCHS = 30
BATCH_SIZE = 4096
LR = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load gen_2 weights
print('Loading gen_2...')
model = NNUE()
sd = torch.load('models/nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
model.load_state_dict(sd, strict=False)
print(f'  gen_2 loaded')

# 2. Generate self-play data with Rust (AB search for move selection + labels)
cfg = json.dumps({"max_depth": DEPTH, "tt_size": 65536,
                   "lmr_moves": 3, "lmr_depth": 1, "nnue_order_depth": 2})
handle = ffi_ab_create(cfg)
handle.set_weights({k: v.cpu() for k, v in model.state_dict().items()})

out_dir = Path('models/nnue_trained')
out_dir.mkdir(exist_ok=True)
data_path = str(out_dir / f'selfplay_d{DEPTH}.bin')
Path(data_path).unlink(missing_ok=True)

print(f'Generating {N_GAMES} self-play games at depth {DEPTH}...')
t0 = time.time()
count = ffi_ab_generate_selfplay_data(handle, num_games=N_GAMES, workers=4, output_path=data_path)
gen_s = time.time() - t0
print(f'  {count} positions in {gen_s:.0f}s ({gen_s/N_GAMES:.2f}s/game)')

# 3. Train with Rust
engine = get_engine()
ft_w = sd.get('ft.weight', model.state_dict()['ft.weight']).cpu().numpy().T.ravel()
ft_b = sd.get('ft.bias', model.state_dict()['ft.bias']).cpu().numpy().ravel()
fc1_w = sd.get('fc1.weight', model.state_dict()['fc1.weight']).cpu().numpy().ravel()
fc1_b = sd.get('fc1.bias', model.state_dict()['fc1.bias']).cpu().numpy().ravel()
fc2_w = sd.get('fc2.weight', model.state_dict()['fc2.weight']).cpu().numpy().ravel()
fc2_b = sd.get('fc2.bias', model.state_dict()['fc2.bias']).cpu().numpy().ravel()
fc3_w = sd.get('fc3.weight', model.state_dict()['fc3.weight']).cpu().numpy().ravel()
fc3_b = sd.get('fc3.bias', model.state_dict()['fc3.bias']).cpu().numpy().ravel()
flat = np.concatenate([ft_w, ft_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b]).astype(np.float32)

ptr = flat.ctypes.data_as(POINTER(c_float))
tcfg = json.dumps({"lr": LR, "wd": 1e-4, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "max_norm": 1.0})
out = create_string_buffer(4096)
print(f'\nTraining {count} positions x {EPOCHS} epochs (Rust, bs={BATCH_SIZE})...')
t0 = time.time()
rc = engine._lib.ffi_nnue_train(
    ptr, c_int32(len(flat)),
    c_char_p(data_path.encode('utf-8')),
    c_char_p(tcfg.encode('utf-8')),
    out, c_int32(4096))
train_s = time.time() - t0
result = json.loads(out.value.decode('utf-8')) if out.value else {}
print(f'  {result}  ({train_s:.0f}s, rc={rc})')

# 4. Save model
off = 0; ft_dim = 64; fc1_dim = 256; fc2_dim = 128; idim = ft_dim*2+66
ft_w2 = flat[off:off+360*ft_dim].reshape(360, ft_dim).T; off += 360*ft_dim
ft_b2 = flat[off:off+ft_dim]; off += ft_dim
fc1_w2 = flat[off:off+fc1_dim*idim].reshape(fc1_dim, idim); off += fc1_dim*idim
fc1_b2 = flat[off:off+fc1_dim]; off += fc1_dim
fc2_w2 = flat[off:off+fc2_dim*fc1_dim].reshape(fc2_dim, fc1_dim); off += fc2_dim*fc1_dim
fc2_b2 = flat[off:off+fc2_dim]; off += fc2_dim
fc3_w2 = flat[off:off+fc2_dim].reshape(1, fc2_dim)
fc3_b2 = flat[off:off+1]

m = NNUE()
m.ft.weight.data = torch.from_numpy(ft_w2)
m.ft.bias.data = torch.from_numpy(ft_b2)
m.fc1.weight.data = torch.from_numpy(fc1_w2)
m.fc1.bias.data = torch.from_numpy(fc1_b2)
m.fc2.weight.data = torch.from_numpy(fc2_w2)
m.fc2.bias.data = torch.from_numpy(fc2_b2)
m.fc3.weight.data = torch.from_numpy(fc3_w2)
m.fc3.bias.data = torch.from_numpy(fc3_b2)
out_path = str(out_dir / 'nnue_improved.pt')
torch.save({'model_state': m.state_dict(), 'val_loss': result.get('best_loss', 0)}, out_path)
print(f'Saved to {out_path}')

# 5. Quick eval vs Random (depth 2, 50 games)
print(f'\nEvaluating vs Random (50 games, AB depth 2)...')
from penguinchess.core import PenguinChessCore
import random as _rnd
handle2 = ffi_ab_create(json.dumps({"max_depth": 2, "tt_size": 65536}))
handle2.set_weights({k: v.cpu() for k, v in m.state_dict().items()})

w = d = l = 0
eval_smap = {'active': 'active', 'occupied': 'occupied', 'used': 'used', 'eliminated': 'eliminated'}
for g in range(50):
    core = PenguinChessCore(seed=g*9973+42).reset(seed=g*9973+42)
    for _ in range(6):
        leg = core.get_legal_actions()
        if leg: core.step(_rnd.choice(leg))
    while not core._terminated and core._episode_steps < 200:
        legal = core.get_legal_actions()
        if not legal: break
        if core.current_player == 0:
            hx = [{'coord': {'q': x.q, 'r': x.r, 's': x.s}, 'state': eval_smap.get(x.state, 'active'),
                   'points': x.points if x.state == 'active' else 0} for x in core.hexes]
            px = [{'id': p.id, 'alive': p.alive,
                   'hex_idx': core.hexes.index(p.hex) if p.alive and p.hex else None,
                   'hex_value': p.hex_value} for p in core.pieces]
            sj = json.dumps({'board': {'cells': hx, 'hex_map': {}, 'neighbors': []}, 'pieces': px,
                             'scores': core.players_scores, 'phase': 'movement',
                             'current_player': core.current_player, 'placement_count': core._placement_count,
                             'episode_steps': core._episode_steps, 'terminated': False, 'last_action': None})
            r = handle2.search(sj, max_depth=2)
            action = r.get('best_action', 0)
        else:
            action = _rnd.choice(legal)
        if action in legal: core.step(action)
        elif legal: core.step(legal[0])
    s1, s2 = core.players_scores
    if s1 > s2: w += 1
    elif s1 == s2: d += 1
    else: l += 1
    if (g+1) % 25 == 0: print(f'  [{g+1}/50] W:{w} L:{l} D:{d}')
print(f'\nvs Random: {w}/{50} wins ({w/50*100:.0f}%)  {d} draws')
print(f'Total time: {time.time()-t0:.0f}s')
