"""PyTorch scratch training: correct Adam, all params unfrozen, CUDA."""
import sys, time, json, torch, numpy as np, random as rnd, struct
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE
import torch.nn as nn, torch.optim as optim

DEVICE = 'cuda'
REC_BYTES = 296; P1_CUTOFF = 180

def load_bin(p):
    raw = open(p, 'rb').read(); n = struct.unpack('<Q', raw[:8])[0]; r = []; off = 8
    for _ in range(n):
        sp = [struct.unpack('<i', raw[off+i*4:off+(i+1)*4])[0] for i in range(6)]; off += 24
        de = [struct.unpack('<f', raw[off+i*4:off+(i+1)*4])[0] for i in range(66)]; off += 264
        la = struct.unpack('<f', raw[off:off+4])[0]; off += 4
        st = struct.unpack('<i', raw[off:off+4])[0]; off += 4
        r.append((sp, de, la, st))
    return r

recs = load_bin('models/nnue_trained/sp_pt_ft.bin')
n = len(recs); split = int(n * 0.9)
train_recs = recs[:split]; val_recs = recs[split:]
print(f'{len(train_recs)} train, {len(val_recs)} val', flush=True)

model = NNUE().to(DEVICE).float().train()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

best_val = float('inf'); best_sd = model.state_dict().copy()
for ep in range(200):
    rnd.shuffle(train_recs)
    tr = 0.0; tcnt = 0
    for i in range(0, len(train_recs), 4096):
        batch = train_recs[i:i+4096]; B = len(batch)
        sf = torch.zeros(B, 360, device=DEVICE); nf = torch.zeros(B, 360, device=DEVICE)
        dt = torch.zeros(B, 66, device=DEVICE); lb = torch.zeros(B, device=DEVICE)
        for j, (sp, de, la, st) in enumerate(batch):
            for f in sp:
                if f < 0 or f >= 360: continue
                p1 = f < P1_CUTOFF
                stm = (st == 0 and p1) or (st == 1 and not p1)
                if stm: sf[j, f] = 1.0
                else: nf[j, f] = 1.0
            dt[j] = torch.tensor(de); lb[j] = la
        x = torch.cat([
            (sf @ model.ft.weight.T + model.ft.bias).clamp(0, 127) * 2 / 127 - 1,
            (nf @ model.ft.weight.T + model.ft.bias).clamp(0, 127) * 2 / 127 - 1,
            dt
        ], dim=1)
        h1 = torch.relu(model.fc1(x))
        h2 = torch.relu(model.fc2(h1))
        out = torch.tanh(model.fc3(h2)).squeeze(1)
        loss = nn.MSELoss()(out, lb)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        tr += loss.item(); tcnt += 1
    
    model.eval(); vl = 0.0; vcnt = 0
    with torch.no_grad():
        for i in range(0, len(val_recs), 4096):
            batch = val_recs[i:i+4096]; B = len(batch)
            sf = torch.zeros(B, 360, device=DEVICE); nf = torch.zeros(B, 360, device=DEVICE)
            dt = torch.zeros(B, 66, device=DEVICE); lb = torch.zeros(B, device=DEVICE)
            for j, (sp, de, la, st) in enumerate(batch):
                for f in sp:
                    if f < 0 or f >= 360: continue
                    p1 = f < P1_CUTOFF
                    stm = (st == 0 and p1) or (st == 1 and not p1)
                    if stm: sf[j, f] = 1.0
                    else: nf[j, f] = 1.0
                dt[j] = torch.tensor(de); lb[j] = la
            x = torch.cat([
                (sf @ model.ft.weight.T + model.ft.bias).clamp(0, 127) * 2 / 127 - 1,
                (nf @ model.ft.weight.T + model.ft.bias).clamp(0, 127) * 2 / 127 - 1,
                dt
            ], dim=1)
            h1 = torch.relu(model.fc1(x))
            h2 = torch.relu(model.fc2(h1))
            out = torch.tanh(model.fc3(h2)).squeeze(1)
            vl += nn.MSELoss()(out, lb).item() * B; vcnt += B
    vl /= vcnt
    if vl < best_val: best_val = vl; best_sd = model.state_dict().copy()
    if ep % 10 == 0: print(f'ep {ep+1:>3d}/200  train={tr/tcnt:.4f}  val={vl:.4f}', flush=True)

model.load_state_dict(best_sd)
torch.save({'model_state': model.state_dict(), 'val_loss': best_val}, 'models/nnue_trained/pt_scratch.pt')
print(f'Best val={best_val:.4f}', flush=True)

# Eval
from penguinchess.rust_ffi import ffi_ab_create
h = ffi_ab_create(json.dumps({'max_depth': 4, 'tt_size': 65536}))
h.set_weights({k: v.cpu() for k, v in model.state_dict().items()})
sm = {'active': 'active'}; w = d = l = 0
for g in range(100):
    c = PenguinChessCore(seed=g*9973+42).reset(seed=g*9973+42)
    for _ in range(6):
        leg = c.get_legal_actions()
        if leg: c.step(rnd.choice(leg))
    while not c._terminated and c._episode_steps < 200:
        leg = c.get_legal_actions()
        if not leg: break
        if c.current_player == 0:
            hx = [{'coord': {'q': x.q, 'r': x.r, 's': x.s},
                   'state': sm.get(x.state, 'active'),
                   'points': x.points if x.state == 'active' else 0} for x in c.hexes]
            px = [{'id': p.id, 'alive': p.alive,
                   'hex_idx': c.hexes.index(p.hex) if p.alive and p.hex else None,
                   'hex_value': p.hex_value} for p in c.pieces]
            sj = json.dumps({'board': {'cells': hx}, 'pieces': px,
                             'scores': c.players_scores, 'phase': 'movement',
                             'current_player': c.current_player,
                             'placement_count': c._placement_count,
                             'episode_steps': c._episode_steps, 'terminated': False})
            ac = h.search(sj, max_depth=4).get('best_action', 0)
        else:
            ac = rnd.choice(leg)
        if ac in leg: c.step(ac)
        elif leg: c.step(leg[0])
    s1, s2 = c.players_scores
    if s1 > s2: w += 1
    elif s1 == s2: d += 1
    else: l += 1
print(f'PyTorch scratch vs Random (CUDA): {w}/100 ({w}%)', flush=True)
