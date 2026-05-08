"""AB+NNUE: Rust data gen + PyTorch fine-tune (correct Adam)."""
import sys, time, json, torch, numpy as np, random as rnd, struct, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import (ffi_ab_create, ffi_ab_generate_selfplay_data)
import torch.nn as nn
import torch.optim as optim

OUT = Path('models/nnue_trained'); OUT.mkdir(exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
REC_BYTES = 296

# ─── Data ────────────────────────────────────────────────

def load_binary(path):
    """Load binary training records: 6×i32 sparse + 66×f32 dense + 1×f32 label + 1×i32 stm."""
    raw = open(path, 'rb').read()
    n = struct.unpack('<Q', raw[:8])[0]
    records = []
    off = 8
    for _ in range(n):
        sparse = [struct.unpack('<i', raw[off+i*4:off+(i+1)*4])[0] for i in range(6)]
        off += 24
        dense = [struct.unpack('<f', raw[off+i*4:off+(i+1)*4])[0] for i in range(66)]
        off += 264
        label = struct.unpack('<f', raw[off:off+4])[0]; off += 4
        stm = struct.unpack('<i', raw[off:off+4])[0]; off += 4
        records.append((sparse, dense, label, stm))
    return records

def gen_data(teacher_path, tag, n_games=800):
    """Rust data gen at depth 6."""
    m = NNUE()
    sd = torch.load(teacher_path, map_location='cpu', weights_only=False)
    sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
    m.load_state_dict(sd, strict=False)
    d = {k: v.cpu() for k, v in m.state_dict().items()}
    h = ffi_ab_create(json.dumps({"max_depth":6,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2,"null_move":True}))
    h.set_weights(d)
    path = str(OUT / f'sp_{tag}.bin')
    Path(path).unlink(missing_ok=True)
    t0 = time.time()
    cnt = ffi_ab_generate_selfplay_data(h, num_games=n_games, workers=8, output_path=path)
    print(f'  {tag}: {cnt} pos in {time.time()-t0:.0f}s', flush=True)
    del h
    return path

# ─── PyTorch NNUE forward ────────────────────────────────

P1_CUTOFF = 180

def nnue_forward(model, records, device):
    """Vectorized NNUE forward. Returns predictions."""
    B = len(records)
    stm_feat = torch.zeros(B, 360, device=device)
    nstm_feat = torch.zeros(B, 360, device=device)
    dense = torch.zeros(B, 66, device=device)
    
    for i, (sp, de, _, stm) in enumerate(records):
        for f in sp:
            if f < 0 or f >= 360: continue
            is_p1 = f < P1_CUTOFF
            is_stm = (stm == 0 and is_p1) or (stm == 1 and not is_p1)
            if is_stm: stm_feat[i, f] = 1.0
            else: nstm_feat[i, f] = 1.0
        for j in range(66):
            dense[i, j] = de[j]
    
    with torch.no_grad():
        # FT: (B, 360) @ (360, FD) + bias → (B, FD)
        ft_w = model.ft.weight.T  # (360, FD)
        ft_b = model.ft.bias
        stm_acc = stm_feat @ ft_w + ft_b  # (B, FD)
        nstm_acc = nstm_feat @ ft_w + ft_b
        
        # CReLU
        stm_cr = stm_acc.clamp(0, 127) * 2.0 / 127.0 - 1.0
        nstm_cr = nstm_acc.clamp(0, 127) * 2.0 / 127.0 - 1.0
        
        x = torch.cat([stm_cr, nstm_cr, dense], dim=1)  # (B, 194)
        
        # FC layers
        h1 = torch.relu(model.fc1(x))  # (B, 256)
        h2 = torch.relu(model.fc2(h1))  # (B, 128)
        out = torch.tanh(model.fc3(h2)).squeeze(1)  # (B,)
    return out

# ─── Training ────────────────────────────────────────────

def ft_epoch(model, optimizer, loader):
    """Fine-tune epoch. FT weights frozen, FC layers trained."""
    model.train()
    total = 0.0; n = 0
    for sp_batch, de_batch, la_batch, stm_batch in loader:
        B = len(sp_batch)
        stm_f = torch.zeros(B, 360, device=DEVICE)
        nstm_f = torch.zeros(B, 360, device=DEVICE)
        de_t = de_batch.to(DEVICE)
        la_t = la_batch.to(DEVICE).float()
        
        for i in range(B):
            for f in sp_batch[i]:
                if f < 0 or f >= 360: continue
                is_p1 = f < P1_CUTOFF
                is_stm = (stm_batch[i] == 0 and is_p1) or (stm_batch[i] == 1 and not is_p1)
                if is_stm: stm_f[i, f] = 1.0
                else: nstm_f[i, f] = 1.0
        
        ft_w = model.ft.weight.T
        ft_b = model.ft.bias
        with torch.no_grad():
            stm_a = stm_f @ ft_w + ft_b
            nstm_a = nstm_f @ ft_w + ft_b
            stm_cr = stm_a.clamp(0, 127) * 2.0 / 127.0 - 1.0
            nstm_cr = nstm_a.clamp(0, 127) * 2.0 / 127.0 - 1.0
            x = torch.cat([stm_cr, nstm_cr, de_t], dim=1).detach()
        
        x.requires_grad = True
        h1 = torch.relu(model.fc1(x))
        h2 = torch.relu(model.fc2(h1))
        out = torch.tanh(model.fc3(h2)).squeeze(1)
        loss = nn.MSELoss()(out, la_t)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.fc1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.fc2.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.fc3.parameters(), 1.0)
        optimizer.step()
        
        total += loss.item() * B; n += B
    return total / n

def train_ft(data_path, teacher_path, out_path, epochs=100, lr=1e-5, bs=4096):
    """Fine-tune gen_2 on new data."""
    records = load_binary(data_path)
    n = len(records)
    split = int(n * 0.9)
    train_recs = records[:split]
    val_recs = records[split:]
    print(f'  Records: {len(train_recs)} train, {len(val_recs)} val', flush=True)
    
    model = NNUE()
    sd = torch.load(teacher_path, map_location='cpu', weights_only=False)
    sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
    model.load_state_dict(sd, strict=False)
    model = model.to(DEVICE).float().train()
    
    # Freeze FT
    for p in model.ft.parameters():
        p.requires_grad = False
    
    optimizer = optim.Adam(itertools.chain(model.fc1.parameters(), model.fc2.parameters(), model.fc3.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    
    collate = lambda batch: (
        [r[0] for r in batch],
        torch.tensor([r[1] for r in batch]),
        torch.tensor([r[2] for r in batch]),
        [r[3] for r in batch],
    )
    train_loader = torch.utils.data.DataLoader(train_recs, bs, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_recs, bs, shuffle=False, collate_fn=collate)
    
    best_val = float('inf')
    best_state = model.state_dict().copy()
    
    for ep in range(epochs):
        tr_loss = ft_epoch(model, optimizer, train_loader)
        
        model.eval()
        val_loss = 0.0; vn = 0
        with torch.no_grad():
            for sp_b, de_b, la_b, stm_b in val_loader:
                pred = nnue_forward(model, list(zip(sp_b, de_b.cpu(), la_b.cpu(), stm_b)), DEVICE)
                val_loss += nn.MSELoss()(pred, la_b.to(DEVICE).float()).item() * len(sp_b)
                vn += len(sp_b)
        val_loss /= vn
        
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict().copy()
        
        if ep % 10 == 0 or ep == epochs - 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'  ep {ep+1:>3d}/{epochs}  train={tr_loss:.4f}  val={val_loss:.4f}  LR={lr_now:.1e}', flush=True)
    
    model.load_state_dict(best_state)
    torch.save({'model_state': model.state_dict()}, out_path)
    print(f'  Saved {out_path} (val_loss={best_val:.4f})', flush=True)
    return model

# ─── Evaluation ──────────────────────────────────────────

def evaluate(model, n_games=100):
    """vs Random at depth 4 using Rust AB search."""
    from penguinchess.rust_ffi import ffi_ab_create
    h = ffi_ab_create('{"max_depth":4,"tt_size":65536}')
    h.set_weights({k: v.cpu() for k, v in model.state_dict().items()})
    sm = {'active': 'active', 'occupied': 'occupied', 'used': 'used', 'eliminated': 'eliminated'}
    w = d = l = 0
    for g in range(n_games):
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
    del h
    wr = w / n_games
    print(f'  vs Random: {w}/{n_games} ({w}%) {d} draws', flush=True)
    return wr

# ─── Main Pipeline ───────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--games', type=int, default=800)
    parser.add_argument('--teacher', type=str, default='models/nnue/nnue_gen_2.pt')
    args = parser.parse_args()
    
    print(f'Device: {DEVICE}', flush=True)
    
    # Step 1: Generate data
    print('=== Data Generation (Rust) ===', flush=True)
    data_path = gen_data(args.teacher, 'pt_ft', n_games=args.games)
    
    # Step 2: Fine-tune (PyTorch)
    print('\n=== Fine-tune (PyTorch) ===', flush=True)
    model = train_ft(data_path, args.teacher,
                     str(OUT / 'pt_finetuned.pt'),
                     epochs=args.epochs, lr=args.lr)
    
    # Step 3: Evaluate
    print('\n=== Evaluation ===', flush=True)
    evaluate(model, n_games=100)
