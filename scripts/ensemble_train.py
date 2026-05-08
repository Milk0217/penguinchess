"""Ensemble + mixed data AB+NNUE training (Rust engine, buggy Adam)."""
import sys, time, json, torch, numpy as np, random as rnd, struct
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import (ffi_ab_create, ffi_ab_generate_selfplay_data,
    get_engine, POINTER, c_float, c_int32, c_char_p, create_string_buffer)

OUT = Path('models/nnue_trained'); OUT.mkdir(exist_ok=True)
CFG = '{"max_depth":6,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2,"null_move":True}'
REC_BYTES = 296

def gen_data(teacher_path: str, tag: str, n_games=800, epsilon=0.0):
    """Generate self-play data. epsilon=0.3 adds 30% random."""

    m = NNUE()
    sd = torch.load(teacher_path, map_location='cpu', weights_only=False)
    sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
    m.load_state_dict(sd, strict=False)
    d = {k: v.cpu() for k, v in m.state_dict().items()}

    cfg = CFG
    if epsilon > 0:
        c = json.loads(cfg); c['epsilon'] = epsilon; cfg = json.dumps(c)

    h = ffi_ab_create(cfg); h.set_weights(d)
    path = str(OUT / f'sp_{tag}.bin')
    Path(path).unlink(missing_ok=True)
    t0 = time.time()
    cnt = ffi_ab_generate_selfplay_data(h, num_games=n_games, workers=8, output_path=path)
    print(f'  {tag}: {cnt} pos in {time.time()-t0:.0f}s', flush=True)
    del h
    return cnt

def train_from_scratch(data_paths: list[str], out_tag: str, lr=3e-4, epochs=50, bs=4096):
    """Train from scratch on combined data."""

    data = bytearray()
    total_pos = 0
    for p in data_paths:
        with open(p, 'rb') as f:
            raw = f.read()
        cnt = struct.unpack('<Q', raw[:8])[0]
        total_pos += cnt
        data.extend(raw[8:])
    merged = bytearray(struct.pack('<Q', total_pos)) + data

    data_path = str(OUT / f'mixed_{out_tag}.bin')
    with open(data_path, 'wb') as f:
        f.write(merged)
    print(f'  Mixed {total_pos} pos from {len(data_paths)} sources', flush=True)

    m = NNUE()
    w = lambda k: m.state_dict()[k].cpu().numpy()
    flat = np.concatenate([
        w('ft.weight').T.ravel(), w('ft.bias').ravel(),
        w('fc1.weight').ravel(), w('fc1.bias').ravel(),
        w('fc2.weight').ravel(), w('fc2.bias').ravel(),
        w('fc3.weight').ravel(), w('fc3.bias').ravel(),
    ]).astype(np.float32)

    eng = get_engine()
    ptr = flat.ctypes.data_as(POINTER(c_float))
    tcfg = json.dumps({'lr': lr, 'wd': 1e-4, 'batch_size': bs, 'epochs': epochs})
    t0 = time.time()
    eng._lib.ffi_nnue_train(ptr, c_int32(len(flat)),
        c_char_p(data_path.encode()), c_char_p(tcfg.encode()),
        create_string_buffer(4096), 4096)
    print(f'  Train: {time.time()-t0:.0f}s', flush=True)

    off = 0; FD, HD, HD2 = 64, 256, 128; idim = FD * 2 + 66
    m3 = NNUE()
    m3.ft.weight.data = torch.from_numpy(flat[off:off+360*FD].reshape(FD, 360)); off += 360*FD
    m3.ft.bias.data = torch.from_numpy(flat[off:off+FD]); off += FD
    m3.fc1.weight.data = torch.from_numpy(flat[off:off+HD*idim].reshape(HD, idim)); off += HD*idim
    m3.fc1.bias.data = torch.from_numpy(flat[off:off+HD]); off += HD
    m3.fc2.weight.data = torch.from_numpy(flat[off:off+HD2*HD].reshape(HD2, HD)); off += HD2*HD
    m3.fc2.bias.data = torch.from_numpy(flat[off:off+HD2]); off += HD2
    m3.fc3.weight.data = torch.from_numpy(flat[off:off+HD2].reshape(1, HD2)); off += HD2
    m3.fc3.bias.data = torch.from_numpy(flat[off:off+1])
    torch.save({'model_state': m3.state_dict()}, str(OUT / f'{out_tag}.pt'))
    print(f'  Saved {out_tag}.pt', flush=True)
    return m3

def evaluate(model, n_games=100):
    """vs Random at depth 4."""
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

if __name__ == '__main__':
    # ── Step 1: Generate 3 diverse datasets ──
    print('=== Data Generation ===', flush=True)
    gen_data('models/nnue/nnue_gen_2.pt', 'd6_gen2', n_games=800)        # gen_2 teacher
    gen_data('models/nnue_trained/gen2v2.pt', 'd6_gen2v2', n_games=800)  # gen2v2 teacher
    gen_data('models/nnue_trained/d6_best.pt', 'd6_best', n_games=800)   # d6_best teacher
    print('', flush=True)

    # ── Step 2: Train ensemble of 5 models (different seeds via data shuffle) ──
    print('=== Training Ensemble ===', flush=True)
    sources = [str(OUT / 'sp_d6_gen2.bin'), str(OUT / 'sp_d6_gen2v2.bin'),
               str(OUT / 'sp_d6_best.bin')]
    
    wrs = []
    for seed in range(5):
        tag = f'ensemble_{seed}'
        print(f'\n--- Model {seed} ---', flush=True)
        model = train_from_scratch(sources, tag, epochs=50, bs=4096)
        wr = evaluate(model, n_games=50)
        wrs.append(wr)
    
    print(f'\nEnsemble WRs: {[f"{x*100:.0f}" for x in wrs]}')
    print(f'Avg: {sum(wrs)/len(wrs)*100:.0f}%')
