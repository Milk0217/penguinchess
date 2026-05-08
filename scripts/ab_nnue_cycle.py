"""Rust AB+NNUE training: depth 6 self-play + buggy Adam."""
import sys, time, json, torch, numpy as np, random as rnd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import (
    ffi_ab_create, ffi_ab_generate_selfplay_data, get_engine,
    POINTER, c_float, c_int32, c_char_p, create_string_buffer)

N_GAMES = 2000
DEPTH = 6
EPOCHS = 50
LR = 3e-4
BATCH_SIZE = 4096
WORKERS = 8
OUT_DIR = Path('models/nnue_trained')

def depth6_training_cycle(teacher_path: str = 'models/nnue/nnue_gen_2.pt',
                           tag: str = 'gen1') -> float:
    """One training cycle: gen data with teacher → train from scratch → eval."""
    OUT_DIR.mkdir(exist_ok=True)
    
    # Load teacher weights
    m = NNUE()
    sd = torch.load(teacher_path, map_location='cpu', weights_only=False)
    sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
    m.load_state_dict(sd, strict=False)
    d = {k: v.cpu() for k, v in m.state_dict().items()}
    print(f'Teacher loaded ({tag})', flush=True)
    
    # Generate self-play data at depth 6
    cfg = json.dumps({'max_depth': DEPTH, 'tt_size': 65536,
                       'lmr_moves': 3, 'lmr_depth': 1,
                       'nnue_order_depth': 2, 'null_move': True})
    handle = ffi_ab_create(cfg)
    handle.set_weights(d)
    
    data_path = str(OUT_DIR / f'sp_{tag}.bin')
    Path(data_path).unlink(missing_ok=True)
    
    t0 = time.time()
    cnt = ffi_ab_generate_selfplay_data(handle, num_games=N_GAMES,
                                          workers=WORKERS, output_path=data_path)
    gen_s = time.time() - t0
    print(f'  {cnt} pos in {gen_s:.0f}s ({gen_s/N_GAMES:.1f}s/game)', flush=True)
    del handle
    
    # Train from scratch using buggy Rust Adam
    m2 = NNUE()
    w = lambda k: m2.state_dict()[k].cpu().numpy()
    flat = np.concatenate([
        w('ft.weight').T.ravel(), w('ft.bias').ravel(),
        w('fc1.weight').ravel(), w('fc1.bias').ravel(),
        w('fc2.weight').ravel(), w('fc2.bias').ravel(),
        w('fc3.weight').ravel(), w('fc3.bias').ravel(),
    ]).astype(np.float32)
    
    eng = get_engine()
    ptr = flat.ctypes.data_as(POINTER(c_float))
    tcfg = json.dumps({'lr': LR, 'wd': 1e-4, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS})
    t0 = time.time()
    eng._lib.ffi_nnue_train(ptr, c_int32(len(flat)),
        c_char_p(data_path.encode()), c_char_p(tcfg.encode()),
        create_string_buffer(4096), 4096)
    print(f'  Train: {time.time()-t0:.0f}s', flush=True)
    
    # Save model
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
    
    model_path = str(OUT_DIR / f'd6_{tag}.pt')
    torch.save({'model_state': m3.state_dict()}, model_path)
    print(f'  Saved {model_path}', flush=True)
    
    # Evaluate vs Random (depth 4, 50 games for quick feedback)
    h_eval = ffi_ab_create(json.dumps({'max_depth': 4, 'tt_size': 65536}))
    h_eval.set_weights({k: v.cpu() for k, v in m3.state_dict().items()})
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
                ac = h_eval.search(sj, max_depth=4).get('best_action', 0)
            else:
                ac = rnd.choice(leg)
            if ac in leg: c.step(ac)
            elif leg: c.step(leg[0])
        s1, s2 = c.players_scores
        if s1 > s2: w += 1
        elif s1 == s2: d += 1
        else: l += 1
    wr = w / 100
    print(f'  vs Random: {w}/100 ({w}%)  {d} draws', flush=True)
    
    # Update saved model with win rate
    torch.save({'model_state': m3.state_dict(), 'win_rate': wr}, model_path)
    return wr


if __name__ == '__main__':
    # Generation 1: from gen_2
    wr1 = depth6_training_cycle(
        teacher_path='models/nnue/nnue_gen_2.pt', tag='gen1')
    print(f'\nGen1 WR: {wr1*100:.0f}%', flush=True)
    
    # Generation 2: from gen1's model
    wr2 = depth6_training_cycle(
        teacher_path=str(OUT_DIR / 'd6_gen1.pt'), tag='gen2')
    print(f'\nGen2 WR: {wr2*100:.0f}%', flush=True)
