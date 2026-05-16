"""
Multi-gen AB+NNUE training with mixed data sources.
Generates gen_3, gen_4, ... each mixing:
  - Self-play data (depth 6, from previous gen as teacher)
  - Random game data (diverse positions, AB-scored)  
  - AZ MCTS data (if available, strong value labels)

Usage:
    uv run python ab_nnue/train.py --generations 3
    uv run python ab_nnue/train.py --resume models/nnue_trained/d6_best.pt
"""
import sys, time, json, math, shutil
import numpy as np
import torch
from pathlib import Path
from functools import partial
sys.path.insert(0, str(Path(__file__).parent.parent))

from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import (
    ffi_ab_create, ffi_ab_generate_selfplay_data, ffi_ab_generate_random_data,
    get_engine, POINTER, c_float, c_int32, c_char_p, create_string_buffer)

N_GAMES = 3000
N_RANDOM = 1000
DEPTH = 6
EPOCHS = 50
LR = 3e-4
BATCH_SIZE = 4096
WORKERS = 8
OUT_DIR = Path('models/nnue_trained')

# Rust AB+NNUE constants (compile-time, must match nnue_rs.rs)
R_FT_DIM = 128
R_HD_DIM = 512   # fc1 output
R_HD2_DIM = 256   # fc2 output / fc3 input
R_INPUT_DIM = R_FT_DIM * 2 + 66  # 322

# Python NNUE class defaults (may differ from Rust)
P_FT_DIM = 128
P_HD_DIM = 512

def load_teacher_weights(teacher_path):
    """Load teacher and convert to Rust-compatible large arch (128/512) if needed."""
    sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state", sd) if isinstance(sd, dict) and "model_state" in sd else sd
    ft_dim = sd["ft.weight"].shape[0]
    hd = sd["fc1.weight"].shape[0]
    
    if ft_dim == R_FT_DIM and hd == R_HD_DIM:
        # Already matching Rust arch
        m = NNUE(ft_dim=R_FT_DIM, hidden_dim=R_HD_DIM)
        m.load_state_dict(sd, strict=False)
        return {k: v.cpu() for k, v in m.state_dict().items()}, True
    
    # Need to pad small -> large
    print(f"  Converting teacher ft={ft_dim} hd={hd} -> ft={R_FT_DIM} hd={R_HD_DIM}", flush=True)
    m_small = NNUE(ft_dim=ft_dim, hidden_dim=hd)
    m_small.load_state_dict(sd, strict=False)
    
    m_big = NNUE(ft_dim=R_FT_DIM, hidden_dim=R_HD_DIM)
    with torch.no_grad():
        # FT: copy first ft_dim rows, rest zero
        m_big.ft.weight.data[:ft_dim, :] = m_small.ft.weight
        m_big.ft.bias.data[:ft_dim] = m_small.ft.bias
        # FC1: input dim differs! (ft*2+66)
        small_input_dim = ft_dim * 2 + 66
        m_big.fc1.weight.data[:hd, :small_input_dim] = m_small.fc1.weight
        m_big.fc1.bias.data[:hd] = m_small.fc1.bias
        # FC2
        small_hd2 = ft_dim * 2  # old hd2
        hd2 = m_small.fc2.weight.shape[0]
        m_big.fc2.weight.data[:hd2, :hd] = m_small.fc2.weight
        m_big.fc2.bias.data[:hd2] = m_small.fc2.bias
        # FC3
        m_big.fc3.weight.data[0, :hd2] = m_small.fc3.weight[0]
        m_big.fc3.bias.data[0] = m_small.fc3.bias[0]
    
    return {k: v.cpu() for k, v in m_big.state_dict().items()}, True

def flat_weights_rust(model):
    """Flatten to Rust AB+NNUE format (small arch: 64/256/128)."""
    w = model.state_dict()
    # Handle PyTorch model: use correct dims regardless of ft_dim
    ft = w['ft.weight'].cpu().numpy()
    if ft.shape[0] != R_FT_DIM:
        # Model is large arch, can't flatten to Rust small arch
        raise ValueError(f"Model ft_dim={ft.shape[0]} != Rust ft_dim={R_FT_DIM}")
    return np.concatenate([
        ft.T.ravel(), w['ft.bias'].cpu().numpy().ravel(),
        w['fc1.weight'].cpu().numpy().ravel(), w['fc1.bias'].cpu().numpy().ravel(),
        w['fc2.weight'].cpu().numpy().ravel(), w['fc2.bias'].cpu().numpy().ravel(),
        w['fc3.weight'].cpu().numpy().ravel(), w['fc3.bias'].cpu().numpy().ravel(),
    ]).astype(np.float32)

def unflat_weights_rust(flat):
    """Unflatten Rust AB+NNUE flat array back into NNUE model (small arch)."""
    m = NNUE(ft_dim=R_FT_DIM, hidden_dim=R_HD_DIM)
    off = 0
    m.ft.weight.data = torch.from_numpy(flat[off:off+360*R_FT_DIM].reshape(R_FT_DIM, 360)); off += 360*R_FT_DIM
    m.ft.bias.data = torch.from_numpy(flat[off:off+R_FT_DIM]); off += R_FT_DIM
    m.fc1.weight.data = torch.from_numpy(flat[off:off+R_HD_DIM*R_INPUT_DIM].reshape(R_HD_DIM, R_INPUT_DIM)); off += R_HD_DIM*R_INPUT_DIM
    m.fc1.bias.data = torch.from_numpy(flat[off:off+R_HD_DIM]); off += R_HD_DIM
    m.fc2.weight.data = torch.from_numpy(flat[off:off+R_HD2_DIM*R_HD_DIM].reshape(R_HD2_DIM, R_HD_DIM)); off += R_HD2_DIM*R_HD_DIM
    m.fc2.bias.data = torch.from_numpy(flat[off:off+R_HD2_DIM]); off += R_HD2_DIM
    m.fc3.weight.data = torch.from_numpy(flat[off:off+R_HD2_DIM].reshape(1, R_HD2_DIM)); off += R_HD2_DIM
    m.fc3.bias.data = torch.from_numpy(flat[off:off+1])
    return m

def evaluate(model, n_games=100, depth=4, seed_offset=0):
    """Quick eval vs Random. Returns win_rate."""
    h_eval = ffi_ab_create(json.dumps({'max_depth': depth, 'tt_size': 65536}))
    h_eval.set_weights({k: v.cpu() for k, v in model.state_dict().items()})
    sm = {'active': 'active', 'occupied': 'occupied', 'used': 'used', 'eliminated': 'eliminated'}
    w = d = l = 0
    for g in range(n_games):
        c = PenguinChessCore(seed=g*9973+42+seed_offset).reset(seed=g*9973+42+seed_offset)
        for _ in range(6):
            leg = c.get_legal_actions()
            if leg: c.step(__import__('random').choice(leg))
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
                ac = h_eval.search(sj, max_depth=depth).get('best_action', 0)
            else:
                ac = __import__('random').choice(leg)
            if ac in leg: c.step(ac)
            elif leg: c.step(leg[0])
        s1, s2 = c.players_scores
        if s1 > s2: w += 1
        elif s1 == s2: d += 1
        else: l += 1
    h_eval = None
    return w / n_games

def gen_selfplay(handle, tag, n_games=N_GAMES, workers=WORKERS):
    """Generate self-play data with AB search. Returns (data_path, num_positions)."""
    data_path = str(OUT_DIR / f'sp_{tag}.bin')
    Path(data_path).unlink(missing_ok=True)
    t0 = time.time()
    cnt = ffi_ab_generate_selfplay_data(handle, num_games=n_games, workers=workers, output_path=data_path)
    elapsed = time.time() - t0
    print(f'  Self-play: {cnt} pos in {elapsed:.0f}s ({elapsed/n_games:.1f}s/game)', flush=True)
    return data_path, cnt

def gen_random_data(tag, n_games=N_RANDOM, depth=DEPTH, workers=WORKERS):
    """Generate random game data with AB score labels."""
    sd = torch.load('models/ab_nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
    sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
    ft_dim = sd['ft.weight'].shape[0]
    hd = sd['fc1.weight'].shape[0]
    m = NNUE(ft_dim=ft_dim, hidden_dim=hd)
    m.load_state_dict(sd, strict=False)
    d = {k: v.cpu() for k, v in m.state_dict().items()}
    cfg = json.dumps({'max_depth': depth, 'tt_size': 65536,
                       'lmr_moves': 3, 'lmr_depth': 1,
                       'nnue_order_depth': 2, 'null_move': True})
    handle = ffi_ab_create(cfg)
    handle.set_weights(d)
    data_path = str(OUT_DIR / f'random_{tag}.bin')
    Path(data_path).unlink(missing_ok=True)
    t0 = time.time()
    cnt = ffi_ab_generate_random_data(handle, num_games=n_games, workers=workers, output_path=data_path)
    elapsed = time.time() - t0
    print(f'  Random: {cnt} pos in {elapsed:.0f}s ({elapsed/n_games:.1f}s/game)', flush=True)
    del handle
    return data_path, cnt

def rust_train(flat_params, data_paths, tag):
    """Train NNUE with buggy Rust Adam on combined data."""
    eng = get_engine()
    ptr = flat_params.ctypes.data_as(POINTER(c_float))
    tcfg = json.dumps({'lr': LR, 'wd': 1e-4, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS})
    data_arg = ','.join(data_paths)
    t0 = time.time()
    eng._lib.ffi_nnue_train(ptr, c_int32(len(flat_params)),
        c_char_p(data_arg.encode()), c_char_p(tcfg.encode()),
        create_string_buffer(4096), 4096)
    elapsed = time.time() - t0
    print(f'  Train: {elapsed:.0f}s', flush=True)
    return flat_params


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Multi-gen AB+NNUE training')
    parser.add_argument('--generations', type=int, default=3, help='Number of generations')
    parser.add_argument('--games', type=int, default=N_GAMES, help='Self-play games per gen')
    parser.add_argument('--random', type=int, default=N_RANDOM, help='Random games per gen')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--teacher', type=str, default='models/ab_nnue/nnue_gen_2.pt')
    parser.add_argument('--resume', type=str, default='', help='Continue from saved .pt')
    args = parser.parse_args()
    
    OUT_DIR.mkdir(exist_ok=True)
    best_wr = 0.0
    best_path = OUT_DIR / 'd6_best.pt'
    total_t0 = time.time()
    
    # Determine starting generation
    start_gen = 1
    teacher_path = args.teacher
    if args.resume:
        start_gen = int(args.resume.split('_gen')[-1].split('.')[0]) if '_gen' in args.resume else 0
        teacher_path = args.resume
    
    for gen in range(start_gen, start_gen + args.generations):
        tag = f'gen{gen}'
        print(f'\n{"="*50}\n  Generation {gen}\n{"="*50}', flush=True)
        t0 = time.time()
        
        # 1. Load teacher (auto-convert to large arch if needed)
        d, ok = None, False
        for try_path in [teacher_path, 'models/ab_nnue/nnue_gen_2.pt']:
            d, ok = load_teacher_weights(try_path)
            if ok:
                print(f'  Teacher: {try_path}', flush=True)
                break
        if not ok:
            raise RuntimeError('No usable teacher found')
        
        cfg = json.dumps({'max_depth': DEPTH, 'tt_size': 65536,
                          'lmr_moves': 3, 'lmr_depth': 1,
                          'nnue_order_depth': 2, 'null_move': True})
        handle = ffi_ab_create(cfg)
        handle.set_weights(d)
        
        # 2. Generate self-play data
        sp_path, n_sp = gen_selfplay(handle, tag, n_games=args.games)
        
        # 3. Generate random diverse data (from fixed gen_2, for diversity)
        r_path, n_r = gen_random_data(tag, n_games=args.random)
        
        # 4. Train from scratch on mixed data using Rust buggy Adam (small arch)
        print(f'  Training from scratch on mixed data...', flush=True)
        fresh_model = NNUE(ft_dim=R_FT_DIM, hidden_dim=R_HD_DIM)
        flat = flat_weights_rust(fresh_model)
        rust_train(flat, [sp_path, r_path], tag)
        model = unflat_weights_rust(flat)
        
        # 5. Evaluate
        wr = evaluate(model, n_games=100, depth=4)
        print(f'  vs Random: {100*wr:.0f}% (gen {gen})', flush=True)
        
        # Save model
        model_path = OUT_DIR / f'd6_{tag}.pt'
        torch.save({'model_state': model.state_dict(), 'win_rate': wr}, str(model_path))
        
        # Track best
        if wr > best_wr:
            best_wr = wr
            shutil.copy2(str(model_path), str(best_path))
            print(f'  [NEW BEST] {100*wr:.0f}%', flush=True)
        
        # Also save best overall
        if wr > 0.7:
            best_name = OUT_DIR / f'd6_{tag}_best.pt'
            shutil.copy2(str(model_path), str(best_name))
        
        # Next gen uses this model as teacher
        teacher_path = str(model_path)
        
        gen_s = time.time() - t0
        print(f'  Time: {gen_s:.0f}s', flush=True)
    
    print(f'\nDone! Total: {time.time()-total_t0:.0f}s', flush=True)
    print(f'Best: {100*best_wr:.0f}% vs Random', flush=True)

if __name__ == '__main__':
    main()
