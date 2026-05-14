"""Kaggle AlphaZero XL training. Run: python kaggle_train_xl.py"""
import os, sys, subprocess, json, time, math, shutil, pickle
from pathlib import Path

# Set up path: the script runs from repo root after git clone
# The script is at KAGGLE_WORKING/penguinchess/kaggle/kaggle_train_xl.py
# REPO_ROOT = parent of kaggle/ directory
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
print('REPO_ROOT:', REPO_ROOT, flush=True)
print('penguinchess dir exists:', (REPO_ROOT / 'penguinchess').exists(), flush=True)

os.environ['GIT_TERMINAL_PROMPT'] = '0'

# Install Rust
if not shutil.which('rustc'):
    print("Installing Rust...")
    subprocess.run(['curl', '--proto', '=https', '--tlsv1.2', '-sSf',
                    'https://sh.rustup.rs', '-o', '/tmp/rustup.sh'], check=True)
    subprocess.run(['sh', '/tmp/rustup.sh', '-y', '--profile', 'minimal'], check=True)
    os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ['PATH']

subprocess.run([sys.executable, '-m', 'pip', 'install', 'numpy', 'tqdm'], check=True)

import torch, numpy as np

print("Python:", sys.version)
print("Rust:", subprocess.run(['rustc', '--version'], capture_output=True, text=True).stdout.strip())
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), int(torch.cuda.mem_get_info()[1]/1024**3), "GB")

# Compile Rust engine
dll = Path('game_engine/target/release/libgame_engine.so')
if not dll.exists():
    print("Compiling Rust engine...")
    subprocess.run(['cargo', 'build', '--release'], cwd='game_engine', check=True)
    print("Rust engine compiled")

from penguinchess.rust_ffi import get_engine
eng = get_engine()
print("Rust engine loaded")

# Load/create XL model
from penguinchess.ai.alphazero_net import AlphaZeroResNetXL, detect_net_arch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dir = Path('models/alphazero')
model_dir.mkdir(parents=True, exist_ok=True)
best_path = model_dir / 'alphazero_resnet_xl_best.pth'
ckpt_path = model_dir / 'alphazero_resnet_xl_checkpoint.pth'

if best_path.exists():
    sd = torch.load(str(best_path), map_location='cpu', weights_only=True)
    net = detect_net_arch(sd)(obs_dim=272).to(device)
    net.load_state_dict(sd)
    start_iter = 0
    print("Loaded best model")
elif ckpt_path.exists():
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    net = AlphaZeroResNetXL(obs_dim=272).to(device)
    net.load_state_dict(ckpt['model_state'])
    start_iter = ckpt['iteration']
    print(f"Resumed from iter {start_iter}")
else:
    net = AlphaZeroResNetXL(obs_dim=272).to(device)
    start_iter = 0
    print("Starting fresh XL model")

params = sum(p.numel() for p in net.parameters())
print(f"Model: {params:,} params")

# Config
CFG = {'iterations': 50, 'games_per_iter': 200, 'simulations': 200, 'c_puct': 3.0,
       'batch_size': 128, 'epochs': 30, 'lr': 1e-3, 'max_buffer': 100000,
       'temp_threshold': 30, 'random_open_moves': 10, 'MCTS_batch_size': 16,
       'checkpoint_interval': 5}

from penguinchess.core import PenguinChessCore
from penguinchess.ai.mcts_core import mcts_search_batched, select_action
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

def encode_obs(core):
    obs = core.get_observation()
    board, pieces = obs["board"], obs["pieces"]
    flat = np.zeros(272, dtype=np.float32)
    idx = 0
    for row in board:
        flat[idx:idx+3] = row; idx += 3
    for row in pieces:
        flat[idx:idx+4] = row; idx += 4
    flat[204] = float(obs["current_player"])
    flat[205] = float(obs["phase"])
    return flat

def self_play_game(net, game_idx):
    core = PenguinChessCore(seed=game_idx*997+42).reset(seed=game_idx*997+42)
    game_data = []
    for _ in range(CFG['random_open_moves']):
        legal = core.get_legal_actions()
        if not legal or core._terminated: core.close(); return []
        flat = encode_obs(core)
        uniform = np.zeros(60, dtype=np.float32)
        uniform[legal] = 1.0 / len(legal)
        game_data.append((flat, uniform, core.current_player))
        core.step(np.random.choice(legal))
    step = CFG['random_open_moves']
    while not core._terminated and step < 500:
        legal = core.get_legal_actions()
        if not legal: break
        counts, _ = mcts_search_batched(core, net, num_simulations=CFG['simulations'],
            c_puct=CFG['c_puct'], batch_size=CFG['MCTS_batch_size'], device=device)
        temp = 1.0 if step < CFG['temp_threshold'] else 0.1
        action = select_action(counts, legal, temp=temp)
        policy = np.zeros(60, dtype=np.float32)
        total = sum(counts.values())
        if total > 0:
            for a, c in counts.items():
                policy[a] = (c**(1/temp)) if temp > 0 else (1.0 if c==max(counts.values()) else 0.0)
            if temp > 0: policy = policy / policy.sum()
        flat = encode_obs(core)
        game_data.append((flat, policy, core.current_player))
        core.step(action)
        step += 1
    s1, s2 = core.players_scores
    outcome = 1.0 if s1 > s2 else (-1.0 if s2 > s1 else 0.0)
    return [(f, p, outcome if stm==0 else -outcome) for f, p, stm in game_data]

def train_on_data(net, replay_buffer, epochs=30, batch_size=128, lr=1e-3):
    n = len(replay_buffer)
    if n < 100: return
    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    obs = np.zeros((n, 272), dtype=np.float32)
    policy = np.zeros((n, 60), dtype=np.float32)
    value = np.zeros((n, 1), dtype=np.float32)
    for i, (o, p, v) in enumerate(replay_buffer):
        obs[i], policy[i], value[i,0] = o, p, v
    obs_t = torch.from_numpy(obs).float()
    pol_t = torch.from_numpy(policy).float()
    val_t = torch.from_numpy(value).float()
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        pl = vl = bc = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            o = obs_t[idx].to(device, non_blocking=True)
            p = pol_t[idx].to(device, non_blocking=True)
            v = val_t[idx].to(device, non_blocking=True)
            logits, val = net(o)
            loss = -(p * F.log_softmax(logits, dim=1)).sum(dim=1).mean() + F.mse_loss(val, v)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            pl += loss.item(); bc += 1
        sched.step()
        if (ep+1)%10==0:
            print(f"  ep {ep+1:>3d}/{epochs}  loss={pl/bc:.4f}  LR={sched.get_last_lr()[0]:.2e}", flush=True)

# Main training
replay_buffer = deque(maxlen=CFG['max_buffer'])
best_wr = 0.0
total_t0 = time.time()

for it in range(start_iter, CFG['iterations']):
    t0 = time.time()
    print(f"\n{'='*50}\nIteration {it+1}/{CFG['iterations']}\n{'='*50}", flush=True)

    net.eval()
    games_data = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(self_play_game, net, g+it*1000) for g in range(CFG['games_per_iter'])]
        for i, f in enumerate(as_completed(futs)):
            games_data.extend(f.result())
            if (i+1)%50==0: print(f"  Games: {i+1}/{CFG['games_per_iter']}", flush=True)

    for item in games_data: replay_buffer.append(item)
    gt = time.time()-t0
    print(f"  Self-play: {len(games_data)} pos in {gt:.0f}s ({gt/CFG['games_per_iter']:.1f}s/game)", flush=True)

    net.train()
    train_on_data(net, list(replay_buffer), epochs=CFG['epochs'], batch_size=CFG['batch_size'], lr=CFG['lr'])
    print(f"  Iter: {time.time()-t0:.0f}s", flush=True)

    if (it+1)%CFG['checkpoint_interval']==0 or it==CFG['iterations']-1:
        torch.save({'model_state': net.state_dict(), 'iteration': it+1}, str(ckpt_path))
        with open('models/alphazero/xl_replay_buffer.pkl', 'wb') as f: pickle.dump(list(replay_buffer), f)
        print(f"  Checkpoint saved", flush=True)

    if (it+1)%5==0:
        net.eval()
        w=d=l=0
        for g in range(30):
            core = PenguinChessCore(seed=g*973+it*999).reset(seed=g*973+it*999)
            for _ in range(6):
                leg = core.get_legal_actions()
                if leg: core.step(np.random.choice(leg))
            while not core._terminated and core._episode_steps<200:
                leg = core.get_legal_actions()
                if not leg: break
                if core.current_player==0:
                    with torch.no_grad():
                        logits,_ = net(torch.from_numpy(encode_obs(core)).float().unsqueeze(0).to(device))
                    action = leg[logits[0,leg].argmax().item()]
                else: action = np.random.choice(leg)
                core.step(action)
            s1,s2 = core.players_scores
            if s1>s2: w+=1
            elif s1==s2: d+=1
            else: l+=1
        wr = w/30
        print(f"  vs Random: {w}/30 ({100*wr:.0f}%)", flush=True)
        if wr > best_wr:
            best_wr = wr
            torch.save(net.state_dict(), str(best_path))
            print(f"  [NEW BEST] wr={wr:.1%}", flush=True)

print(f"\nDone! Total: {time.time()-total_t0:.0f}s", flush=True)
print(f"Best: {100*best_wr:.0f}% vs Random", flush=True)
