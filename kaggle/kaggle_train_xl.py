"""Kaggle XL training — optimized with AMP, Dirichlet noise, checkpoint resume."""
import os, sys
os.chdir('/kaggle/working')
sys.path.insert(0, '/kaggle/working/penguinchess')

import subprocess, shutil, json, time, math, pickle, signal
from pathlib import Path

ROOT = Path('/kaggle/working/penguinchess')
CKPT_DIR = Path('/kaggle/output/penguinchess_checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

subprocess.run([sys.executable, '-m', 'pip', 'install', 'numpy', 'tqdm'], check=True)
import torch, numpy as np
print("Python:", sys.version)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), int(torch.cuda.mem_get_info()[1]/1024**3), "GB")

from penguinchess.ai.alphazero_net import AlphaZeroResNetXL, detect_net_arch
from penguinchess.core import PenguinChessCore
from penguinchess.ai.mcts_core import mcts_search_batched, select_action
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dir = ROOT / 'models/alphazero'
model_dir.mkdir(parents=True, exist_ok=True)
best_path = model_dir / 'alphazero_resnet_xl_best.pth'

CKPT_FILE = CKPT_DIR / 'checkpoint.pt'
BUFFER_FILE = CKPT_DIR / 'replay_buffer.pkl'
BEST_FILE = CKPT_DIR / 'best.pth'

# ─── Resume logic ─────────────────────────────────────
loaded_ckpt = False; start_iter = 0; replay_buffer = None
if CKPT_FILE.exists():
    print("Found persistent checkpoint, resuming...")
    ckpt = torch.load(str(CKPT_FILE), map_location='cpu', weights_only=False)
    net = AlphaZeroResNetXL(obs_dim=272).to(device)
    net.load_state_dict(ckpt['model_state'])
    start_iter = ckpt['iteration']
    if BUFFER_FILE.exists():
        with open(BUFFER_FILE, 'rb') as f: replay_buffer = pickle.load(f)
        print(f"  Replay buffer: {len(replay_buffer)} positions")
    print(f"  Resumed from iter {start_iter}")
    loaded_ckpt = True
elif best_path.exists():
    print("Loading best model from working dir...")
    sd = torch.load(str(best_path), map_location='cpu', weights_only=True)
    net = detect_net_arch(sd)(obs_dim=272).to(device)
    net.load_state_dict(sd)
else:
    print("Starting fresh XL model")
    net = AlphaZeroResNetXL(obs_dim=272).to(device)
if not loaded_ckpt: start_iter = 0
params = sum(p.numel() for p in net.parameters())
print(f"Model: {params:,} params")

# ─── Hyperparameters ──────────────────────────────────
CFG = {
    'iterations': 50, 'games_per_iter': 200, 'simulations': 200, 'c_puct': 3.0,
    'batch_size': 2048, 'epochs': 10, 'lr': 1e-3, 'max_buffer': 100000,
    'temp_threshold': 30, 'random_open_moves': 10, 'MCTS_batch_size': 256,
    'checkpoint_interval': 5, 'workers': 8,
}

# ─── Observation encoding ─────────────────────────────
def encode_obs(core):
    obs = core.get_observation()
    board, pieces = obs["board"], obs["pieces"]
    flat = np.zeros(272, dtype=np.float32)
    idx = 0
    for row in board: flat[idx:idx+3] = row; idx += 3
    for row in pieces: flat[idx:idx+4] = row; idx += 4
    flat[204] = float(obs["current_player"]); flat[205] = float(obs["phase"])
    return flat

# ─── Self-play with proper AZ target/behavior separation ──
def self_play_game(net, game_idx):
    core = PenguinChessCore(seed=game_idx*997+42).reset(seed=game_idx*997+42)
    game_data = []
    # Random opening — no MCTS, uniform policy targets
    for _ in range(CFG['random_open_moves']):
        legal = core.get_legal_actions()
        if not legal or core._terminated: core.close(); return []
        flat = encode_obs(core)
        uniform = np.zeros(60, dtype=np.float32); uniform[legal] = 1.0/len(legal)
        game_data.append((flat, uniform, core.current_player))
        core.step(np.random.choice(legal))
    step = CFG['random_open_moves']
    while not core._terminated and step < 500:
        legal = core.get_legal_actions()
        if not legal: break
        # MCTS search with Dirichlet noise (training=True)
        counts, _ = mcts_search_batched(core, net, num_simulations=CFG['simulations'],
            c_puct=CFG['c_puct'], batch_size=CFG['MCTS_batch_size'], training=True)
        # Behavior policy: temperature for action selection
        temp = 1.0 if step < CFG['temp_threshold'] else 0.1
        action = select_action(counts, legal, temp=temp)
        # Training target: ALWAYS use t=1.0 soft visit distribution
        total = sum(counts.values())
        policy = np.zeros(60, dtype=np.float32)
        if total > 0:
            for a, c in counts.items():
                policy[a] = c / total  # t=1.0
        flat = encode_obs(core)
        game_data.append((flat, policy, core.current_player))
        core.step(action); step += 1
    s1, s2 = core.players_scores
    outcome = 1.0 if s1 > s2 else (-1.0 if s2 > s1 else 0.0)
    return [(f, p, outcome if stm == 0 else -outcome) for f, p, stm in game_data]

# ─── Training with AMP ────────────────────────────────
scaler = torch.amp.GradScaler("cuda") if device == 'cuda' else None

def train(net, buf, epochs=10, bs=2048, lr=1e-3):
    n = len(buf)
    if n < 100: return
    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    obs = np.zeros((n,272),dtype=np.float32); pol = np.zeros((n,60),dtype=np.float32); val = np.zeros((n,1),dtype=np.float32)
    for i,(o,p,v) in enumerate(buf): obs[i],pol[i],val[i,0]=o,p,v
    ot=torch.from_numpy(obs).float(); pt=torch.from_numpy(pol).float(); vt=torch.from_numpy(val).float()
    net.train()
    for ep in range(epochs):
        perm=torch.randperm(n); pl=bc=0
        for i in range(0,n,bs):
            idx=perm[i:i+bs]; o=ot[idx].to(device); p=pt[idx].to(device); v=vt[idx].to(device)
            if scaler:
                with torch.amp.autocast("cuda"):
                    logits,val_out=net(o)
                    loss=-(p*F.log_softmax(logits,dim=1)).sum(dim=1).mean()+F.mse_loss(val_out,v)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(),1.0)
                scaler.step(opt); scaler.update()
            else:
                logits,val_out=net(o)
                loss=-(p*F.log_softmax(logits,dim=1)).sum(dim=1).mean()+F.mse_loss(val_out,v)
                loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            pl+=loss.item(); bc+=1
        sched.step()
        if (ep+1)%5==0: print(f"  ep {ep+1:>3d}/{epochs}  loss={pl/bc:.4f}", flush=True)

def atomic_save(path, data):
    tmp = path.with_suffix('.tmp')
    torch.save(data, str(tmp)); tmp.rename(path)

# Graceful shutdown
interrupted = False
def _handler(sig, frame):
    global interrupted; interrupted = True
    print("\nSaving checkpoint...", flush=True)
signal.signal(signal.SIGTERM, _handler)

# ─── Main training loop ───────────────────────────────
if replay_buffer is None:
    replay_buffer = deque(maxlen=CFG['max_buffer'])
best_wr = float('-inf'); total_t0 = time.time()

for it in range(start_iter, CFG['iterations']):
    if interrupted: break
    t0 = time.time()
    print(f"\n{'='*50}\nIter {it+1}/{CFG['iterations']}\n{'='*50}", flush=True)

    # ── Self-play ──
    net.eval(); games_data = []
    with ThreadPoolExecutor(max_workers=CFG['workers']) as pool:
        futs = [pool.submit(self_play_game, net, g+it*1000) for g in range(CFG['games_per_iter'])]
        for i, f in enumerate(as_completed(futs)):
            if interrupted: pool.shutdown(wait=False); break
            games_data.extend(f.result())
            if (i+1)%50==0: print(f"  Games: {i+1}/{CFG['games_per_iter']}", flush=True)
    for item in games_data: replay_buffer.append(item)
    print(f"  Self-play: {len(games_data)} pos in {time.time()-t0:.0f}s | buffer: {len(replay_buffer)}", flush=True)

    # ── Train ──
    net.train()
    train(net, list(replay_buffer), CFG['epochs'], CFG['batch_size'], CFG['lr'])
    print(f"  Iter: {time.time()-t0:.0f}s", flush=True)

    # ── Checkpoint ──
    if (it+1)%CFG['checkpoint_interval']==0 or it==CFG['iterations']-1 or interrupted:
        atomic_save(CKPT_FILE, {'model_state': net.state_dict(), 'iteration': it+1})
        with open(BUFFER_FILE, 'wb') as f: pickle.dump(list(replay_buffer), f)
        print(f"  Checkpoint saved to {CKPT_DIR}", flush=True)
    torch.save(net.state_dict(), str(model_dir / 'alphazero_resnet_xl_latest.pth'))

    # ── Evaluation (bare forward, not MCTS — Kaggle CPU too slow for MCTS eval) ──
    if (it+1)%5==0:
        net.eval(); w=d=l=0
        for g in range(30):
            core=PenguinChessCore(seed=g*973+it*999).reset(seed=g*973+it*999)
            for _ in range(6):
                leg=core.get_legal_actions()
                if leg: core.step(np.random.choice(leg))
            while not core._terminated and core._episode_steps<200:
                leg=core.get_legal_actions()
                if not leg: break
                if core.current_player==0:
                    with torch.no_grad():
                        logits,_=net(torch.from_numpy(encode_obs(core)).float().unsqueeze(0).to(device))
                    action=leg[logits[0,leg].argmax().item()]
                else: action=np.random.choice(leg)
                core.step(action)
            s1,s2=core.players_scores
            if s1>s2: w+=1
            elif s1==s2: d+=1
            else: l+=1
        wr=w/30
        print(f"  vs Random: {w}/{30-d}/{d} (W/L/D) {100*wr:.0f}%", flush=True)
        if wr>best_wr:
            best_wr=wr
            torch.save(net.state_dict(), str(best_path))
            atomic_save(BEST_FILE, net.state_dict())
            print(f"  [NEW BEST] wr={wr:.1%}", flush=True)

print(f"\n{'='*50}", flush=True)
if interrupted:
    print(f"Interrupted at iter {it+1}. Re-run to resume.", flush=True)
else:
    print(f"Complete! Total: {time.time()-total_t0:.0f}s", flush=True)
print(f"Best: {100*best_wr:.0f}% vs Random | Resume from: {CKPT_DIR}", flush=True)
