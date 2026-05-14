"""Kaggle AlphaZero XL training notebook.
Upload as Kaggle Notebook, set GPU accelerator (T4 x2), run end-to-end.

Structure:
  1. Install Rust, compile game_engine
  2. Load project code from dataset
  3. Download best model from previous Kaggle run (or start fresh)
  4. Train AlphaZeroResNetXL (312M params) with Python MCTS + GPU batch eval
  5. Save model, upload back to dataset
"""

# ═══════════════════════════════════════════════
# Cell 1 — Install Rust & Dependencies
# ═══════════════════════════════════════════════
import sys, os, subprocess, json, time, math, shutil, pickle
from pathlib import Path
from urllib.request import urlretrieve

# Install Rust (if not already)
if not shutil.which('rustc'):
    print("Installing Rust...")
    subprocess.run(['curl', '--proto', '=https', '--tlsv1.2', '-sSf', 
                    'https://sh.rustup.rs', '-o', '/tmp/rustup.sh'], check=True)
    subprocess.run(['sh', '/tmp/rustup.sh', '-y', '--profile', 'minimal'], check=True)
    rust_home = os.path.expanduser('~/.cargo/bin')
    os.environ['PATH'] = f"{rust_home}:{os.environ['PATH']}"
    print("Rust installed")

# Install Python deps
subprocess.run([sys.executable, '-m', 'pip', 'install', 
    'torch', 'numpy', 'tqdm', 'kagglehub'], check=True)

print(f"Python: {sys.version}")
print(f"Rust:   {subprocess.run(['rustc', '--version'], capture_output=True, text=True).stdout.strip()}")
print(f"CUDA:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)} ({torch.cuda.mem_get_info()[1]/1024**3:.0f}GB)")


# ═══════════════════════════════════════════════
# Cell 2 — Setup project & compile Rust engine
# ═══════════════════════════════════════════════
import torch

# Clone/pull project
PROJECT_DIR = Path('/kaggle/working/penguinchess')
if not PROJECT_DIR.exists():
    print("Cloning project...")
    subprocess.run(['git', 'clone', 'https://github.com/YOUR_USER/penguinchess.git', str(PROJECT_DIR)], check=True)
    
    # Or copy from dataset
    # shutil.copytree('/kaggle/input/penguinchess-code', str(PROJECT_DIR))
    
os.chdir(str(PROJECT_DIR))
print(f"Working dir: {PROJECT_DIR}")

# Check if we need to compile Rust
dll_path = PROJECT_DIR / 'game_engine' / 'target' / 'release' / 'libgame_engine.so'
if not dll_path.exists():
    print("Compiling Rust engine...")
    subprocess.run(['cargo', 'build', '--release'], cwd=str(PROJECT_DIR / 'game_engine'), check=True)
    print(f"Rust engine compiled: {dll_path}")
else:
    print(f"Rust engine already compiled")

# Verify Python can load it
sys.path.insert(0, str(PROJECT_DIR))
from penguinchess.rust_ffi import get_engine
eng = get_engine()
print(f"Rust engine loaded: {eng._lib}")


# ═══════════════════════════════════════════════
# Cell 3 — Load/Resume Model
# ═══════════════════════════════════════════════
from penguinchess.ai.alphazero_net import AlphaZeroResNetXL, detect_net_arch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Try to load model from previous run (local upload) or start fresh
model_dir = PROJECT_DIR / 'models' / 'alphazero'
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / 'alphazero_resnet_xl_checkpoint.pth'
best_path = model_dir / 'alphazero_resnet_xl_best.pth'

if best_path.exists():
    print(f"Loading best model: {best_path}")
    sd = torch.load(str(best_path), map_location='cpu', weights_only=True)
    net = detect_net_arch(sd)(obs_dim=272).to(device)
    net.load_state_dict(sd)
    start_iter = 0
elif model_path.exists():
    print(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
    net = AlphaZeroResNetXL(obs_dim=272).to(device)
    net.load_state_dict(ckpt['model_state'])
    start_iter = ckpt['iteration']
    print(f"  Resuming from iter {start_iter}")
else:
    print("Starting fresh XL model from scratch")
    net = AlphaZeroResNetXL(obs_dim=272).to(device)
    start_iter = 0

total_params = sum(p.numel() for p in net.parameters())
print(f"Model: {total_params:,} params, {sum(p.numel() for p in net.parameters() if p.requires_grad):,} trainable")


# ═══════════════════════════════════════════════
# Cell 4 — Training Hyperparameters
# ═══════════════════════════════════════════════
CFG = {
    'iterations': 50,
    'games_per_iter': 200,
    'simulations': 200,
    'c_puct': 3.0,
    'batch_size': 128,  # Small batch for XL (312M) — larger = slower
    'epochs': 30,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'max_buffer': 100000,
    'temp_threshold': 30,
    'random_open_moves': 10,
    'MCTS_batch_size': 16,
    'checkpoint_interval': 5,
}
print(f"Training config: {json.dumps(CFG, indent=2)}")


# ═══════════════════════════════════════════════
# Cell 5 — Self-play with Python MCTS + GPU eval
# ═══════════════════════════════════════════════
from penguinchess.core import PenguinChessCore
from penguinchess.ai.mcts_core import mcts_search_batched, select_action
from penguinchess.rust_ffi import get_engine  # Only for RustCore fallback, fallback to Python

def _encode_obs(core):
    """Encode 272-dim observation from Python core."""
    obs = core.get_observation()
    board = obs["board"]
    pieces = obs["pieces"]
    flat = np.zeros(272, dtype=np.float32)
    idx = 0
    for row in board:
        flat[idx:idx+3] = row
        idx += 3
    for row in pieces:
        flat[idx:idx+4] = row
        idx += 4
    flat[204] = float(obs["current_player"])
    flat[205] = float(obs["phase"])
    return flat

def self_play_game(net, game_idx, device='cuda'):
    """Self-play one game using Python MCTS + GPU batch eval."""
    core = PenguinChessCore(seed=game_idx * 997 + 42).reset(seed=game_idx * 997 + 42)
    game_data = []
    
    # Random opening
    for step in range(CFG['random_open_moves']):
        legal = core.get_legal_actions()
        if not legal or core._terminated:
            core.close()
            return []
        flat = _encode_obs(core)
        uniform = np.zeros(60, dtype=np.float32)
        uniform[legal] = 1.0 / len(legal)
        game_data.append((flat, uniform, core.current_player))
        core.step(np.random.choice(legal))
    
    # MCTS self-play
    step_count = CFG['random_open_moves']
    while not core._terminated and step_count < 500:
        legal = core.get_legal_actions()
        if not legal:
            break
        
        # Search with MCTS
        counts, values = mcts_search_batched(
            core, net, num_simulations=CFG['simulations'],
            c_puct=CFG['c_puct'], batch_size=CFG['MCTS_batch_size'],
            device=device)
        
        temp = 1.0 if step_count < CFG['temp_threshold'] else 0.1
        action = select_action(counts, legal, temp=temp)
        
        # Record training data
        policy = np.zeros(60, dtype=np.float32)
        total = sum(counts.values())
        if total > 0:
            for a, c in counts.items():
                policy[a] = (c ** (1/temp)) if temp > 0 else (1.0 if c == max(counts.values()) else 0.0)
            if temp > 0:
                policy = policy / policy.sum()
        
        flat = _encode_obs(core)
        game_data.append((flat, policy, core.current_player))
        core.step(action)
        step_count += 1
    
    # Compute outcome
    s1, s2 = core.players_scores
    outcome = 1.0 if s1 > s2 else (-1.0 if s2 > s1 else 0.0)
    
    # Assign value labels
    result = []
    for flat, pol, stm in game_data:
        value = outcome if stm == 0 else -outcome
        result.append((flat, pol, value))
    
    return result


# ═══════════════════════════════════════════════
# Cell 6 — Training loop (pure PyTorch)
# ═══════════════════════════════════════════════
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

def train_on_data(net, replay_buffer, epochs=30, batch_size=128, lr=1e-3, device='cuda'):
    """Train AZ network on replay buffer data."""
    n = len(replay_buffer)
    if n < 100:
        return
    
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Pre-allocate arrays
    obs = np.zeros((n, 272), dtype=np.float32)
    policy = np.zeros((n, 60), dtype=np.float32)
    value = np.zeros((n, 1), dtype=np.float32)
    
    for i, (o, p, v) in enumerate(replay_buffer):
        obs[i] = o
        policy[i] = p
        value[i, 0] = v
    
    obs_t = torch.from_numpy(obs).float()
    policy_t = torch.from_numpy(policy).float()
    value_t = torch.from_numpy(value).float()
    
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_p_loss = 0.0
        total_v_loss = 0.0
        batches = 0
        
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            o = obs_t[idx].to(device, non_blocking=True)
            p = policy_t[idx].to(device, non_blocking=True)
            v = value_t[idx].to(device, non_blocking=True)
            
            logits, val = net(o)
            p_loss = -(p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            v_loss = F.mse_loss(val, v)
            loss = p_loss + v_loss
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            batches += 1
        
        scheduler.step()
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:>3d}/{epochs}  P={total_p_loss/batches:.4f}  V={total_v_loss/batches:.4f}  LR={scheduler.get_last_lr()[0]:.2e}", flush=True)


# ═══════════════════════════════════════════════
# Cell 7 — Main Training Loop
# ═══════════════════════════════════════════════
import pickle

replay_buffer = deque(maxlen=CFG['max_buffer'])
best_wr = 0.0

# Load replay buffer from checkpoint if resuming
ckpt_buffer = PROJECT_DIR / 'data' / 'xl_replay_buffer.pkl'
if ckpt_buffer.exists():
    with open(ckpt_buffer, 'rb') as f:
        replay_buffer.extend(pickle.load(f))
    print(f"Loaded {len(replay_buffer)} positions from replay buffer")

total_t0 = time.time()
for it in range(start_iter, CFG['iterations']):
    t0 = time.time()
    print(f"\n{'='*50}")
    print(f"Iteration {it+1}/{CFG['iterations']}")
    print(f"{'='*50}")
    
    # Self-play
    net.eval()
    games_data = []
    n_games = CFG['games_per_iter']
    
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(self_play_game, net, g + it * 1000) for g in range(n_games)]
        for i, f in enumerate(as_completed(futs)):
            gd = f.result()
            games_data.extend(gd)
            if (i+1) % 50 == 0:
                print(f"  Games: {i+1}/{n_games}", flush=True)
    
    # Add to replay buffer
    for item in games_data:
        replay_buffer.append(item)
    
    gen_time = time.time() - t0
    print(f"  Self-play: {len(games_data)} positions in {gen_time:.0f}s ({gen_time/n_games:.1f}s/game)")
    
    # Training
    net.train()
    train_on_data(net, list(replay_buffer), epochs=CFG['epochs'],
                  batch_size=CFG['batch_size'], lr=CFG['lr'], device=device)
    
    iter_time = time.time() - t0
    print(f"  Iteration time: {iter_time:.0f}s ({iter_time/60:.1f}min)")
    
    # Save checkpoint
    if (it + 1) % CFG['checkpoint_interval'] == 0 or it == CFG['iterations'] - 1:
        torch.save({
            'model_state': net.state_dict(),
            'iteration': it + 1,
        }, str(model_path))
        # Save replay buffer
        with open(ckpt_buffer, 'wb') as f:
            pickle.dump(list(replay_buffer), f)
        print(f"  Checkpoint saved (iter {it+1})")
        
    # Brief evaluation (vs random, simplified)
    if (it + 1) % 5 == 0:
        net.eval()
        n_eval = 30
        w = d = l = 0
        for g in range(n_eval):
            core = PenguinChessCore(seed=g*973+it*999).reset(seed=g*973+it*999)
            for _ in range(6):
                leg = core.get_legal_actions()
                if leg: core.step(np.random.choice(leg))
            while not core._terminated and core._episode_steps < 200:
                leg = core.get_legal_actions()
                if not leg: break
                if core.current_player == 0:
                    obs_t = torch.from_numpy(_encode_obs(core)).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits, _ = net(obs_t)
                    action = leg[logits[0, leg].argmax().item()]
                else:
                    action = np.random.choice(leg)
                core.step(action)
            s1, s2 = core.players_scores
            if s1 > s2: w += 1
            elif s1 == s2: d += 1
            else: l += 1
        wr = w / n_eval
        print(f"  vs Random: {w}/{n_eval} ({100*wr:.0f}%)")
        
        if wr > best_wr:
            best_wr = wr
            torch.save(net.state_dict(), str(best_path))
            print(f"  [NEW BEST] wr={wr:.1%}")

total_time = time.time() - total_t0
print(f"\n{'='*50}")
print(f"Training complete! Total: {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"Best vs Random: {100*best_wr:.0f}%")
print(f"Model: {best_path}")
print(f"{'='*50}")
