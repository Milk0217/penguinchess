"""Kaggle XL — Rust MCTS + GPU AMP, checkpoint resume."""
import os, sys
os.chdir('/kaggle/working')
sys.path.insert(0, '/kaggle/working/penguinchess')

import subprocess, shutil, json, time, math, pickle, signal
from pathlib import Path

ROOT = Path('/kaggle/working/penguinchess')
CKPT_DIR = Path('/kaggle/output/penguinchess_checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Install Rust & compile engine
if not shutil.which('rustc'):
    print("Installing Rust...")
    subprocess.run(['curl', '--proto', '=https', '--tlsv1.2', '-sSf',
        'https://sh.rustup.rs', '-o', '/tmp/rustup.sh'], check=True)
    subprocess.run(['sh', '/tmp/rustup.sh', '-y', '--profile', 'minimal'], check=True)
    os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ['PATH']

dll = ROOT / 'game_engine/target/release/libgame_engine.so'
if not dll.exists():
    print("Compiling Rust engine (one-time ~2min)...")
    subprocess.run(['cargo', 'build', '--release'], cwd=str(ROOT/'game_engine'), check=True)

subprocess.run([sys.executable, '-m', 'pip', 'install', 'numpy', 'tqdm'], check=True)
import torch, numpy as np
print("Python:", sys.version, "CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), int(torch.cuda.mem_get_info()[1]/1024**3), "GB")

from penguinchess.rust_ffi import get_engine, mcts_search_rust_handle
from penguinchess.rust_core import RustCore
from penguinchess.ai.alphazero_net import AlphaZeroResNetXL, detect_net_arch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

device = 'cuda' if torch.cuda.is_available() else 'cpu'
engine = get_engine()

model_dir = ROOT / 'models/alphazero'; model_dir.mkdir(parents=True, exist_ok=True)
best_path = model_dir / 'alphazero_resnet_xl_best.pth'
CKPT_FILE = CKPT_DIR / 'checkpoint.pt'; BUFFER_FILE = CKPT_DIR / 'replay_buffer.pkl'
BEST_FILE = CKPT_DIR / 'best.pth'

loaded_ckpt = False; start_iter = 0; replay_buffer = None
if CKPT_FILE.exists():
    print("Resuming from checkpoint...")
    ckpt = torch.load(str(CKPT_FILE), map_location='cpu', weights_only=False)
    net = AlphaZeroResNetXL(obs_dim=272).to(device)
    net.load_state_dict(ckpt['model_state']); start_iter = ckpt['iteration']
    if BUFFER_FILE.exists():
        with open(BUFFER_FILE, 'rb') as f: replay_buffer = pickle.load(f)
        print(f"  Buffer: {len(replay_buffer)} pos")
    loaded_ckpt = True
elif best_path.exists():
    sd = torch.load(str(best_path), map_location='cpu', weights_only=True)
    net = detect_net_arch(sd)(obs_dim=272).to(device); net.load_state_dict(sd)
    print("Loaded best model, training from iter 0")
else:
    net = AlphaZeroResNetXL(obs_dim=272).to(device); print("Fresh XL model")
if not loaded_ckpt: start_iter = 0
print(f"Model: {sum(p.numel() for p in net.parameters()):,} params")

CFG = {'iterations': 50, 'games_per_iter': 200, 'simulations': 200, 'c_puct': 3.0,
       'batch_size': 2048, 'epochs': 10, 'lr': 1e-3, 'max_buffer': 100000,
       'temp_threshold': 30, 'random_open_moves': 10, 'mcts_batch_size': 128,
       'checkpoint_interval': 5, 'workers': 8}

def encode_obs(core):
    obs = core.get_observation(); b, p = obs["board"], obs["pieces"]
    flat = np.zeros(272, dtype=np.float32); idx = 0
    for r in b: flat[idx:idx+3] = r; idx += 3
    for r in p: flat[idx:idx+4] = r; idx += 4
    flat[204] = float(obs["current_player"]); flat[205] = float(obs["phase"])
    return flat

def sel_action(counts, legal, temp=1.0):
    if temp == 0:
        b = max(counts.values()); acts = [a for a, c in counts.items() if c == b]
        return np.random.choice(acts)
    p = np.zeros(60, dtype=np.float32)
    for a, c in counts.items(): p[a] = c ** (1.0 / temp)
    s = p.sum()
    if s > 0: p /= s
    else: p[legal] = 1.0 / len(legal)
    return np.random.choice(60, p=p)

def self_play(net, game_idx, eng):
    seed = game_idx * 997 + 42 + start_iter * 10000
    core = RustCore(engine=eng, seed=seed).reset(seed=seed)
    data = []; terminated = False
    for _ in range(CFG['random_open_moves']):
        legal = core.get_legal_actions()
        if not legal: core.close(); return []
        flat = encode_obs(core)
        u = np.zeros(60, dtype=np.float32); u[legal] = 1.0/len(legal)
        data.append((flat, u, core.current_player))
        _, _, term, _ = core.step(np.random.choice(legal))
        if term: terminated = True; break
    step = CFG['random_open_moves']
    while not terminated and step < 500:
        legal = core.get_legal_actions()
        if not legal: break
        raw = mcts_search_rust_handle(core.handle, model=net,
            num_simulations=CFG['simulations'], c_puct=CFG['c_puct'],
            batch_size=CFG['mcts_batch_size'])
        counts = {int(k): v for k, v in raw.items()}
        if not counts: core.close(); return []
        temp = 1.0 if step < CFG['temp_threshold'] else 0.1
        action = sel_action(counts, legal, temp=temp)
        total = sum(counts.values()); policy = np.zeros(60, dtype=np.float32)
        if total > 0:
            for a, c in counts.items(): policy[a] = c / total
        flat = encode_obs(core); data.append((flat, policy, core.current_player))
        _, _, term, _ = core.step(action)
        if term: break; step += 1
    s1, s2 = core.players_scores
    outcome = 1.0 if s1 > s2 else (-1.0 if s2 > s1 else 0.0)
    result = [(f, p, outcome if stm == 0 else -outcome) for f, p, stm in data]
    core.close(); return result

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
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(),1.0); scaler.step(opt); scaler.update()
            else:
                logits,val_out=net(o)
                loss=-(p*F.log_softmax(logits,dim=1)).sum(dim=1).mean()+F.mse_loss(val_out,v)
                loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            pl+=loss.item(); bc+=1
        sched.step()
        if (ep+1)%5==0: print(f"  ep {ep+1:>3d}/{epochs}  loss={pl/bc:.4f}", flush=True)

def atomic_save(path, data):
    tmp = path.with_suffix('.tmp'); torch.save(data, str(tmp)); tmp.rename(path)

interrupted = False
def _handler(sig, frame): global interrupted; interrupted = True; print("\nSaving...", flush=True)
signal.signal(signal.SIGTERM, _handler)

if replay_buffer is None: replay_buffer = deque(maxlen=CFG['max_buffer'])
best_wr = float('-inf'); total_t0 = time.time()

for it in range(start_iter, CFG['iterations']):
    t0 = time.time()
    if interrupted: break
    print(f"\n{'='*50}\nIter {it+1}/{CFG['iterations']}\n{'='*50}", flush=True)
    net.eval(); gd = []
    with ThreadPoolExecutor(max_workers=CFG['workers']) as pool:
        futs = [pool.submit(self_play, net, g+it*1000, engine) for g in range(CFG['games_per_iter'])]
        for i, f in enumerate(as_completed(futs)):
            if interrupted: pool.shutdown(wait=False); break
            r = f.result()
            if r: gd.extend(r)
            if (i+1)%50==0: print(f"  Games: {i+1}/{CFG['games_per_iter']}", flush=True)
    for item in gd: replay_buffer.append(item)
    print(f"  Self-play: {len(gd)} pos in {time.time()-t0:.0f}s | buf: {len(replay_buffer)}", flush=True)
    net.train(); train(net, list(replay_buffer), CFG['epochs'], CFG['batch_size'], CFG['lr'])
    print(f"  Iter: {time.time()-t0:.0f}s", flush=True)
    if (it+1)%CFG['checkpoint_interval']==0 or it==CFG['iterations']-1 or interrupted:
        atomic_save(CKPT_FILE, {'model_state': net.state_dict(), 'iteration': it+1})
        with open(BUFFER_FILE, 'wb') as f: pickle.dump(list(replay_buffer), f)
    torch.save(net.state_dict(), str(model_dir/'alphazero_resnet_xl_latest.pth'))
    if (it+1)%5==0:
        net.eval();         w=d=l=0
        for g in range(30):
            core=RustCore(engine=engine, seed=g*973+it*999).reset(seed=g*973+it*999)
            term=False; ep=0
            for _ in range(6):
                leg=core.get_legal_actions()
                if leg: _,_,term,_=core.step(np.random.choice(leg))
                ep+=1
                if term: break
            while not term and ep<200:
                leg=core.get_legal_actions()
                if not leg: break
                if core.current_player==0:
                    with torch.no_grad():
                        logits,_=net(torch.from_numpy(encode_obs(core)).float().unsqueeze(0).to(device))
                    action=leg[logits[0,leg].argmax().item()]
                else: action=np.random.choice(leg)
                _,_,term,_=core.step(action)
                if term: break
            s1,s2=core.players_scores; core.close()
            if s1>s2: w+=1
            elif s1==s2: d+=1
            else: l+=1
        wr=w/30; print(f"  vs Random: {w}/{30-d}/{d} {100*wr:.0f}%", flush=True)
        if wr>best_wr:
            best_wr=wr; torch.save(net.state_dict(), str(best_path))
            atomic_save(BEST_FILE, net.state_dict())
            print(f"  [NEW BEST] {wr:.1%}", flush=True)

print(f"\n{'='*50}", flush=True)
print(f"{'Interrupted' if interrupted else 'Complete'}! Total: {time.time()-total_t0:.0f}s", flush=True)
print(f"Best: {100*best_wr:.0f}% vs Random", flush=True)
