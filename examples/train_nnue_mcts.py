"""
NNUE-MCTS AlphaZero-style training.

Pipeline:
  1. Load NNUEMCTS model (initialize policy head, load NNUE trunk)
  2. Self-play: Rust MCTS search (NNUE eval) → (state, visit_dist, result)
  3. Train: CE(policy, visits) + MSE(value, result)
  4. Evaluate vs Random, save best

Usage:
    uv run python examples/train_nnue_mcts.py --iters 10 --games 100 --sims 200
"""
import os, sys, time, json, math, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import FT_DIM, HIDDEN_DIM
from penguinchess.ai.nnue_mcts import NNUEMCTSModel
from penguinchess.ai.sparse_features import state_to_features
from penguinchess.rust_ffi import (
    get_engine, mcts_search_nnue_handle, RustStatefulGame, NNUEMCTSNative,
)


class MCTSGameDataset(Dataset):
    def __init__(self, data: list): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        return d['sparse'], d['dense'], d['policy'], d['value'], d['player']


def collate_mcts(batch):
    """Vectorized batch collation: pad sparse to (B,6), return stm_players."""
    B = len(batch)
    sparse = torch.zeros(B, 6, dtype=torch.long)
    for i, b in enumerate(batch):
        feat = b[0]  # list of ints
        if feat:
            sparse[i, :len(feat)] = torch.tensor(feat, dtype=torch.long)
    dense = torch.stack([torch.from_numpy(b[1]) for b in batch]).float()
    policy = torch.stack([torch.from_numpy(b[2]) for b in batch]).float()
    value = torch.tensor([b[3] for b in batch], dtype=torch.float)
    stm = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return sparse, dense, policy, value, stm


def make_nnue_eval_fn(model: NNUEMCTSModel, device: str = 'cuda'):
    """Create a callable for mcts_search_nnue_handle: (sparse_batch, dense, stm_list) -> (logits_np, values_np)."""
    model.to(device).eval()

    def eval_fn(sparse_batch, dense_np, stm_list):
        dense_t = torch.from_numpy(dense_np).to(device)
        with torch.no_grad():
            logits, values = model.forward(sparse_batch, dense_t, stm_list)
        return logits.cpu().numpy(), values.cpu().numpy()

    return eval_fn


def self_play_game(model: NNUEMCTSModel, native_mcts: NNUEMCTSNative,
                   num_sims: int = 200, seed: int = 42) -> list[dict]:
    """
    Play one self-play game using Rust MCTS with NNUE evaluation.
    """
    core = PenguinChessCore(seed=seed).reset(seed=seed)
    game_data = []

    engine = get_engine()
    rgame = RustStatefulGame(engine, seed + 9999)

    for step in range(200):
        if core._terminated: break
        legal = core.get_legal_actions()
        if not legal: break

        sparse, dense = state_to_features(core)
        player = core.current_player

        # Rust-native NNUE MCTS
        action_counts = native_mcts.search(
            rgame.handle, num_simulations=num_sims, c_puct=1.4)

        # MCTS visit distribution as policy target
        visit_counts = np.zeros(60, dtype=np.float32)
        for a_str, count in action_counts.items():
            a = int(str(a_str))  # Rust returns string keys
            if a < 60:
                visit_counts[a] = count
        if visit_counts.sum() > 0:
            policy_target = visit_counts / visit_counts.sum()
        else:
            policy_target = np.ones(60) / 60

        game_data.append({
            'sparse': sparse, 'dense': dense.astype(np.float32),
            'policy': policy_target, 'player': player,
        })

        # Select action from MCTS = most visited
        if not action_counts:
            action = legal[0] if legal else None
        else:
            action = max(int(k) for k in action_counts.keys())

        if action in legal:
            core.step(action)
            rgame.step(action)
        elif legal:
            core.step(legal[0])
            rgame.step(legal[0])
        else:
            break

    # Value targets from game outcome
    s1, s2 = core.players_scores
    outcome = 1.0 if s1 > s2 else (-1.0 if s2 > s1 else 0.0)
    for d in game_data:
        d['value'] = outcome if d['player'] == 0 else -outcome

    return game_data


def train_model(model: NNUEMCTSModel, data: list[dict], epochs: int = 30,
                batch_size: int = 4096, lr: float = 1e-3, device: str = 'cuda',
                out_path: str | None = None, max_samples: int = 10000) -> NNUEMCTSModel:
    """Train NNUEMCTS with combined policy + value loss.
    
    If data is larger than max_samples, randomly sample a subset.
    """
    import random as _rnd
    train_data = data
    if len(data) > max_samples:
        train_data = _rnd.sample(data, max_samples)
        print(f'    Sampled {len(train_data)}/{len(data)} for training')

    split = int(len(train_data) * 0.8)
    train_loader = DataLoader(MCTSGameDataset(train_data[:split]), batch_size=batch_size, shuffle=True, collate_fn=collate_mcts)
    val_loader = DataLoader(MCTSGameDataset(train_data[split:]), batch_size=batch_size * 2, shuffle=False, collate_fn=collate_mcts)
    model = model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler(device) if device == 'cuda' else None
    best_val = float('inf')
    best_state = model.state_dict().copy()

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for sparse, dense, policy_tgt, value_tgt, stm in train_loader:
            dense = dense.to(device, non_blocking=True)
            policy_tgt = policy_tgt.to(device, non_blocking=True)
            value_tgt = value_tgt.to(device, non_blocking=True)
            stm = stm.to(device, non_blocking=True)
            sparse = sparse.to(device, non_blocking=True)

            if scaler:
                with torch.amp.autocast(device):
                    logits, values = model.forward(sparse, dense, stm)
                    loss = F.cross_entropy(logits, policy_tgt) + 0.5 * F.mse_loss(values, value_tgt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, values = model.forward(sparse, dense, stm)
                loss = F.cross_entropy(logits, policy_tgt) + 0.5 * F.mse_loss(values, value_tgt)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tr_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sparse, dense, policy_tgt, value_tgt, stm in val_loader:
                dense = dense.to(device, non_blocking=True)
                policy_tgt = policy_tgt.to(device, non_blocking=True)
                value_tgt = value_tgt.to(device, non_blocking=True)
                stm = stm.to(device, non_blocking=True)
                sparse = sparse.to(device, non_blocking=True)
                logits, values = model.forward(sparse, dense, stm)
                val_loss += (F.cross_entropy(logits, policy_tgt) + 0.5 * F.mse_loss(values, value_tgt)).item()

        tr_loss /= max(1, len(train_loader)); val_loss /= max(1, len(val_loader))
        scheduler.step(val_loss)
        if val_loss < best_val: best_val = val_loss; best_state = model.state_dict().copy()
        if epoch % 5 == 0: print(f'  ep {epoch+1:>3d}/{epochs}  train={tr_loss:.4f}  val={val_loss:.4f}')

    model.load_state_dict(best_state)
    if out_path:
        torch.save({'model_state': model.state_dict(), 'val_loss': best_val}, out_path)
        print(f'  Saved: {out_path} (val_loss={best_val:.4f})')
    return model


def evaluate_model(model: NNUEMCTSModel, native_mcts: NNUEMCTSNative,
                   num_games: int = 30, num_sims: int = 100) -> float:
    """Evaluate model vs Random using Rust-native NNUE MCTS."""
    model.eval()
    engine = get_engine()
    wins = 0

    for g in range(num_games):
        seed = g * 9973 + 42
        core = PenguinChessCore(seed=seed).reset(seed=seed)
        rgame = RustStatefulGame(engine, seed + 9999)
        for _ in range(6):
            leg = core.get_legal_actions()
            if leg:
                a = random.choice(leg); core.step(a); rgame.step(a)
        while not core._terminated and core._episode_steps < 200:
            legal = core.get_legal_actions()
            if not legal: break
            if core.current_player == 0:
                ac = native_mcts.search(rgame.handle, num_simulations=num_sims, c_puct=1.4)
                action = max(int(k) for k in ac.keys()) if ac else random.choice(legal)
            else:
                action = random.choice(legal)
            if action in legal: core.step(action); rgame.step(action)
            elif legal: core.step(legal[0]); rgame.step(legal[0])
        if core.players_scores[0] > core.players_scores[1]: wins += 1
    return wins / num_games


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--games', type=int, default=500)
    parser.add_argument('--sims', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--resume', type=str, default='models/nnue/nnue_gen_2.pt')
    parser.add_argument('--out-dir', type=str, default='models/nnue_mcts')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-replay', type=int, default=200000)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = NNUEMCTSModel()
    if args.resume and Path(args.resume).exists():
        sd = torch.load(args.resume, map_location='cpu', weights_only=False)
        sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
        model.load_nnue_trunk(sd)
        print(f'Loaded NNUE trunk from {args.resume}')
    else:
        print('Fresh model')

    best_path = out_dir / 'nnue_mcts_best.pt'
    if best_path.exists():
        sd = torch.load(str(best_path), map_location='cpu', weights_only=False)
        sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
        model.load_state_dict(sd, strict=False)
        print(f'Loaded best from {best_path}')

    best_wr = 0.0
    engine = get_engine()

    # Create one native MCTS engine, reuse across iterations
    native_sd_init = {k: v.cpu() for k, v in model.state_dict().items()}
    native_mcts = NNUEMCTSNative(native_sd_init)

    # Replay buffer: accumulate all training data
    replay_buffer: list = []

    for it in range(args.iters):
        print(f"\n{'='*50}")
        print(f'Iteration {it+1}/{args.iters}')
        print(f"{'='*50}")
        print(f'  Self-play: {args.games} games x {args.sims} sims (replay buffer: {len(replay_buffer)})...')
        t0 = time.time()

        # Update weights for current model state
        cur_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        native_mcts.update_weights(cur_sd)

        # Parallel self-play
        all_data = []

        def _play(seed):
            return self_play_game(model, native_mcts, num_sims=args.sims, seed=seed)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_play, it * 10000 + i) for i in range(args.games)]
            for i, f in enumerate(as_completed(futs)):
                try:
                    all_data.extend(f.result())
                except Exception as e:
                    print(f'  Game {i} failed: {e}')
                    import traceback; traceback.print_exc()
                if (i + 1) % 50 == 0:
                    print(f'  [{i+1}/{args.games}] {len(all_data)} pos')

        elapsed = time.time() - t0
        print(f'  {len(all_data)} positions in {elapsed:.0f}s ({elapsed/args.games:.1f}s/game)')

        # Accumulate into replay buffer
        replay_buffer.extend(all_data)
        if len(replay_buffer) > args.max_replay:
            import random as _rnd
            _rnd.shuffle(replay_buffer)
            replay_buffer = replay_buffer[:args.max_replay]
            print(f'  Replay buffer trimmed to {len(replay_buffer)}')

        print(f'  Training {args.epochs} epochs on {len(replay_buffer)} positions...')
        max_samp = min(10000 + it * 2000, 50000)  # grow sample cap over iterations
        model = train_model(model, replay_buffer, epochs=args.epochs,
                            batch_size=args.batch_size, lr=args.lr, device=device,
                            out_path=str(out_dir / f'nnue_mcts_iter_{it+1}.pt'),
                            max_samples=max_samp)

        print(f'  Evaluating vs Random (30 games)...')
        wr = evaluate_model(model, native_mcts, num_games=30, num_sims=max(100, args.sims // 2))
        print(f'  Win rate: {wr*100:.1f}%')
        if wr > best_wr:
            best_wr = wr
            torch.save({'model_state': model.state_dict(), 'win_rate': wr}, str(out_dir / 'nnue_mcts_best.pt'))
            print(f'  [NEW BEST] {wr*100:.1f}%')

    print(f'\nDone! Best WR: {best_wr*100:.1f}%')

if __name__ == '__main__':
    main()

