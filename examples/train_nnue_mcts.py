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
    get_engine, mcts_search_nnue_handle, RustStatefulGame,
    CFUNCTYPE, POINTER, c_float, c_int32, create_string_buffer,
)


class MCTSGameDataset(Dataset):
    def __init__(self, data: list): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        return d['sparse'], d['dense'], d['policy'], d['value']


def collate_mcts(batch):
    sparse = [b[0] for b in batch]
    dense = torch.stack([torch.from_numpy(b[1]) for b in batch]).float()
    policy = torch.stack([torch.from_numpy(b[2]) for b in batch]).float()
    value = torch.tensor([b[3] for b in batch]).float()
    return sparse, dense, policy, value


def make_nnue_eval_fn(model: NNUEMCTSModel, device: str = 'cuda'):
    """Create a callable for mcts_search_nnue_handle: (sparse_batch, dense, stm_list) -> (logits_np, values_np)."""
    model.to(device).eval()

    def eval_fn(sparse_batch, dense_np, stm_list):
        dense_t = torch.from_numpy(dense_np).to(device)
        with torch.no_grad():
            logits, values = model.forward(sparse_batch, dense_t, stm_list)
        return logits.cpu().numpy(), values.cpu().numpy()

    return eval_fn


def self_play_game(model: NNUEMCTSModel, num_sims: int = 200,
                   seed: int = 42, temp: float = 1.0) -> list[dict]:
    """
    Play one self-play game using Rust MCTS with NNUE evaluation.
    """
    core = PenguinChessCore(seed=seed).reset(seed=seed)
    game_data = []
    nnue_eval = make_nnue_eval_fn(model)

    # Create Rust stateful game for MCTS
    engine = get_engine()
    rgame = RustStatefulGame(engine, seed)

    for step in range(200):
        if core._terminated: break
        legal = core.get_legal_actions()
        if not legal: break

        sparse, dense = state_to_features(core)
        player = core.current_player

        # Rust MCTS with NNUE eval
        action_counts = mcts_search_nnue_handle(
            rgame.handle, nnue_eval,
            num_simulations=num_sims, c_puct=1.4, batch_size=32,
        )

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
                batch_size: int = 256, lr: float = 1e-3, device: str = 'cuda',
                out_path: str | None = None) -> NNUEMCTSModel:
    """Train NNUEMCTS with combined policy + value loss."""
    split = int(len(data) * 0.8)
    train_loader = DataLoader(MCTSGameDataset(data[:split]), batch_size=batch_size, shuffle=True, collate_fn=collate_mcts)
    val_loader = DataLoader(MCTSGameDataset(data[split:]), batch_size=batch_size * 2, shuffle=False, collate_fn=collate_mcts)
    model = model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val = float('inf')
    best_state = model.state_dict().copy()

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for sparse, dense, policy_tgt, value_tgt in train_loader:
            dense, policy_tgt, value_tgt = dense.to(device), policy_tgt.to(device), value_tgt.to(device)
            logits, values = model.forward(sparse, dense)
            loss = F.cross_entropy(logits, policy_tgt) + 0.5 * F.mse_loss(values, value_tgt)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sparse, dense, policy_tgt, value_tgt in val_loader:
                dense, policy_tgt, value_tgt = dense.to(device), policy_tgt.to(device), value_tgt.to(device)
                logits, values = model.forward(sparse, dense)
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


def evaluate_model(model: NNUEMCTSModel, num_games: int = 30,
                   num_sims: int = 100, device: str = 'cuda') -> float:
    """Evaluate model vs Random using Rust MCTS."""
    model.to(device).eval()
    engine = get_engine()
    nnue_eval = make_nnue_eval_fn(model)
    wins = 0

    for g in range(num_games):
        seed = g * 9973 + 42
        core = PenguinChessCore(seed=seed).reset(seed=seed)
        rgame = RustStatefulGame(engine, seed)
        for _ in range(6):
            leg = core.get_legal_actions()
            if leg:
                a = random.choice(leg); core.step(a); rgame.step(a)
        while not core._terminated and core._episode_steps < 200:
            legal = core.get_legal_actions()
            if not legal: break
            if core.current_player == 0:
                ac = mcts_search_nnue_handle(rgame.handle, nnue_eval, num_simulations=num_sims, c_puct=1.4, batch_size=32)
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
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--sims', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--resume', type=str, default='models/nnue/nnue_gen_2.pt')
    parser.add_argument('--out-dir', type=str, default='models/nnue_mcts')
    parser.add_argument('--workers', type=int, default=4)
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

    for it in range(args.iters):
        print(f"\n{'='*50}")
        print(f'Iteration {it+1}/{args.iters}')
        print(f"{'='*50}")
        print(f'  Self-play: {args.games} games x {args.sims} sims...')
        t0 = time.time()

        # Parallel self-play
        all_data = []
        nnue_eval = make_nnue_eval_fn(model)

        def _play(seed):
            return self_play_game(model, num_sims=args.sims, seed=seed, temp=1.0)

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

        print(f'  Training {args.epochs} epochs...')
        model = train_model(model, all_data, epochs=args.epochs,
                            batch_size=args.batch_size, lr=args.lr, device=device,
                            out_path=str(out_dir / f'nnue_mcts_iter_{it+1}.pt'))

        print(f'  Evaluating vs Random (30 games)...')
        wr = evaluate_model(model, num_games=30, num_sims=max(100, args.sims // 2), device=device)
        print(f'  Win rate: {wr*100:.1f}%')
        if wr > best_wr:
            best_wr = wr
            torch.save({'model_state': model.state_dict(), 'win_rate': wr}, str(out_dir / 'nnue_mcts_best.pt'))
            print(f'  [NEW BEST] {wr*100:.1f}%')

    print(f'\nDone! Best WR: {best_wr*100:.1f}%')

if __name__ == '__main__':
    main()

