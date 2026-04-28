"""
NNUE training pipeline.

Generates self-play data using the existing MCTS agent,
then trains the NNUE model via supervised learning on game outcomes.

Training data:
- (board_features, game_outcome) pairs
- Features: sparse PieceHex indices + dense hex values
- Target: game outcome from current player's perspective (-1, 0, +1)
"""

import os
import sys
import math
import time
import random
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE, ARCH_NAME
from penguinchess.ai.sparse_features import (
    state_to_features,
    PIECE_HEX_DIM, DENSE_DIM,
)


# ─── Data Generation ──────────────────────────────────────────


def generate_games(
    num_games: int,
    worker: int = 0,
    seed_offset: int = 0,
) -> list[dict]:
    """
    Generate self-play games using random play.
    
    Args:
        num_games: number of games to generate
        worker: worker ID for unique seeds
        seed_offset: additional seed offset
        
    Returns:
        list of dicts with 'sparse', 'dense', 'value' keys
    """
    data = []
    
    for g in range(num_games):
        seed = seed_offset + worker * 10000 + g
        core = PenguinChessCore(seed=seed + worker * 10000 + g)
        core.reset(seed=seed + worker * 10000 + g)
        
        # Store features for each step
        game_features = []
        players = []
        
        while True:
            sparse, dense = state_to_features(core)
            player = core.current_player
            
            # Get legal action (random play)
            legal = core.get_legal_actions()
            if not legal:
                break
            
            # Pick random action
            action = random.choice(legal)
            
            game_features.append((sparse, dense, player))
            _, _, terminated, _ = core.step(action)
            
            if terminated or core.phase == 'gameover':
                break
        
        # Compute game outcome
        s1, s2 = core.players_scores
        if s1 > s2:
            outcome = +1  # P1 wins
        elif s2 > s1:
            outcome = -1  # P2 wins
        else:
            outcome = 0   # Draw
        
        # Add to dataset (value from P1's perspective)
        for sparse, dense, player in game_features:
            if player == 0:  # P1 to move
                value = outcome
            else:  # P2 to move
                value = -outcome  # Flip for opponent
            data.append({
                'sparse': sparse,
                'dense': dense.tolist(),
                'value': value,
            })
    
    return data


def generate_games_mcts(
    agent,
    num_games: int,
    worker: int = 0,
    seed_offset: int = 0,
    use_rust: bool = False,
) -> list[dict]:
    """
    Generate games using an MCTS agent (higher quality data).
    
    Args:
        agent: an Agent with select_action(core, legal)
        num_games: number of games
        worker: worker ID
        seed_offset: seed offset
        
    Returns:
        list of feature dicts
    """
    data = []
    
    for g in range(num_games):
        if use_rust:
            from penguinchess.rust_ffi import get_engine
            from penguinchess.rust_core import RustCore
            engine = get_engine()
            core = RustCore(engine=engine).reset(seed_offset + worker * 10000 + g)
        else:
            core = PenguinChessCore(seed=seed_offset + worker * 10000 + g)
            core.reset(seed=seed_offset + worker * 10000 + g)
        
        game_features = []
        players = []
        
        while True:
            sparse, dense = state_to_features(core)
            
            player = core.current_player
            legal = core.get_legal_actions()
            if not legal:
                break
            
            action = agent.select_action(core, legal)
            game_features.append((sparse, dense, player))
            
            _, _, terminated, _ = core.step(action)
            
            if terminated or core.phase == 'gameover':
                break
        
        s1, s2 = core.players_scores
        if s1 > s2:
            outcome = +1
        elif s2 > s1:
            outcome = -1
        else:
            outcome = 0
        
        for sparse, dense, player in game_features:
                value = outcome if player == 0 else -outcome
                data.append({
                    'sparse': sparse,
                    'dense': dense.tolist(),
                    'value': value,
                })
    
    return data


# ─── Dataset ──────────────────────────────────────────────────


class NNUEDataset(Dataset):
    """Dataset for NNUE training."""

    def __init__(self, data: list[dict]):
        self.sparse_list = [d['sparse'] for d in data]
        self.dense_list = torch.tensor(
            np.array([d['dense'] for d in data], dtype=np.float32)
        )
        self.value_list = torch.tensor(
            np.array([d['value'] for d in data], dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.value_list)

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.sparse_list[idx],
            self.dense_list[idx],
            self.value_list[idx],
        )


def collate_nnue(batch: list[tuple]) -> tuple:
    """Collate function for NNUE DataLoader."""
    sparse_batch = [b[0] for b in batch]
    dense_batch = torch.stack([b[1] for b in batch])
    value_batch = torch.stack([b[2] for b in batch])
    return sparse_batch, dense_batch, value_batch


# ─── Training ─────────────────────────────────────────────────


def train_nnue(
    model: NNUE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cpu',
    model_dir: Optional[str] = None,
    generation: int = 0,
) -> NNUE:
    """
    Train NNUE model.
    
    Args:
        model: NNUE model instance
        train_loader: DataLoader for training
        val_loader: optional validation DataLoader
        epochs: training epochs
        lr: learning rate
        device: 'cpu' or 'cuda'
        model_dir: directory to save models
        generation: generation number for naming
        
    Returns:
        trained model
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        t0 = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for sparse_batch, dense_batch, value_batch in train_loader:
            dense_batch = dense_batch.to(device)
            value_batch = value_batch.to(device)
            
            pred = model(sparse_batch, dense_batch)
            loss = criterion(pred, value_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= max(1, num_batches)
        
        # Validation
        val_loss = train_loss
        if val_loader:
            model.eval()
            val_loss = 0.0
            v_batches = 0
            with torch.no_grad():
                for sparse_batch, dense_batch, value_batch in val_loader:
                    dense_batch = dense_batch.to(device)
                    value_batch = value_batch.to(device)
                    pred = model(sparse_batch, dense_batch)
                    val_loss += criterion(pred, value_batch).item()
                    v_batches += 1
            val_loss /= max(1, v_batches)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
        
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(
            f"  [{generation}] epoch {epoch:>3d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"LR={lr_now:.1e}  {elapsed:.0f}s"
        )
    
    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Save model
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"nnue_gen_{generation}.pt")
        torch.save({
            'model_state': model.state_dict(),
            'generation': generation,
            'val_loss': best_val_loss,
        }, path)
        print(f"  Model saved: {path}")
    
    return model


# ─── Main pipeline ────────────────────────────────────────────


def run_pipeline(
    num_games: int = 500,
    generations: int = 5,
    epochs_per_gen: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    ft_dim: int = 64,
    hidden_dim: int = 64,
    model_dir: str = 'models/nnue',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    workers: int = 4,
    eval_episodes: int = 100,
):
    """
    Full NNUE training pipeline: self-play → train → evaluate → repeat.
    
    Each generation:
    1. Generate self-play data (random play for gen 0, NNUE agent for gen 1+)
    2. Train NNUE on the data
    3. Evaluate vs baseline
    4. Generate next generation's data with improved agent
    """
    print(f"NNUE Training Pipeline")
    print(f"  Device: {device}")
    print(f"  Games/gen: {num_games}, Generations: {generations}")
    
    model = NNUE(ft_dim=ft_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    
    best_agent = None
    
    for gen in range(generations):
        print(f"\n{'='*50}")
        print(f"Generation {gen + 1}/{generations}")
        print(f"{'='*50}")
        
        # Generate self-play data
        t0 = time.time()
        print(f"  Generating {num_games} games...")
        
        all_data = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            games_per_worker = num_games // max(1, workers)
            futs = []
            for w in range(workers):
                futs.append(pool.submit(
                    generate_games, games_per_worker, w, gen * 100000
                ))
            for fut in as_completed(futs):
                all_data.extend(fut.result())
        
        print(f"  Generated {len(all_data)} positions in {time.time()-t0:.0f}s")
        
        # Split train/val
        random.shuffle(all_data)
        split = int(len(all_data) * 0.8)
        train_data = all_data[:split]
        val_data = all_data[split:]
        
        train_dataset = NNUEDataset(train_data)
        val_dataset = NNUEDataset(val_data)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, collate_fn=collate_nnue,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2,
            shuffle=False, collate_fn=collate_nnue,
        )
        
        # Train
        model = train_nnue(
            model, train_loader, val_loader,
            epochs=epochs_per_gen,
            lr=lr,
            device=device,
            model_dir=model_dir,
            generation=gen,
        )
        
        # Evaluate
        model.eval()
        if gen == 0:
            baseline_wr = _eval_vs_random(model, eval_episodes)
        else:
            baseline_wr = _eval_vs_random(model, eval_episodes)
        
        print(f"  Gen {gen} vs random: {baseline_wr:.1%}")


def _eval_vs_random(model: NNUE, episodes: int = 100) -> float:
    """Evaluate NNUE model vs random play."""
    from penguinchess.eval_utils import compete
    from penguinchess.ai.nnue_agent import NNUEAgent
    
    agent = NNUEAgent(model, max_depth=4)
    
    wins = 0
    total = 0
    for ep in range(episodes):
        core = PenguinChessCore(seed=ep * 1000)
        core.reset(seed=ep * 1000)
        done = False
        while not done and core.phase != 'gameover':
            legal = core.get_legal_actions()
            if not legal:
                break
            action = agent.select_action(core, legal)
            _, _, terminated, _ = core.step(action)
            if terminated:
                break
        
        s1, s2 = core.players_scores
        if s1 > s2:
            wins += 1
        total += 1
    
    return wins / max(1, total)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="NNUE Training")
    parser.add_argument('--games', type=int, default=500,
                        help='Games per generation')
    parser.add_argument('--generations', type=int, default=5,
                        help='Training generations')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Epochs per generation')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ft-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--model-dir', type=str, default='models/nnue')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    
    run_pipeline(
        num_games=args.games,
        generations=args.generations,
        epochs_per_gen=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ft_dim=args.ft_dim,
        hidden_dim=args.hidden_dim,
        model_dir=args.model_dir,
        workers=args.workers,
    )
