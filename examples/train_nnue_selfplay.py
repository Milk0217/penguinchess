"""
NNUE self-play training: Rust Alpha-Beta search + NNUE self-play.

Steps:
  1. Load NNUE model, create Rust AB search handle with weights
  2. For each game: Rust search selects moves, Python records features
  3. Train new NNUE on the generated data
  4. Repeat for multiple generations

Usage:
    uv run python examples/train_nnue_selfplay.py
    uv run python examples/train_nnue_selfplay.py --generations 3
    uv run python examples/train_nnue_selfplay.py --depth 4 --epochs 30
"""

import os, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed

from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE
from penguinchess.ai.nnue_train import NNUEDataset, collate_nnue
from penguinchess.ai.sparse_features import state_to_features
from penguinchess.rust_ffi import ffi_ab_create, get_engine


def _core_to_json(core: PenguinChessCore) -> str:
    """Serialize PenguinChessCore to Rust GameState JSON format."""
    smap = {'active': 'active', 'occupied': 'occupied', 'used': 'used', 'eliminated': 'eliminated'}
    hexes = [{
        'coord': {'q': h.q, 'r': h.r, 's': h.s},
        'state': smap.get(h.state, 'active'),
        'points': h.points if h.state == 'active' else 0,
    } for h in core.hexes]
    pieces = [{
        'id': p.id, 'alive': p.alive,
        'hex_idx': core.hexes.index(p.hex) if p.alive and p.hex else None,
        'hex_value': p.hex_value,
    } for p in core.pieces]
    return json.dumps({
        'board': {'cells': hexes, 'hex_map': {}, 'neighbors': []},
        'pieces': pieces, 'scores': core.players_scores,
        'phase': 'placement' if core.phase == 'placement' else 'movement',
        'current_player': core.current_player,
        'placement_count': core._placement_count,
        'episode_steps': core._episode_steps,
        'terminated': getattr(core, '_terminated', False),
        'last_action': None,
    })


def generate_selfplay_data(
    ab_handle,
    num_games: int,
    max_depth: int = 4,
    seed_offset: int = 0,
    game_workers: int = 4,
) -> list[dict]:
    """
    Generate training data using Rust Alpha-Beta search for move selection.
    Uses PenguinChessCore for game loop and feature recording.
    """
    def _play_game(seed: int) -> list[dict]:
        core = PenguinChessCore(seed=seed).reset(seed=seed)
        game_data = []

        while True:
            sparse, dense = state_to_features(core)
            player = core.current_player
            legal = core.get_legal_actions()
            if not legal:
                break

            if core.phase == 'placement':
                # Placement: one-step (highest immediate value)
                best_action = legal[0]
                best_score = -1e9
                for a in legal:
                    snap = core.get_snapshot()
                    _, reward, _, _ = core.step(a)
                    core.restore_snapshot(snap)
                    if reward > best_score:
                        best_score = reward
                        best_action = a
                action = best_action
            else:
                # Movement: Rust Alpha-Beta search
                state_json = _core_to_json(core)
                result = ab_handle.search(state_json, max_depth=max_depth)
                if 'error' in result:
                    action = legal[0]  # fallback
                else:
                    action = result.get('best_action', legal[0])

            game_data.append((sparse, dense, player))
            _, _, terminated, _ = core.step(action)
            if terminated or core.phase == 'gameover':
                break

        s1, s2 = core.players_scores
        if s1 > s2:
            outcome = 1
        elif s2 > s1:
            outcome = -1
        else:
            outcome = 0

        records = []
        for sparse, dense, player in game_data:
            value = outcome if player == 0 else -outcome
            records.append({
                'sparse': sparse,
                'dense': dense.tolist(),
                'value': value,
            })
        return records

    all_data = []
    with ThreadPoolExecutor(max_workers=game_workers) as pool:
        futs = [pool.submit(_play_game, seed_offset + i) for i in range(num_games)]
        for i, f in enumerate(as_completed(futs)):
            try:
                all_data.extend(f.result())
            except Exception as e:
                print(f"  Game {i} failed: {e}")
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{num_games}] {len(all_data)} positions")
    return all_data


def train_nnue_from_data(
    data: list[dict],
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = 'cpu',
    out_path: str = None,
) -> NNUE:
    """Train NNUE on generated data. Returns model with best validation loss."""
    split = int(len(data) * 0.8)
    train_ds = NNUEDataset(data[:split])
    val_ds = NNUEDataset(data[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_nnue)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_nnue)

    model = NNUE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = model.state_dict().copy()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for sb, db, vb, stm_b in train_loader:
            db, vb = db.to(device), vb.to(device)
            pred = model(sb, db, stm_players=stm_b)
            loss = criterion(pred, vb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sb, db, vb, stm_b in val_loader:
                db, vb = db.to(device), vb.to(device)
                pred = model(sb, db, stm_players=stm_b)
                val_loss += criterion(pred, vb).item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict().copy()

        if epoch % 10 == 9 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1:>3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  LR={lr_now:.1e}")

    model.load_state_dict(best_state)
    if out_path:
        torch.save({"model_state": best_state, "val_loss": best_val}, out_path)
        print(f"\n  Saved: {out_path} (val_loss={best_val:.4f})")
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NNUE self-play training (Rust AB search)")
    parser.add_argument("--resume", type=str, default="models/nnue/nnue_gen_1.pt",
                        help="Starting NNUE model (pass empty string for fresh)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start from fresh random weights (ignore --resume)")
    parser.add_argument("--games-per-gen", type=int, default=200)
    parser.add_argument("--depth", type=int, default=2,
                        help="Alpha-Beta search depth (placement=1 always, depth 2 = 5s/game, depth 1 = 0.4s/game)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="models/nnue")
    parser.add_argument("--game-workers", type=int, default=4)
    parser.add_argument("--tt-size", type=int, default=262144)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load starting model
    model = NNUE()
    if args.fresh:
        print("Fresh model (random weights)")
    elif args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            sd = torch.load(str(resume_path), map_location="cpu", weights_only=False)
            if "model_state" in sd:
                model.load_state_dict(sd["model_state"])
            else:
                model.load_state_dict(sd)
            print(f"Loaded: {resume_path}")
        else:
            print(f"Resume file not found: {resume_path}, starting fresh")

    # Detect starting generation
    start_gen = 0
    if not args.fresh and 'resume_path' in dir() and resume_path and resume_path.exists():
        stem = resume_path.stem
        if stem.startswith("nnue_gen_"):
            try:
                start_gen = int(stem.split("_")[-1]) + 1
            except ValueError:
                pass

    # Create Rust AB search handle and set weights
    print("Creating Rust Alpha-Beta search handle...")
    ab_handle = ffi_ab_create(json.dumps({
        'max_depth': args.depth, 'time_limit_ms': 0,
        'tt_size': args.tt_size, 'lmr_moves': 3, 'lmr_depth': 1,
    }))
    ok = ab_handle.set_weights(model.state_dict())
    print(f"Rust AB handle created, weights: {ok}")

    total_t0 = time.time()
    for gen in range(start_gen, start_gen + args.generations):
        print(f"\n{'='*50}")
        print(f"Generation {gen}")
        print(f"{'='*50}")

        print(f"  Self-play: {args.games_per_gen} games × Rust AB depth {args.depth} ...")
        t0 = time.time()
        data = generate_selfplay_data(
            ab_handle, args.games_per_gen, max_depth=args.depth,
            seed_offset=gen * 10000, game_workers=args.game_workers)
        gen_s = time.time() - t0
        print(f"  {len(data)} positions in {gen_s:.0f}s ({gen_s/args.games_per_gen:.1f}s/game)")

        # Train new model
        print(f"  Training {args.epochs} epochs...")
        out_path = str(out_dir / f"nnue_gen_{gen}.pt")
        new_model = train_nnue_from_data(
            data, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, device=device, out_path=out_path)

        # Update Rust search handle with new weights
        new_model = new_model.to("cpu")
        ok = ab_handle.set_weights(new_model.state_dict())
        print(f"  Updated weights: {ok}")
        model = new_model

        # Register
        try:
            from penguinchess.model_registry import register_model
            register_model(f"nnue_gen_{gen}", "nnue",
                           f"nnue/nnue_gen_{gen}.pt",
                           generation=gen, arch="nnue")
        except Exception as e:
            print(f"  Register: {e}")

    ab_handle.free()
    elapsed = time.time() - total_t0
    print(f"\nDone! {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
