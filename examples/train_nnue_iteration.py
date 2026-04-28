"""
One-shot NNUE training: generate MCTS game data → train NNUE.

Usage:
    uv run python examples/train_nnue_iteration.py
    uv run python examples/train_nnue_iteration.py --games 500 --epochs 50 --batch-size 256
"""

import os, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE
from penguinchess.ai.nnue_train import NNUEDataset, collate_nnue, generate_games_mcts
from penguinchess.ai.sparse_features import state_to_features
from penguinchess.eval_utils import AZMCTSWrapper, compete, RandomAgent


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NNUE one-shot training")
    parser.add_argument("--games", type=int, default=300, help="MCTS games to generate")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per move")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="models/nnue/nnue_gen_1.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("NNUE Training — Generation 1")
    print("=" * 50)
    print(f"  Device: {device}")
    print(f"  Games:  {args.games} × {args.sims} sims")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}")
    print()

    # 1. Load AZ ResNet model
    print("[1/4] Loading AZ ResNet...")
    from penguinchess.ai.alphazero_net import detect_net_arch
    az_path = "models/alphazero/alphazero_resnet_best.pth"
    state = torch.load(az_path, map_location="cpu", weights_only=False)
    NetClass = detect_net_arch(state)
    net = NetClass()
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    print(f"  Loaded {NetClass.__name__} ({sum(p.numel() for p in net.parameters()):,} params)")

    # 2. Generate MCTS training data using Rust MCTS (fast)
    print(f"\n[2/4] Generating {args.games} MCTS games ({args.sims} sims x Rust)...")
    t0 = time.time()
    from penguinchess.eval_utils import AlphaZeroMCTSAgent
    wrapper = AlphaZeroMCTSAgent(net, num_simulations=args.sims, c_puct=1.4, batch_size=32)
    data = generate_games_mcts(wrapper, args.games, seed_offset=0, use_rust=True)
    print(f"  Generated {len(data)} positions in {time.time()-t0:.0f}s")

    # 3. Train NNUE
    print(f"\n[3/4] Training NNUE ({args.epochs} epochs)...")
    split = int(len(data) * 0.8)
    train_ds = NNUEDataset(data[:split])
    val_ds = NNUEDataset(data[split:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_nnue)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_nnue)

    model = NNUE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
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
            best_state = model.state_dict()

        if epoch % 10 == 9 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1:>3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  LR={lr_now:.1e}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"model_state": model.state_dict(), "val_loss": best_val}, str(out_path))
    print(f"\n  Model saved: {out_path}  (val_loss={best_val:.4f})")

    # 4. Quick evaluation vs random
    print(f"\n[4/4] Quick eval vs Random (20 games)...")
    from penguinchess.ai.nnue_agent import NNUEAgent
    model.cpu()
    agent = NNUEAgent(model, max_depth=4)
    result = compete(agent, None, 20, use_rust=True, seed_offset=100, game_workers=4)
    print(f"  NNUE vs Random: {result['p1_win']:.0%} win, {result['draw']:.0%} draw")

    # Register
    try:
        from penguinchess.model_registry import register_model
        register_model("nnue_gen_1", "nnue", f"nnue/{out_path.name}",
                       generation=1, arch="nnue", elo=1200)
    except Exception as e:
        print(f"  Register: {e}")

    print(f"\nDone! Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
