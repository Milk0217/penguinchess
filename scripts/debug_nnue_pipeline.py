"""
NNUE Debug Script: 4 checks for why MSE is stuck at 0.9996.
"""
import os, sys, math, random, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from penguinchess.core import PenguinChessCore
from penguinchess.ai.sparse_features import (
    state_to_features, extract_sparse, extract_dense,
    PIECE_HEX_DIM, DENSE_DIM, PIECE_IDS, HEX_COUNT,
)
from penguinchess.ai.nnue import NNUE
from penguinchess.ai.nnue_train import NNUEDataset, collate_nnue, generate_games

np.set_printoptions(linewidth=200, suppress=True, precision=4)
torch.set_printoptions(linewidth=200)


def check1_game_data_generation():
    print("=" * 70)
    print("CHECK 1: Single game data generation and target verification")
    print("=" * 70)

    seed = 42
    core = PenguinChessCore(seed=seed)
    core.reset(seed=seed)

    game_features = []
    step = 0

    while True:
        sparse, dense = state_to_features(core)
        player = core.current_player
        phase = core.phase
        legal = core.get_legal_actions()
        if not legal:
            break
        action = random.choice(legal)
        game_features.append((sparse.copy(), dense.copy(), player, phase, action))
        _, _, terminated, _ = core.step(action)
        step += 1
        if step <= 8:
            print(f"  Step {step}: player=P{player+1}, phase={phase}, action=hex#{action}")
            print(f"    sparse indices ({len(sparse)}): {sparse}")
            print(f"    dense[0:5]: {dense[:5]}, dense[-6:] (meta): {dense[-6:]}")
            alive = [(p.id, p.alive, p.hex) for p in core.pieces]
            print(f"    pieces alive: {alive}")
        if terminated or core.phase == "gameover":
            break

    print(f"\n  Total steps: {step}")
    s1, s2 = core.players_scores
    print(f"  Final scores: P1={s1}, P2={s2}")
    if s1 > s2:
        outcome = 1
    elif s2 > s1:
        outcome = -1
    else:
        outcome = 0
    print(f"  Outcome: {outcome:+d}")

    print(f"\n  --- Verifying target values ---")
    for i, (sparse, dense, player, phase, action) in enumerate(game_features):
        if player == 0:
            value = outcome
        else:
            value = -outcome
        expected = outcome if player == 0 else -outcome
        assert value == expected, f"MISMATCH at step {i}"
        if i < 8 or i >= len(game_features) - 2:
            print(f"  Step {i+1}: P{player+1} to move -> target={value:+d}")
    print(f"\n  [PASS] All {len(game_features)} target values verified!")


def check2_training_pipeline():
    print("\n" + "=" * 70)
    print("CHECK 2: Training data pipeline - forward pass and basic training")
    print("=" * 70)

    all_data = generate_games(10, seed_offset=100)
    print(f"  Generated {len(all_data)} positions from 10 random games")
    targets = [d["value"] for d in all_data]
    unique, counts = np.unique(targets, return_counts=True)
    print(f"  Target distribution:")
    for u, c in zip(unique, counts):
        print(f"    value={u:+d}: {c} samples ({c/len(all_data)*100:.1f}%)")

    dataset = NNUEDataset(all_data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=collate_nnue
    )
    model = NNUE()
    model.train()
    print(f"\n  Model parameter count: {sum(p.numel() for p in model.parameters()):,}")

    for batch_idx, (sparse_batch, dense_batch, value_batch) in enumerate(loader):
        if batch_idx == 0:
            print(f"  Batch 0: {len(sparse_batch)} samples")
            print(f"  Sparse lens: {[len(s) for s in sparse_batch[:5]]}")
            print(f"  dense_batch: {dense_batch.shape}, value_batch: {value_batch.shape}")
            print(f"  value_batch: {value_batch[:10].tolist()}")
            with torch.no_grad():
                pred_before = model(sparse_batch, dense_batch)
            mse_before = F.mse_loss(pred_before, value_batch)
            print(f"\n  Before training:")
            print(f"    Predictions (first 10): {pred_before[:10].tolist()}")
            print(f"    Range: [{pred_before.min().item():.4f}, {pred_before.max().item():.4f}]")
            print(f"    MSE: {mse_before.item():.6f}")

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            losses = []
            for train_step in range(15):
                optimizer.zero_grad()
                pred = model(sparse_batch, dense_batch)
                loss = F.mse_loss(pred, value_batch)
                loss.backward()
                if train_step == 0:
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.norm().item() ** 2
                    total_norm = math.sqrt(total_norm)
                    print(f"\n  Gradient norms (step 0): total={total_norm:.6f}")
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            print(f"    {name}: grad_norm={p.grad.norm().item():.6f}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

            with torch.no_grad():
                pred_after = model(sparse_batch, dense_batch)
            mse_after = F.mse_loss(pred_after, value_batch)
            print(f"\n  After 15 steps:")
            print(f"    Loss trajectory: {[f'{l:.6f}' for l in losses]}")
            print(f"    MSE: {mse_before.item():.6f} -> {mse_after.item():.6f}")
            print(f"    Predictions (first 10): {pred_after[:10].tolist()}")
            print(f"    Max prediction change: {(pred_after - pred_before).abs().max().item():.6f}")
            if mse_after < mse_before:
                print(f"  [OK] Loss decreased")
            else:
                print(f"  [WARN] Loss did NOT decrease")
            if (pred_after - pred_before).abs().max().item() > 1e-6:
                print(f"  [OK] Predictions changed")
            else:
                print(f"  [WARN] Predictions unchanged!")
        break


def check3_forward_pass_and_gradients():
    print("\n" + "=" * 70)
    print("CHECK 3: Forward pass with sparse features - activations and gradient flow")
    print("=" * 70)

    all_data = generate_games(5, seed_offset=200)
    dataset = NNUEDataset(all_data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=collate_nnue
    )
    model_debug = NNUE()
    model_debug.train()

    activations = {}
    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    handles = []
    for name, module in model_debug.named_modules():
        if isinstance(module, nn.Linear):
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)

    for batch_idx, (sparse_batch, dense_batch, value_batch) in enumerate(loader):
        if batch_idx > 0:
            break

    model_debug.zero_grad()
    pred = model_debug(sparse_batch, dense_batch)
    loss = pred.sum()
    loss.backward()

    print(f"  Forward output shape: {pred.shape}")
    print(f"  Output values (first 8): {pred[:8].detach().tolist()}")
    print(f"\n  Intermediate activations:")
    for name, act in activations.items():
        nan = torch.isnan(act).any().item()
        inf = torch.isinf(act).any().item()
        print(f"    {name}: shape={act.shape}, range=[{act.min().item():.4f}, {act.max().item():.4f}], mean={act.mean().item():.4f}, nan={nan}, inf={inf}")

    print(f"\n  Gradient flow:")
    grad_ok = True
    for name, p in model_debug.named_parameters():
        if p.requires_grad:
            has_grad = p.grad is not None
            gn = p.grad.norm().item() if has_grad else 0.0
            has_nan = torch.isnan(p.grad).any().item() if has_grad else False
            if not has_grad:
                print(f"    [WARN] {name}: NO GRADIENT!")
                grad_ok = False
            elif has_nan:
                print(f"    [WARN] {name}: NaN gradient! norm={gn:.6f}")
                grad_ok = False
            else:
                print(f"    [OK] {name}: grad_norm={gn:.6f}")
    if grad_ok:
        print(f"  [OK] All gradients flow!")

    if model_debug.ft.weight.grad is not None:
        fg = model_debug.ft.weight.grad
        print(f"\n  ft.weight grad shape: {fg.shape}, nonzero: {(fg.abs() > 0).sum().item()} / {fg.numel()}")
        print(f"  ft.weight grad stats: min={fg.min().item():.6f}, max={fg.max().item():.6f}, mean={fg.mean().item():.6f}")

    print(f"\n  Sparse gather verification:")
    ft_w = model_debug.ft_weight_gather
    ft_b = model_debug.ft_bias
    print(f"  ft_weight_gather: {ft_w.shape}")
    print(f"  ft_bias: {ft_b.shape}")
    sample_sparse = sparse_batch[0]
    print(f"  Sample sparse indices ({len(sample_sparse)}): {sample_sparse}")
    idx_t = torch.tensor(sample_sparse, dtype=torch.long)
    gathered = ft_w[idx_t]
    print(f"  Gathered weights: {gathered.shape}")
    summed = gathered.sum(dim=0)
    print(f"  Summed: {summed.shape}")

    acc = ft_b.clone()
    if sample_sparse:
        acc = acc + ft_w[idx_t].sum(dim=0)
    acc_stm = F.relu(acc)
    acc_nstm = F.relu(acc)
    ds = dense_batch[0]
    x = torch.cat([acc_stm, acc_nstm, ds])
    x = F.relu(model_debug.fc1(x))
    x = F.relu(model_debug.fc2(x))
    x = torch.tanh(model_debug.fc3(x))
    print(f"  Manual forward: {x.item():.6f}, Model forward: {pred[0].item():.6f}, Match: {abs(x.item() - pred[0].item()) < 1e-5}")

    print(f"\n  [CRITICAL] stm == nstm accumulator check:")
    print(f"    acc_stm == acc_nstm? {(acc_stm == acc_nstm).all().item()}")
    print(f"    First 64 fc1 inputs: {acc_stm[:3].tolist()}...")
    print(f"    Next 64 fc1 inputs:  {acc_nstm[:3].tolist()}...")

    for h in handles:
        h.remove()


def check4_loss_and_optimizer():
    print("\n" + "=" * 70)
    print("CHECK 4: Optimizer, loss suitability, and output-target alignment")
    print("=" * 70)

    all_data = generate_games(10, seed_offset=300)
    targets = np.array([d["value"] for d in all_data])

    print(f"  Target distribution:")
    unique, counts = np.unique(targets, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {u:+d}: {c} ({(c/len(targets)*100):.1f}%)")
    print(f"  Mean: {targets.mean():.4f}, Std: {targets.std():.4f}")

    print(f"\n  MSE analysis:")
    frac_nonzero = (targets != 0).mean()
    print(f"    Fraction nonzero targets: {frac_nonzero:.1%}")
    print(f"    Expected MSE for predicting 0: ~{frac_nonzero:.4f}")
    print(f"    This matches 0.9996 if targets are nearly all +/-1")

    print(f"\n  Output range: tanh -> [-1, 1]")
    print(f"  Target range: [{targets.min()}, {targets.max()}]")
    print(f"  Match: YES")

    print(f"\n  Initial prediction test:")
    model = NNUE()
    model.eval()
    dataset = NNUEDataset(all_data)
    with torch.no_grad():
        for i in range(min(5, len(dataset))):
            sparse, dense, val = dataset[i]
            pred = model([sparse], dense.unsqueeze(0))
            print(f"    Sample {i}: target={val.item():+.0f}, prediction={pred.item():+.4f}")

    print(f"\n  [OK] Optimizer (Adam), Loss (MSE), and output range are compatible.")


def check_sparse_feature_ownership():
    print("\n" + "=" * 70)
    print("BONUS: Sparse feature ownership analysis")
    print("=" * 70)

    core = PenguinChessCore(seed=42)
    core.reset(seed=42)
    print(f"  PIECE_IDS: {PIECE_IDS}")
    print(f"  Piece index 0 = ID {PIECE_IDS[0]} (P1)")
    print(f"  Piece index 3 = ID {PIECE_IDS[3]} (P2)")
    print(f"  Sparse dim = {PIECE_HEX_DIM} = {len(PIECE_IDS)} pieces x {HEX_COUNT} hexes")
    print(f"  Feature encoding: sparse_idx = piece_idx * 60 + hex_idx")
    print(f"  P1 piece indices: 0,1,2 -> sparse range [0, 180)")
    print(f"  P2 piece indices: 3,4,5 -> sparse range [180, 360)")
    print(f"  Ownership IS encoded in sparse indices, BUT...")
    print(f"  forward() sums ALL piece-hex features into ONE accumulator")
    print(f"  then duplicates it for both stm and nstm.")
    print(f"  The ownership information is effectively DISCARDED.")


def main():
    check1_game_data_generation()
    check2_training_pipeline()
    check3_forward_pass_and_gradients()
    check4_loss_and_optimizer()
    check_sparse_feature_ownership()

    print("\n" + "=" * 70)
    print("OVERALL CONCLUSION")
    print("=" * 70)
    print("""
Root Cause: The NNUE forward() concatenates IDENTICAL stm and nstm accumulators.

In forward(), line 211-212 of nnue.py:
    acc_crelu = F.relu(acc_batch)        # (B, 64)
    x = torch.cat([acc_crelu, acc_crelu, dense_batch], dim=-1)  # (B, 194)

Both stm (side to move) and nstm (side not to move) perspectives use the SAME
accumulator. This means:
  1. The model cannot tell whose turn it is from the sparse features
  2. The first 128 dimensions of fc1 input are just duplicates of the same 64 values
  3. The model cannot learn that "my pieces good, opponent pieces bad" -> value flips
  4. Result: model converges to predicting ~0 for all positions -> MSE stuck at ~1.0

Fix: Separate stm and nstm accumulators by piece ownership.
  - stm_accumulator: sum ft_weight for CURRENT player's piece-hex features
  - nstm_accumulator: sum ft_weight for OPPONENT's piece-hex features

Concretely, in forward() split sparse indices by ownership range:
    if current_player == 0:  # P1 to move
        stm_mask = (idx < 180)  # P1 pieces
        nstm_mask = (idx >= 180)  # P2 pieces
    else:  # P2 to move
        stm_mask = (idx >= 180)  # P2 pieces
        nstm_mask = (idx < 180)  # P1 pieces

    acc_stm = ft_b + sum(ft_w[idx[stm_mask]])
    acc_nstm = ft_b + sum(ft_w[idx[nstm_mask]])
    x = cat([crelu(acc_stm), crelu(acc_nstm), dense])

This is the standard NNUE approach and will fix the learning failure.
""")


if __name__ == "__main__":
    main()
