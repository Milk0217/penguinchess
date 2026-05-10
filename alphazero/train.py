#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaZero self-play training — Rust-native MCTS + AVX2 inference.
Replaces Python eval callback with BN-folded Rust AZ model evaluation.

Key changes vs old train_alphazero.py:
  - Self-play uses mcts_search_rust_handle_az (no Python callback)
  - BN folding happens in Python before loading to Rust
  - After train iteration, weights re-exported to Rust handle

Usage:
    uv run python alphazero/train.py --iterations 50 --games 1000
    uv run python alphazero/train.py --resume models/alphazero/alphazero_resnet_best.pth
"""

import os, sys, math, time, copy, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim

from concurrent.futures import ThreadPoolExecutor, as_completed
from penguinchess.ai.alphazero_net import AlphaZeroNet, AlphaZeroResNet, AlphaZeroResNetLarge, AlphaZeroResNetXL, detect_net_arch
from penguinchess.ai.mcts_core import select_action
from penguinchess.rust_core import RustCore
from penguinchess.rust_ffi import get_engine, mcts_search_rust_handle_az, ffi_az_create, AZModelHandle, AZMCTSReuseTree
from penguinchess._compat import ensure_utf8_stdout
from penguinchess.training_status import update_status as _update_ts, clear_status as _clear_ts
ensure_utf8_stdout()

# ───── Config ────────────────────────────────────────────────

OUT_DIR = Path('models') / 'alphazero'

# ───── BN Folding ────────────────────────────────────────────

def fold_bn_to_linear(weight, bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm into Linear layer.
    Returns (folded_weight, folded_bias) as numpy arrays.
    weight: (out_features, in_features)
    bias: (out_features,)
    bn_*: (out_features,)
    """
    w = weight.cpu().numpy()
    b = bias.cpu().numpy()
    gw = bn_weight.cpu().numpy()
    gb = bn_bias.cpu().numpy()
    mean = bn_mean.cpu().numpy()
    var = bn_var.cpu().numpy()
    
    scale = gw / np.sqrt(var + eps)
    w_folded = w * scale[:, None]
    b_folded = (b - mean) * scale + gb
    return w_folded.astype(np.float32), b_folded.astype(np.float32)


def export_resnet_to_rust(state_dict):
    """Export AlphaZeroResNet weights to Rust format (BN-folded flat arrays).
    Handles both configurable (fc_in/res_blocks/fc_out) and legacy (fc1/bn1/fc2/...) architectures.
    """
    layer_info = []
    weights_list = []
    biases_list = []
    
    def add_layer(wk, bk, bnw, bnb, bnm, bnv, relu=True, res=False):
        w, b = fold_bn_to_linear(state_dict[wk], state_dict[bk], state_dict[bnw], state_dict[bnb], state_dict[bnm], state_dict[bnv])
        r, c = w.shape
        layer_info.append((r, c, relu, res))
        weights_list.append(w.ravel())
        biases_list.append(b)
    
    def add_linear(wk, bk, relu=False, res=False):
        w = state_dict[wk].cpu().numpy().astype(np.float32)
        b = state_dict[bk].cpu().numpy().astype(np.float32)
        layer_info.append((w.shape[0], w.shape[1], relu, res))
        weights_list.append(w.ravel())
        biases_list.append(b)
    
    # Detect architecture type from keys
    is_legacy = 'fc1.weight' in state_dict
    is_config = 'fc_in.weight' in state_dict
    
    if is_legacy:
        # Legacy AlphaZeroResNet: fc1/bn1 → fc2/bn2 → fc3/bn3
        add_layer('fc1.weight','fc1.bias','bn1.weight','bn1.bias','bn1.running_mean','bn1.running_var', relu=True)
        add_layer('fc2.weight','fc2.bias','bn2.weight','bn2.bias','bn2.running_mean','bn2.running_var', relu=True, res=True)
        add_layer('fc3.weight','fc3.bias','bn3.weight','bn3.bias','bn3.running_mean','bn3.running_var', relu=True)
    elif is_config:
        # Configurable: fc_in/bn_in → res_blocks.N.0/res_blocks.N.1 → ... → fc_out/bn_out
        add_layer('fc_in.weight','fc_in.bias','bn_in.weight','bn_in.bias','bn_in.running_mean','bn_in.running_var', relu=True)
        idx = 0
        while f'res_blocks.{idx}.0.weight' in state_dict:
            add_layer(f'res_blocks.{idx}.0.weight', f'res_blocks.{idx}.0.bias',
                      f'res_blocks.{idx}.1.weight', f'res_blocks.{idx}.1.bias',
                      f'res_blocks.{idx}.1.running_mean', f'res_blocks.{idx}.1.running_var', relu=True)
            add_layer(f'res_blocks.{idx}.3.weight', f'res_blocks.{idx}.3.bias',
                      f'res_blocks.{idx}.4.weight', f'res_blocks.{idx}.4.bias',
                      f'res_blocks.{idx}.4.running_mean', f'res_blocks.{idx}.4.running_var', relu=True, res=True)
            idx += 1
        add_layer('fc_out.weight','fc_out.bias','bn_out.weight','bn_out.bias','bn_out.running_mean','bn_out.running_var', relu=True)
    else:
        raise KeyError(f"Unknown ResNet architecture: {list(state_dict.keys())[:5]}")
    
    # Policy head
    policy_idx = len(layer_info)
    add_linear('policy_fc.weight', 'policy_fc.bias')
    
    # Value head
    value1_idx = len(layer_info)
    if 'value_fc1.weight' in state_dict:
        add_linear('value_fc1.weight', 'value_fc1.bias', relu=True)
        value2_idx = len(layer_info)
        add_linear('value_fc2.weight', 'value_fc2.bias')
    elif 'v_fc.weight' in state_dict:
        add_linear('v_fc.weight', 'v_fc.bias', relu=True)
        value2_idx = len(layer_info)
        add_linear('v_fc2.weight', 'v_fc2.bias')
    else:
        # Fallback: single value output
        value2_idx = value1_idx
    
    weights = np.concatenate(weights_list).astype(np.float32)
    biases = np.concatenate(biases_list).astype(np.float32)
    return layer_info, weights, biases, policy_idx, value1_idx, value2_idx


def export_mlp_to_rust(state_dict):
    """Export AlphaZeroNet (MLP) to Rust format."""
    layer_info = []
    weights_list = []
    biases_list = []
    
    def add_layer(wk, bk, bnw, bnb, bnm, bnv, relu=True):
        w_f, b_f = fold_bn_to_linear(state_dict[wk], state_dict[bk], state_dict[bnw], state_dict[bnb], state_dict[bnm], state_dict[bnv])
        r, c = w_f.shape
        layer_info.append((r, c, relu, False))
        weights_list.append(w_f.ravel())
        biases_list.append(b_f)
    
    def add_linear(wk, bk, relu=False):
        w = state_dict[wk].cpu().numpy().astype(np.float32)
        b = state_dict[bk].cpu().numpy().astype(np.float32)
        r, c = w.shape
        layer_info.append((r, c, relu, False))
        weights_list.append(w.ravel())
        biases_list.append(b)
    
    # MLP trunk
    add_layer('fc1.weight', 'fc1.bias', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var')
    add_layer('fc2.weight', 'fc2.bias', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var')
    
    policy_idx = len(layer_info)
    add_linear('policy_fc.weight', 'policy_fc.bias')
    
    value1_idx = len(layer_info)
    add_linear('v_fc.weight', 'v_fc.bias', relu=True)
    value2_idx = len(layer_info)
    add_linear('v_fc2.weight', 'v_fc2.bias')
    
    weights = np.concatenate(weights_list).astype(np.float32)
    biases = np.concatenate(biases_list).astype(np.float32)
    return layer_info, weights, biases, policy_idx, value1_idx, value2_idx


def create_az_handle(net, device='cpu'):
    """Create Rust AZ handle from a PyTorch AlphaZero network."""
    sd = {k: v.to(device) for k, v in net.state_dict().items()}
    
    if hasattr(net, 'arch_name') and net.arch_name in ('resnet', 'resnet_large', 'resnet_xl', 'resnet_large', 'resnet_xl'):
        layer_info, weights, biases, pi, v1, v2 = export_resnet_to_rust(sd)
        arch_str = net.arch_name.replace('resnet', 'resnet')
    else:
        layer_info, weights, biases, pi, v1, v2 = export_mlp_to_rust(sd)
        arch_str = 'mlp'
    
    return ffi_az_create(arch=arch_str, layer_info=layer_info, weights=weights, biases=biases, policy_idx=pi, value1_idx=v1, value2_idx=v2)


def update_az_weights(az_handle: AZModelHandle, net, device='cpu'):
    """Update Rust AZ handle weights from PyTorch net."""
    sd = {k: v.to(device) for k, v in net.state_dict().items()}
    
    if hasattr(net, 'arch_name') and 'resnet' in net.arch_name:
        _, weights, biases, _, _, _ = export_resnet_to_rust(sd)
    else:
        _, weights, biases, _, _, _ = export_mlp_to_rust(sd)
    
    lib = get_engine()._lib
    from ctypes import POINTER, c_float, c_int32
    w_ptr = weights.ctypes.data_as(POINTER(c_float))
    b_ptr = biases.ctypes.data_as(POINTER(c_float))
    lib.ffi_az_set_weights(c_int32(az_handle._handle), w_ptr, c_int32(len(weights)), b_ptr, c_int32(len(biases)))


# ───── Observation encoding ──────────────────────────────────

def _encode_flat_obs(core) -> np.ndarray:
    """Encode game state to 272-dim observation vector (206 + 66 dense features)."""
    import json
    s = json.loads(core.to_json())
    hexes = s.get('board', {}).get('cells', [])
    pieces = s.get('pieces', [])
    scores = s.get('scores', (0, 0))
    cp = s.get('current_player', 0)
    phase = s.get('phase', 'placement')
    
    flat = np.zeros(272, dtype=np.float32)
    for i, cell in enumerate(hexes[:60]):
        val = cell.get('points', 0) if cell.get('state') == 'active' else 0
        c = cell.get('coord', {})
        flat[i * 3] = c.get('q', 0) / 8.0
        flat[i * 3 + 1] = c.get('r', 0) / 8.0
        flat[i * 3 + 2] = val / 3.0
    
    for i, p in enumerate(pieces[:6]):
        base = 180 + i * 4
        if p.get('alive') and p.get('hex_idx') is not None:
            flat[base] = p.get('id', 0) / 10.0
            hi = p['hex_idx']
            if hi < len(hexes):
                hc = hexes[hi].get('coord', {})
                flat[base + 1] = hc.get('q', 0) / 8.0
                flat[base + 2] = hc.get('r', 0) / 8.0
                flat[base + 3] = hc.get('s', 0) / 8.0
        else:
            flat[base] = -1.0
    
    flat[204] = cp
    flat[205] = 1.0 if phase == 'movement' else 0.0
    
    # Dense features (206-271)
    for i, cell in enumerate(hexes[:60]):
        flat[206 + i] = cell.get('points', 0) / 3.0 if cell.get('state') == 'active' else 0.0
    flat[266] = scores[0] / 100.0
    flat[267] = scores[1] / 100.0
    flat[268] = 1.0 if phase == 'movement' else 0.0
    p1_alive = sum(1 for p in pieces[:6] if p.get('alive') and p.get('hex_idx') is not None and p.get('id', 0) in (4,6,8))
    p2_alive = sum(1 for p in pieces[:6] if p.get('alive') and p.get('hex_idx') is not None and p.get('id', 0) in (5,7,9))
    flat[269] = p1_alive / 3.0
    flat[270] = p2_alive / 3.0
    flat[271] = s.get('episode_steps', 0) / 500.0
    return flat


# ───── Self-play with Rust MCTS ──────────────────────────────

def self_play_game(
    az_handle,
    num_simulations: int = 200,
    temperature: float = 1.0,
    temp_threshold: int = 30,
    tree_reuse: bool = True,
    additional_sims: int = 20,
) -> list:
    """Self-play using Rust MCTS. With tree_reuse=True, tree is reused across moves."""
    import random
    engine = get_engine()
    seed = random.randint(0, 2**31 - 1)
    core = RustCore(engine=engine, seed=seed).reset(seed=seed)
    game_data = []

    # Build initial tree
    if tree_reuse:
        from penguinchess.rust_ffi import AZMCTSReuseTree
        mcts_tree = AZMCTSReuseTree(
            engine, core.handle, az_handle,
            num_simulations=num_simulations, c_puct=3.0,
            batch_size=min(1024, max(128, num_simulations // 2)),
        )

    for step in range(500):
        t = temperature if step < temp_threshold else 0.1

        if tree_reuse:
            counts = mcts_tree._raw
        else:
            raw = mcts_search_rust_handle_az(
                core.handle, az_handle._handle,
                num_simulations=num_simulations,
                c_puct=3.0,
                batch_size=min(1024, max(128, num_simulations // 2)),
            )
            counts = {int(k): v for k, v in raw.items()}

        if not counts:
            game_data = []
            if tree_reuse: mcts_tree.free()
            core.close()
            return game_data, 2

        total = sum(counts.values())
        policy = np.zeros(60, dtype=np.float32)
        if t > 0:
            for a, c in counts.items():
                policy[a] = c ** (1.0 / t)
        else:
            best_cnt = max(counts.values())
            for a, c in counts.items():
                policy[a] = 1.0 if c == best_cnt else 0.0
        s = policy.sum()
        if s > 0:
            policy /= s
        else:
            legal = core.get_legal_actions()
            if legal:
                policy[legal] = 1.0 / len(legal)
            else:
                game_data = []
                if tree_reuse: mcts_tree.free()
                core.close()
                return game_data, 2

        # Encode observation
        flat_obs = _encode_flat_obs(core)
        game_data.append((flat_obs, policy, core.current_player))

        action = select_action(counts, temperature=t)
        _, _, terminated, _ = core.step(action)

        # Tree reuse: move to child, add sims
        if tree_reuse and not terminated:
            mcts_tree.step(action, additional_sims=additional_sims, c_puct=3.0,
                           batch_size=min(1024, max(128, num_simulations // 2)))

        if terminated:
            break

    if tree_reuse: mcts_tree.free()
    if core.players_scores[0] > core.players_scores[1]: winner = 0
    elif core.players_scores[1] > core.players_scores[0]: winner = 1
    else: winner = 2

    result = []
    for flat_obs, policy, cp in game_data:
        if winner == 2:
            value = 0.0
        else:
            value = 1.0 if cp == winner else -1.0
        result.append((flat_obs, policy, value))

    core.close()
    return result, winner


# ───── Training ──────────────────────────────────────────────

def train_on_data(net, replay_buffer, batch_size=4096, epochs=15, lr=1e-3, device='cuda'):
    """Train network on replay buffer data (FIFO, keep last 150K)."""
    # FIFO truncation: keep only most recent positions
    MAX_BUFFER = 150000
    if len(replay_buffer) > MAX_BUFFER:
        replay_buffer[:] = replay_buffer[-MAX_BUFFER:]
    
    n = len(replay_buffer)
    if n < 100: return
    
    obs = np.zeros((n, 206), dtype=np.float32)
    policy = np.zeros((n, 60), dtype=np.float32)
    value = np.zeros(n, dtype=np.float32)
    for i, (o, p, v) in enumerate(replay_buffer):
        obs[i] = o; policy[i] = p; value[i] = v
    
    obs_t = torch.from_numpy(obs).to(device)
    policy_t = torch.from_numpy(policy).to(device)
    value_t = torch.from_numpy(value).to(device).unsqueeze(1)
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    best_state = net.state_dict().copy()
    
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            o, p, v = obs_t[idx], policy_t[idx], value_t[idx]
            
            logits, val = net(o)
            policy_loss = -torch.mean(torch.sum(p * F.log_softmax(logits, dim=1), dim=1))
            value_loss = F.mse_loss(val, v)
            loss = policy_loss + value_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / max(1, n // batch_size)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = net.state_dict().copy()
        
        if ep % 5 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'  ep {ep+1:>3d}/{epochs}  loss={avg_loss:.4f}  LR={lr_now:.1e}', flush=True)
    
    net.load_state_dict(best_state)


# ───── Main Pipeline ─────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='AlphaZero training (Rust MCTS)')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--games', type=int, default=500)
    parser.add_argument('--simulations', type=int, default=800)
    parser.add_argument('--games-per-iter', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = args.device
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize network
    if args.resume:
        sd = torch.load(args.resume, map_location='cpu', weights_only=True)
        NetClass = detect_net_arch(sd)
        net = NetClass().to(device)
        net.load_state_dict(sd)
        print(f'Loaded: {args.resume} ({NetClass.__name__})', flush=True)
    else:
        net = AlphaZeroResNet().to(device)
        print(f'Fresh: {net.__class__.__name__}', flush=True)
    
    net.train()
    
    # Create Rust AZ handle from PyTorch weights
    az_handle = create_az_handle(net, device='cpu')
    print(f'Rust AZ handle created', flush=True)
    
    replay_buffer = []
    best_wr = 0.0
    total_t0 = time.time()
    
    for i_iter in range(1, args.iterations + 1):
        t0 = time.time()
        print(f'\n{"="*50}', flush=True)
        print(f'Iteration {i_iter}/{args.iterations}', flush=True)
        print(f'{"="*50}', flush=True)
        
        # Self-play with Rust MCTS (parallel games)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        iter_data = []
        game_workers = args.workers
        
        def play_one_game(g):
            data, winner = self_play_game(
                az_handle, num_simulations=args.simulations,
                temperature=1.0, temp_threshold=30,
                tree_reuse=(args.simulations >= 200), additional_sims=20)
            return data if data else []
        
        with ThreadPoolExecutor(max_workers=game_workers) as pool:
            futures = [pool.submit(play_one_game, g) for g in range(args.games)]
            for i, f in enumerate(as_completed(futures)):
                data = f.result()
                if data:
                    iter_data.extend(data)
                if (i + 1) % 50 == 0:
                    print(f'  Games: {i+1}/{args.games} ({len(iter_data)} pos)', flush=True)
        
        replay_buffer.extend(iter_data)
        print(f'  {len(iter_data)} positions, buffer={len(replay_buffer)}', flush=True)
        
        # Train
        train_on_data(net, replay_buffer, batch_size=args.batch_size,
                      epochs=args.epochs, lr=args.lr, device=device)
        
        # Re-export weights to Rust handle
        update_az_weights(az_handle, net, device='cpu')
        
        # Save iteration model
        it_path = OUT_DIR / f'alphazero_{net.arch_name}_iter_{i_iter}.pth'
        torch.save(net.state_dict(), it_path)
        print(f'  Saved: {it_path}', flush=True)
        
        # Periodic evaluation
        if i_iter % args.eval_interval == 0:
            from penguinchess.rust_ffi import mcts_search_rust_handle_az
            a_wins = 0
            for ep in range(min(50, args.games)):
                core = RustCore(engine=get_engine()).reset(ep + 99999)
                while True:
                    legal = core.get_legal_actions()
                    if not legal: break
                    raw = mcts_search_rust_handle_az(core.handle, az_handle._handle, num_simulations=args.simulations, c_puct=3.0, batch_size=128)
                    counts = {int(k): v for k, v in raw.items()}
                    if not counts: break
                    action = max(counts, key=counts.__getitem__)
                    _, _, term, _ = core.step(action)
                    if term: break
                if core.players_scores[0] > core.players_scores[1]: a_wins += 1
                core.close()
            wr = a_wins / 50
            print(f'  vs Random (50 games): {a_wins}/50 ({wr*100:.0f}%)', flush=True)
            
            if wr > 0.55 and wr > best_wr:
                best_wr = wr
                best_path = OUT_DIR / f'alphazero_{net.arch_name}_best.pth'
                torch.save(net.state_dict(), best_path)
                print(f'  New best: {best_path}', flush=True)
        
        iter_s = time.time() - t0
        print(f'  Time: {iter_s:.0f}s', flush=True)
    
    # Save final
    final_path = OUT_DIR / f'alphazero_{net.arch_name}_final.pth'
    torch.save(net.state_dict(), final_path)
    print(f'\nDone! {time.time()-total_t0:.0f}s ({iter_s*args.iterations:.0f}s total)', flush=True)
    az_handle.free()


if __name__ == '__main__':
    main()
