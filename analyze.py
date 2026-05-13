"""Analyze current model and training state."""
import sys, os, glob, re, json, numpy as np
sys.path.insert(0, '.')
import torch
from penguinchess.ai.alphazero_net import AlphaZeroResNet2M

# 1. Model capacity
net = AlphaZeroResNet2M()
total = sum(p.numel() for p in net.parameters())
trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Total params: {total:,}")
print(f"Trainable: {trainable:,}")

val_params = sum(p.numel() for n,p in net.named_parameters() if 'value' in n)
pol_params = sum(p.numel() for n,p in net.named_parameters() if 'policy' in n)
print(f"Value head: {val_params:,} ({100*val_params/total:.1f}%)")
print(f"Policy head: {pol_params:,} ({100*pol_params/total:.1f}%)")

# 2. Recent model files
files = sorted(glob.glob('models/alphazero/alphazero_resnet_2m_iter_*.pth'),
               key=lambda p: int(re.search(r'iter_(\d+)', p).group(1)))
print(f"Saved models: {len(files)}")
for f in files[-10:]:
    sz = os.path.getsize(f)
    print(f"  {os.path.basename(f)} ({sz/1024:.0f}KB)")

# 3. Check value head of best model
best_path = 'models/alphazero/alphazero_resnet_2m_best.pth'
if os.path.exists(best_path):
    sd = torch.load(best_path, map_location='cpu', weights_only=False)
    # Check value range
    if 'v_fc2.weight' in sd:
        w = sd['v_fc2.weight']
        print(f"\nBest model value head: {w.shape}, mean={w.mean().item():.4f}, std={w.std().item():.4f}")
    elif 'value_fc2.weight' in sd:
        w = sd['value_fc2.weight']
        print(f"\nBest model value head: {w.shape}, mean={w.mean().item():.4f}, std={w.std().item():.4f}")

# 4. Check policy head
if os.path.exists(best_path):
    sd = torch.load(best_path, map_location='cpu', weights_only=False)
    pk = [k for k in sd if 'policy' in k]
    print(f"Policy keys: {pk}")
    if pk:
        w = sd[pk[0]]
        print(f"  {pk[0]}: {w.shape}, mean={w.mean().item():.4f}, std={w.std().item():.4f}")

# 5. Compare iter models value evolution
print(f"\nValue head evolution across iterations:")
for f in files[::5]:
    sd = torch.load(f, map_location='cpu', weights_only=False)
    # Find value fc weight key
    vk = next((k for k in sd if 'value' in k and 'weight' in k), None)
    if vk:
        w = sd[vk]
        print(f"  {os.path.basename(f)}: {vk} mean={w.mean().item():.4f} std={w.std().item():.4f}")
