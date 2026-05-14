import sys; sys.path.insert(0, '.')
import torch
from penguinchess.ai.nnue import NNUE

# Create new model (128-dim FT)
m = NNUE()
total = sum(p.numel() for p in m.parameters())
print(f'New NNUE: {total:,} params, FT_DIM={m.ft_dim}')

# Try loading old gen_2 (64-dim FT)
try:
    sd = torch.load('models/ab_nnue/nnue_gen_2.pt', map_location='cpu', weights_only=True)
    m2 = NNUE(ft_dim=64, hidden_dim=256)
    m2.load_state_dict(sd)
    flat = sum(p.numel() for p in m2.parameters())
    print(f'gen_2 loaded OK: {flat:,} params, FT_DIM=64')
except Exception as e:
    print(f'gen_2 load failed: {e}')
