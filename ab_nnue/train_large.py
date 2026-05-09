"""Train large AB+NNUE model via Rust self-play data generation.
Uses ffi_ab_generate_random_data (random moves + AB labels) for initial data,
then trains the NNUE model. Iterative self-distillation.
"""
import os, sys, time, json, math, random as _rnd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from penguinchess.ai.nnue import NNUE, FT_DIM, HIDDEN_DIM
from penguinchess.rust_ffi import ffi_ab_create, ffi_ab_generate_random_data


def load_binary_data(path: str) -> list[dict]:
    """Load Rust-generated binary training data."""
    with open(path, 'rb') as f:
        count = int.from_bytes(f.read(8), 'little')
        records = []
        for _ in range(count):
            raw = f.read(292)  # 6×i32(24) + 66×f32(264) + 1×f32(4) = 292
            if len(raw) < 292: break
            sparse = [int.from_bytes(raw[i*4:(i+1)*4], 'little') for i in range(6)]
            sparse = [s for s in sparse if 0 <= s < 360]
            dense = np.frombuffer(raw[24:288], dtype=np.float32).copy()
            outcome = np.frombuffer(raw[288:292], dtype=np.float32)[0]
            records.append({'sparse': sparse, 'dense': dense, 'value': outcome})
    return records


def train_model(model, records, epochs=30, batch_size=4096, lr=1e-3, device='cpu'):
    """Train NNUE on records. Returns model."""
    from torch.utils.data import Dataset, DataLoader
    
    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, i):
            d = self.data[i]
            sp = torch.zeros(6, dtype=torch.long)
            for j, s in enumerate(d['sparse'][:6]): sp[j] = s
            return sp, torch.from_numpy(d['dense'].copy()), torch.tensor([d['value']])
    
    split = int(len(records) * 0.8)
    train_ds = SimpleDataset(records[:split])
    val_ds = SimpleDataset(records[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)
    
    model = model.to(device).float().train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    amp = device == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if amp else None
    
    best_val = float('inf')
    best_sd = model.state_dict().copy()
    
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        batches = 0
        for sp, de, va in train_loader:
            de, va = de.to(device, non_blocking=True), va.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
            sp_list = [sp[i][sp[i] >= 0].tolist() for i in range(sp.shape[0])]
            
            with torch.amp.autocast(device) if amp else torch.no_grad().__enter__():
                pred = model.forward(sp_list, de, [0]*len(sp_list))
                loss = criterion(pred.squeeze(), va.squeeze())
            
            if torch.isnan(loss) or torch.isinf(loss): continue
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tr_loss += loss.item()
            batches += 1
        
        model.eval()
        vl = 0.0
        vb = 0
        with torch.no_grad():
            for sp, de, va in val_loader:
                de, va = de.to(device, non_blocking=True), va.to(device, non_blocking=True)
                sp = sp.to(device, non_blocking=True)
                sp_list = [sp[i][sp[i] >= 0].tolist() for i in range(sp.shape[0])]
                pred = model.forward(sp_list, de, [0]*len(sp_list))
                loss = criterion(pred.squeeze(), va.squeeze())
                if not torch.isnan(loss):
                    vl += loss.item(); vb += 1
        
        tr_loss = tr_loss / max(1, batches)
        vl = vl / max(1, vb)
        scheduler.step(vl)
        
        if vl < best_val:
            best_val = vl
            best_sd = model.state_dict().copy()
        
        if ep % 5 == 0 or ep == epochs-1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'  ep {ep+1:>3d}/{epochs}  train={tr_loss:.4f}  val={vl:.4f}  LR={lr_now:.1e}')
    
    model.load_state_dict(best_sd)
    return model, best_val


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=50, help='Self-distillation iterations')
    parser.add_argument('--games', type=int, default=500, help='Games per iteration')
    parser.add_argument('--depth', type=int, default=4, help='AB search depth for labels')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--out-dir', type=str, default='models/nnue_large')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = NNUE()
    ckpt_path = out_dir / 'nnue_large_checkpoint.pt'
    start_iter = 0
    best_loss = float('inf')
    
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        start_iter = ckpt.get('iteration', 0)
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f'Resumed from iter {start_iter}, best_loss={best_loss:.4f}')
    else:
        print(f'Starting from scratch')
    
    data_path = str(out_dir / 'temp_data.bin')
    
    for it in range(start_iter, args.iters):
        print(f'\n{"="*50}')
        print(f'Iteration {it+1}/{args.iters}')
        print(f'{"="*50}')
        
        # Generate data using AB search with current weights
        # Use ffi_ab_generate_random_data (random moves + AB labels)
        cfg = json.dumps({"max_depth": args.depth, "tt_size": 65536,
                          "lmr_moves": 3, "lmr_depth": 1, "nnue_order_depth": 2})
        handle = ffi_ab_create(cfg)
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        handle.set_weights(sd)
        
        t0 = time.time()
        count = ffi_ab_generate_random_data(handle, num_games=args.games,
            workers=args.workers, output_path=data_path)
        gen_s = time.time() - t0
        print(f'  Data: {count} positions in {gen_s:.0f}s ({gen_s/args.games:.1f}s/game)')
        # handle freed implicitly on garbage collection
        
        if count < 100:
            print(f'  ⚠ Too few positions, skipping')
            continue
        
        # Load and train
        records = load_binary_data(data_path)
        print(f'  Loaded {len(records)} records')
        print(f'  Training {args.epochs} epochs...')
        t0 = time.time()
        model, val_loss = train_model(model, records, epochs=args.epochs,
                                       batch_size=args.batch_size, lr=args.lr, device=device)
        train_s = time.time() - t0
        print(f'  val_loss={val_loss:.4f} ({train_s:.0f}s)')
        
        # Save
        iter_path = out_dir / f'nnue_large_iter_{it+1}.pt'
        torch.save({'model_state': model.state_dict(), 'val_loss': val_loss}, str(iter_path))
        print(f'  Saved: {iter_path.name}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'model_state': model.state_dict(), 'val_loss': val_loss},
                       str(out_dir / 'nnue_large_best.pt'))
            print(f'  [NEW BEST] val_loss={val_loss:.4f}')
        
        # Checkpoint (every 5 iters)
        if (it + 1) % 5 == 0 or it == args.iters - 1:
            torch.save({
                'model_state': model.state_dict(),
                'best_loss': best_loss,
                'iteration': it + 1,
            }, str(ckpt_path))
            print(f'  Checkpoint saved')
    
    print(f'\nDone! Best val_loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
