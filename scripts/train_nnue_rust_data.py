"""Train NNUE on Rust-generated binary data."""
import sys, time, struct, numpy as np, torch
from torch.utils.data import DataLoader
import torch.nn as nn, torch.optim as optim
sys.path.insert(0, '.')
from penguinchess.ai.nnue import NNUE
from penguinchess.ai.nnue_train import NNUEDataset, collate_nnue
from penguinchess.rust_ffi import ffi_ab_create, ffi_ab_generate_random_data

def load_binary_data(path):
    """Load Rust-generated binary NNUE data (6i32 + 66f32 + 1f32 per record)."""
    with open(path, 'rb') as f:
        count = struct.unpack('Q', f.read(8))[0]
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    records = raw.reshape(-1, 292)
    # Reinterpret bytes: first 24 bytes = 6 int32, next 264 bytes = 66 float32, last 4 = 1 float32
    sparse = np.frombuffer(records[:, 0:24].tobytes(), dtype=np.int32).reshape(-1, 6)
    dense = np.frombuffer(records[:, 24:288].tobytes(), dtype=np.float32).reshape(-1, 66)
    values = np.frombuffer(records[:, 288:292].tobytes(), dtype=np.float32)
    return sparse, dense, values, count

def train(data_path, out_path, epochs=60):
    print(f'Loading {data_path}...', flush=True)
    sparse, dense, values, count = load_binary_data(data_path)
    print(f'Loaded {count} records', flush=True)

    all_data = []
    for i in range(count):
        sp = [int(x) for x in sparse[i] if x >= 0]
        all_data.append({'sparse': sp, 'dense': dense[i].tolist(), 'value': float(values[i])})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NNUE().to(device).float()
    split = int(len(all_data) * 0.8)
    tl = DataLoader(NNUEDataset(all_data[:split]), batch_size=8192, shuffle=True, collate_fn=collate_nnue)
    vl = DataLoader(NNUEDataset(all_data[split:]), batch_size=16384, shuffle=False, collate_fn=collate_nnue)
    opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    crit = nn.MSELoss()
    best_val = float('inf')
    best_state = model.state_dict().copy()

    for ep in range(epochs):
        model.train()
        tr = 0.0
        for sb, db, vb, stmb in tl:
            db, vb = db.to(device), vb.to(device)
            p = model(sb, db, stm_players=stmb)
            l = crit(p, vb); opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr += l.item()
        model.eval()
        vr = 0.0
        with torch.no_grad():
            for sb, db, vb, stmb in vl:
                db, vb = db.to(device), vb.to(device)
                vr += crit(model(sb, db, stm_players=stmb), vb).item()
        tr /= max(1, len(tl)); vr /= max(1, len(vl))
        sched.step(vr)
        if vr < best_val: best_val = vr; best_state = model.state_dict().copy()
        if ep % 10 == 0: print(f'  ep{ep+1:3d} trn={tr:.4f} val={vr:.4f}', flush=True)

    model.load_state_dict(best_state)
    torch.save({'model_state': model.state_dict(), 'val_loss': best_val}, out_path)
    print(f'Saved {out_path} (val_loss={best_val:.4f})', flush=True)
    return model

if __name__ == '__main__':
    train('data_nnue.bin', 'models/nnue/nnue_gen_rust.pt', epochs=60)
