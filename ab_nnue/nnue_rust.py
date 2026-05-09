"""Train NNUE (gen_2 style) via Rust self-play data generation + PyTorch training.
Generates self-play data using Rust AB search, trains with PyTorch.
"""
import sys, time, json, random, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import ffi_ab_create, ffi_ab_generate_selfplay_data

RECORD_BYTES = 296  # 6×i32(24) + 66×f32(264) + 1×f32(4) + 1×i32(4)

def load_binary(path):
    with open(path,'rb') as f:
        n = int.from_bytes(f.read(8),'little')
        records = []
        for _ in range(n):
            raw = f.read(RECORD_BYTES)
            if len(raw) < RECORD_BYTES: break
            sp = [int.from_bytes(raw[i*4:(i+1)*4],'little') for i in range(6)]
            sp = [s for s in sp if 0 <= s < 360]
            de = np.frombuffer(raw[24:288], dtype=np.float32).copy()
            vl = float(np.frombuffer(raw[288:292], dtype=np.float32)[0])
            stm = int.from_bytes(raw[292:296], 'little')
            records.append({'sparse':sp,'dense':de,'value':vl,'stm':stm})
    return records

class NDataset(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r=self.d[i]; sp=torch.zeros(6,dtype=torch.long)
        for j,s in enumerate(r['sparse'][:6]): sp[j]=s
        return sp, torch.from_numpy(r['dense'].copy()), torch.tensor(r['value']), r['stm']

def train_model(model, records, epochs=100, batch_size=4096, lr=3e-4, device='cpu'):
    split = int(len(records)*0.8)
    train_loader = DataLoader(NDataset(records[:split]), batch_size, True, num_workers=0)
    valid_loader = DataLoader(NDataset(records[split:]), batch_size, False, num_workers=0)
    model = model.to(device).float().train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, 5)
    crit = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler(device) if device=='cuda' else None
    best_vl = float('inf'); best_sd = model.state_dict().copy()
    t0 = time.time()
    for ep in range(epochs):
        model.train(); tr=0; bt=0
        for sp,de,va,stm in train_loader:
            de,va=de.to(device,non_blocking=True),va.to(device,non_blocking=True)
            sp=sp.to(device,non_blocking=True)
            sl=[sp[i][sp[i]>=0].tolist() for i in range(sp.shape[0])]
            with torch.amp.autocast(device) if scaler else torch.no_grad().__enter__():
                pred=model.forward(sl,de,stm.tolist())
                loss=crit(pred.squeeze(),va.squeeze())
            if torch.isnan(loss) or torch.isinf(loss): continue
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
            else:
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tr+=loss.item(); bt+=1
        model.eval(); val_loss=0; vb=0
        with torch.no_grad():
            for sp,de,va,stm in valid_loader:
                de,va=de.to(device),va.to(device); sp=sp.to(device)
                sl=[sp[i][sp[i]>=0].tolist() for i in range(sp.shape[0])]
                loss=crit(model.forward(sl,de,stm.tolist()).squeeze(),va.squeeze())
                if not torch.isnan(loss): val_loss+=loss.item(); vb+=1
        tr/=max(1,bt); val_loss/=max(1,vb); sched.step(val_loss)
        if val_loss<best_vl: best_vl=val_loss; best_sd=model.state_dict().copy()
        if ep%10==0 or ep==epochs-1:
            lr_now=opt.param_groups[0]['lr']
            print(f'  ep {ep+1:>3d}/{epochs}  train={tr:.4f}  val={val_loss:.4f}  LR={lr_now:.1e}')
        elapsed = time.time()-t0
        eta = elapsed/(ep+1)*(epochs-ep-1)
        print(f'  elapsed: {elapsed/60:.1f}min  ETA: {eta/60:.1f}min')
    model.load_state_dict(best_sd)
    return model, best_vl

# ─── Main ──────────────────────────────────────────────────────

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
out_dir = Path('models/nnue_trained')
out_dir.mkdir(parents=True, exist_ok=True)

# Create/reuse AB handle
cfg = json.dumps({"max_depth":2,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2})
model = NNUE()
sd = {k:v.cpu() for k,v in model.state_dict().items()}
handle = ffi_ab_create(cfg)
handle.set_weights(sd)

# Generate self-play data
data_path = str(out_dir / 'selfplay_data.bin')
print(f'Generating self-play data (depth 2, 2000 games)...')
t0 = time.time()
count = ffi_ab_generate_selfplay_data(handle, num_games=2000, workers=4, output_path=data_path)
gen_s = time.time() - t0
print(f'  {count} positions in {gen_s:.0f}s ({count/(gen_s or 1):.0f} pos/s, {gen_s/2000:.1f}s/game)')
print(f'  ETA for 2000 games: {gen_s:.0f}s' if count > 0 else '  Data gen failed')

# Train
print(f'\nLoading data...')
t0 = time.time()
records = load_binary(data_path)
print(f'  {len(records)} records loaded in {time.time()-t0:.0f}s')

print(f'\nTraining {len(records)} positions...')
model, vloss = train_model(model, records, epochs=100, batch_size=4096, lr=3e-4, device=device)

# Save
torch.save({'model_state':model.state_dict(),'val_loss':vloss}, str(out_dir/'nnue_trained_best.pt'))
print(f'\nDone! Saved to {out_dir}/nnue_trained_best.pt (val_loss={vloss:.4f})')
