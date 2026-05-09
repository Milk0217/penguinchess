"""One-shot large NNUE training: generate data once, train once."""
import os, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, numpy as np
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import ffi_ab_create, ffi_ab_generate_random_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_dir = Path('models/nnue_large')
out_dir.mkdir(parents=True, exist_ok=True)

# Generate data: 2000 games at depth 2 (fast)
cfg = json.dumps({"max_depth":2,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2})
print('Creating initial model...')
model = NNUE()
sd = {k:v.cpu() for k,v in model.state_dict().items()}
handle = ffi_ab_create(cfg)
handle.set_weights(sd)

data_path = str(out_dir / 'train_data.bin')
print(f'Generating 2000 games at depth 2...')
t0 = time.time()
count = ffi_ab_generate_random_data(handle, num_games=2000, workers=4, output_path=data_path)
gen_s = time.time() - t0
print(f'  {count} positions in {gen_s:.0f}s ({gen_s/2000:.1f}s/game)')
del handle

# Load data
print('Loading data...')
t0 = time.time()
with open(data_path,'rb') as f:
    npos = int.from_bytes(f.read(8),'little')
    records = []
    for _ in range(npos):
        raw = f.read(296)
        if len(raw)<296: break
        sparse = [int.from_bytes(raw[i*4:(i+1)*4],'little') for i in range(6)]
        sparse = [s for s in sparse if 0<=s<360]
        dense = np.frombuffer(raw[24:288], dtype=np.float32).copy()
        outcome = float(np.frombuffer(raw[288:292], dtype=np.float32)[0])
        stm = int.from_bytes(raw[292:296],'little')
        records.append({'sparse':sparse,'dense':dense,'value':outcome,'stm':stm})
print(f'  {len(records)} records loaded in {time.time()-t0:.0f}s')

# Train
from torch.utils.data import Dataset, DataLoader
class N(Dataset):
    def __init__(self,d): self.d=d
    def __len__(self): return len(self.d)
    def __getitem__(self,i):
        r=self.d[i]; sp=torch.zeros(6,dtype=torch.long)
        for j,s in enumerate(r['sparse'][:6]): sp[j]=s
        return sp,torch.from_numpy(r['dense'].copy()),torch.tensor(r['value']),r['stm']

split = int(len(records)*0.8)
train_ds = N(records[:split]); val_ds = N(records[split:])
train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)

model = model.to(device).float().train()
opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
criterion = torch.nn.MSELoss()
amp = device=='cuda'
scaler = torch.amp.GradScaler(device) if amp else None

best_val = float('inf'); best_sd = model.state_dict().copy()
epochs = 50
for ep in range(epochs):
    model.train(); tr=0; bt=0
    for sp,de,va,stm in train_loader:
        de,va = de.to(device,non_blocking=True),va.to(device,non_blocking=True)
        sp = sp.to(device,non_blocking=True)
        sl = [sp[i][sp[i]>=0].tolist() for i in range(sp.shape[0])]
        stml = stm.tolist()
        with torch.amp.autocast(device) if amp else torch.no_grad().__enter__():
            pred = model.forward(sl, de, stml)
            loss = criterion(pred.squeeze(), va.squeeze())
        if torch.isnan(loss) or torch.isinf(loss): continue
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
        else:
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tr+=loss.item(); bt+=1
    model.eval(); vl=0; vb=0
    with torch.no_grad():
        for sp,de,va,stm in val_loader:
            de,va = de.to(device,non_blocking=True),va.to(device,non_blocking=True)
            sp=sp.to(device,non_blocking=True)
            sl=[sp[i][sp[i]>=0].tolist() for i in range(sp.shape[0])]
            pred=model.forward(sl,de,stm.tolist())
            loss=criterion(pred.squeeze(),va.squeeze())
            if not torch.isnan(loss): vl+=loss.item(); vb+=1
    tr/=max(1,bt); vl/=max(1,vb); sched.step(vl)
    if vl<best_val: best_val=vl; best_sd=model.state_dict().copy()
    if ep%5==0:
        print(f'  ep {ep+1:>3d}/{epochs} train={tr:.4f} val={vl:.4f} LR={opt.param_groups[0]["lr"]:.1e}')

model.load_state_dict(best_sd)
torch.save({'model_state':model.state_dict(),'val_loss':best_val}, str(out_dir/'nnue_large_best.pt'))
print(f'\nDone! Best val_loss: {best_val:.4f}')
