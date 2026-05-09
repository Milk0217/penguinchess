"""Fine-tune gen_2 with more self-play data. Rust data gen + PyTorch training."""
import sys, time, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from torch.utils.data import Dataset, DataLoader
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import ffi_ab_create, ffi_ab_generate_selfplay_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Load gen_2
model = NNUE()
sd = torch.load('models/nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
model.load_state_dict(sd, strict=False)
print(f'Loaded gen_2')

# Generate MORE self-play data at depth 4
cfg = json.dumps({"max_depth":4,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2})
handle = ffi_ab_create(cfg)
handle.set_weights({k:v.cpu() for k,v in model.state_dict().items()})

out_dir = Path('models/nnue_trained')
out_dir.mkdir(exist_ok=True)
data_path = str(out_dir / 'selfplay4.bin')

print(f'Generating 2000 self-play games at depth 4...')
t0 = time.time()
count = ffi_ab_generate_selfplay_data(handle, num_games=2000, workers=4, output_path=data_path)
gen_s = time.time() - t0
print(f'  {count} positions in {gen_s:.0f}s ({gen_s/2000:.1f}s/game)')

# Quick load + train
with open(data_path,'rb') as f:
    npos = int.from_bytes(f.read(8),'little')
    records = []
    for _ in range(npos):
        raw = f.read(296)
        if len(raw)<296: break
        sp = [int.from_bytes(raw[i*4:(i+1)*4],'little') for i in range(6)]
        sp = [s for s in sp if 0<=s<360]
        de = np.frombuffer(raw[24:288], dtype=np.float32).copy()
        vl = float(np.frombuffer(raw[288:292], dtype=np.float32)[0])
        stm = int.from_bytes(raw[292:296],'little')
        records.append({'sparse':sp,'dense':de,'value':vl,'stm':stm})

class DS(Dataset):
    def __init__(self,d): self.d=d
    def __len__(self): return len(self.d)
    def __getitem__(self,i):
        r=self.d[i]; sp=torch.zeros(6,dtype=torch.long)
        for j,s in enumerate(r['sparse'][:6]): sp[j]=s
        return sp,torch.from_numpy(r['dense'].copy()),torch.tensor(r['value']),r['stm']

n = len(records); split = int(n*0.8)
tl = DataLoader(DS(records[:split]), 4096, True, num_workers=0)
vl = DataLoader(DS(records[split:]), 4096, False, num_workers=0)

model = model.to(device).float().train()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
crit = torch.nn.MSELoss()
best_vl = float('inf'); best_sd = model.state_dict().copy()

for ep in range(30):
    model.train(); tr=0; bt=0
    for sp,de,va,stm in tl:
        de,va=de.to(device),va.to(device); sp=sp.to(device)
        sl=[sp[i][sp[i]>=0].tolist() for i in range(sp.shape[0])]
        pred=model.forward(sl,de,stm.tolist())
        loss=crit(pred.squeeze(),va.squeeze())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); tr+=loss.item(); bt+=1
    model.eval(); val=0; vb=0
    with torch.no_grad():
        for sp,de,va,stm in vl:
            de,va=de.to(device),va.to(device); sp=sp.to(device)
            sl=[sp[i][sp[i]>=0].tolist() for i in range(sp.shape[0])]
            l=crit(model.forward(sl,de,stm.tolist()).squeeze(),va.squeeze())
            if not torch.isnan(l): val+=l.item(); vb+=1
    val/=max(1,vb)
    if val<best_vl: best_vl=val; best_sd=model.state_dict().copy()
    print(f'  ep {ep+1:>2d}/30  loss={val:.4f}')

model.load_state_dict(best_sd)
torch.save({'model_state':model.state_dict(),'val_loss':best_vl},
           str(out_dir/'nnue_finetuned.pt'))
print(f'\nSaved! val_loss={best_vl:.4f}')
