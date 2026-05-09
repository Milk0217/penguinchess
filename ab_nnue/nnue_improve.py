"""Train NNUE using Rust AB self-play data (depth 4). More data > gen_2."""
import sys, time, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from torch.utils.data import Dataset, DataLoader
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import ffi_ab_create, ffi_ab_generate_selfplay_data

RECORD_BYTES = 296
def load_binary(path):
    with open(path,'rb') as f:
        n = int.from_bytes(f.read(8),'little')
        recs = []
        for _ in range(n):
            raw = f.read(RECORD_BYTES)
            if len(raw) < RECORD_BYTES: break
            sp = [int.from_bytes(raw[i*4:(i+1)*4],'little') for i in range(6)]
            sp = [s for s in sp if 0 <= s < 360]
            de = np.frombuffer(raw[24:288], dtype=np.float32).copy()
            vl = float(np.frombuffer(raw[288:292], dtype=np.float32)[0])
            stm = int.from_bytes(raw[292:296], 'little')
            recs.append({'sparse':sp,'dense':de,'value':vl,'stm':stm})
    return recs

class DS(Dataset):
    def __init__(self,d): self.d=d
    def __len__(self): return len(self.d)
    def __getitem__(self,i):
        r=self.d[i]; sp=torch.zeros(6,dtype=torch.long)
        for j,s in enumerate(r['sparse'][:6]): sp[j]=s if s<360 else 0
        return sp,torch.from_numpy(r['dense'].copy()).float(),torch.tensor(r['value'],dtype=torch.float32),r['stm']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_dir = Path('models/nnue_trained')
out_dir.mkdir(exist_ok=True)

# Load gen_2 as starting point
model = NNUE()
sd = torch.load('models/nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
model.load_state_dict(sd, strict=False)
print(f'gen_2 loaded')

# Rust self-play data gen (depth 4 for quality labels)
cfg = json.dumps({"max_depth":4,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2})
handle = ffi_ab_create(cfg)
handle.set_weights({k:v.cpu() for k,v in model.state_dict().items()})
data_path = str(out_dir / 'selfplay_d4.bin')

t0 = time.time()
count = ffi_ab_generate_selfplay_data(handle, num_games=2000, workers=4, output_path=data_path)
gen_s = time.time() - t0
print(f'Data: {count} pos in {gen_s:.0f}s')
print(f'  {gen_s/2000:.1f}s/game')

records = load_binary(data_path)
n = len(records); split = int(n*0.8)
tl = DataLoader(DS(records[:split]), 4096, True, num_workers=0)
vl = DataLoader(DS(records[split:]), 4096, False, num_workers=0)

model = model.to(device).float().train()
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
crit = torch.nn.MSELoss()
best_vl = float('inf'); best_sd = model.state_dict().copy()
t1 = time.time()

for ep in range(30):
    model.train(); tr=0; bt=0
    for sp,de,va,stm in tl:
        de,va=de.to(device).float(),va.to(device).float(); sp=sp.to(device)
        sl=[sp[i][sp[i]>=0].tolist() for i in range(sp.shape[0])]
        stm_l = stm.tolist() if hasattr(stm,'tolist') else list(stm)
        pred=model.forward(sl,de,stm_l)
        loss=crit(pred.squeeze(),va.squeeze())
        if torch.isnan(loss) or torch.isinf(loss): continue
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tr+=loss.item(); bt+=1
    model.eval(); val=0; vb=0
    with torch.no_grad():
        for sp,de,va,stm in vl:
            de,va=de.to(device).float(),va.to(device).float(); sp=sp.to(device)
            sl=[sp[i][sp[i]>=0].tolist() for i in range(sp.shape[0])]
            stm_l = stm.tolist() if hasattr(stm,'tolist') else list(stm)
            l=crit(model.forward(sl,de,stm_l).squeeze(),va.squeeze())
            if not torch.isnan(l): val+=l.item(); vb+=1
    val/=max(1,vb)
    if val<best_vl: best_vl=val; best_sd=model.state_dict().copy()
    print(f'  ep{ep+1:>2d}  loss={val:.4f}  ({time.time()-t1:.0f}s)')

model.load_state_dict(best_sd)
torch.save({'model_state':model.state_dict(),'val_loss':best_vl},
           str(out_dir/'nnue_improved.pt'))
print(f'\nSaved {out_dir}/nnue_improved.pt  val_loss={best_vl:.4f}')
print(f'Total: {time.time()-t0:.0f}s ({time.time()-t0:.0f}s)')
