"""Fine-tune gen_2 with more self-play data. Pure Python data gen (fast NNUE forward)."""
import sys, time, json, random, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from torch.utils.data import Dataset, DataLoader
from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE
from penguinchess.ai.sparse_features import state_to_features
from concurrent.futures import ThreadPoolExecutor as TPE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Load gen_2
model = NNUE()
sd = torch.load('models/nnue/nnue_gen_2.pt', map_location='cpu', weights_only=False)
sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
model.load_state_dict(sd, strict=False)
model = model.to(device).float().eval()
print(f'Loaded gen_2')

CACHE_FILE = 'models/nnue_trained/nnue_data.npz'
if Path(CACHE_FILE).exists():
    print(f'Loading cached data from {CACHE_FILE}...')
    d = np.load(CACHE_FILE, allow_pickle=True)
    all_data = list(d['data'])
    print(f'  {len(all_data)} positions loaded')
else:
    # Generate self-play data
    def gen_game(seed):
        core = PenguinChessCore(seed=seed).reset(seed=seed)
        data = []
        while not core._terminated and core._episode_steps < 200:
            sp, de = state_to_features(core)
            stm = core.current_player
            legal = core.get_legal_actions()
            if not legal: break
            de_t = torch.from_numpy(de).unsqueeze(0).float().to(device)
            with torch.no_grad():
                val = model.forward([sp], de_t, [stm]).item()
            data.append({'sparse': sp, 'dense': de.copy(), 'value': val, 'stm': stm})
            if core.phase == 'placement':
                a = random.choice(legal)
            else:
                best_a, best_s = legal[0], -999
                for a_cand in legal:
                    snap = core.get_snapshot()
                    _, r, term, _ = core.step(a_cand)
                    if term:
                        score = r + (1.0 if core.players_scores[0] > core.players_scores[1] else 0.0)
                    else:
                        sp2, de2 = state_to_features(core)
                        de2_t = torch.from_numpy(de2).unsqueeze(0).float().to(device)
                        with torch.no_grad():
                            v2 = model.forward([sp2], de2_t, [1-stm]).item()
                        score = r - v2
                    core.restore_snapshot(snap)
                    if score > best_s: best_s, best_a = score, a_cand
                a = best_a
                if random.random() < 0.1: a = random.choice(legal)
            core.step(a) if a in legal else core.step(legal[0])
        s1, s2 = core.players_scores
        final = 1 if s1>s2 else (-1 if s2>s1 else 0)
        for d in data:
            d['value'] = final if d['stm'] == 0 else -final
        return data

    print(f'Generating 1000 self-play games...')
    t0 = time.time()
    all_data = []
    with TPE(max_workers=8) as pool:
        for i, result in enumerate(pool.map(gen_game, range(1000))):
            all_data.extend(result)
            if (i+1)%200 == 0:
                print(f'  [{i+1}/1000] {len(all_data)} pos, {time.time()-t0:.0f}s')
    print(f'  {len(all_data)} positions in {time.time()-t0:.0f}s')
    # Cache
    Path(CACHE_FILE).parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE_FILE, data=np.array(all_data, dtype=object))
    print(f'  Cached to {CACHE_FILE}')

# Train
n = len(all_data)
split = int(n*0.9)
train = all_data[:split]; valid = all_data[split:]
# Subsample to 50000 for faster training
import random as _rnd
_rnd.shuffle(train)
train = train[:50000]
print(f'\nTraining {len(train)} + {len(valid)} validation (subsampled)...')

class DS(Dataset):
    def __init__(self,d): self.d=d
    def __len__(self): return len(self.d)
    def __getitem__(self,i):
        r=self.d[i]; sp=torch.zeros(6,dtype=torch.long)
        for j,s in enumerate(r['sparse'][:6]): sp[j]=s if s<360 else 0
        return sp,torch.from_numpy(r['dense'].copy()).float(),torch.tensor(r['value'],dtype=torch.float32),r['stm']

tl = DataLoader(DS(train), 4096, True, num_workers=0)
vl = DataLoader(DS(valid), 4096, False, num_workers=0)
model2 = NNUE().to(device).float().train()
opt = torch.optim.Adam(model2.parameters(), lr=3e-4)
crit = torch.nn.MSELoss()
best_val = float('inf')

for ep in range(50):
    model2.train(); tr=0; bt=0
    for sp,de,va,stm in tl:
        de,va=de.to(device).float(),va.to(device).float(); sp=sp.to(device)
        sl=[sp[i][sp[i]>=0].tolist() for i in range(sp.shape[0])]
        stm_list = stm.tolist() if hasattr(stm, 'tolist') else list(stm)
        pred=model2.forward(sl,de,stm_list)
        loss=crit(pred.squeeze(),va.squeeze())
        if torch.isnan(loss) or torch.isinf(loss): continue
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model2.parameters(),1.0); opt.step()
        tr+=loss.item(); bt+=1
    model2.eval(); val=0; vb=0
    with torch.no_grad():
        for sp,de,va,stm in vl:
            de,va=de.to(device).float(),va.to(device).float(); sp=sp.to(device)
            sl=[sp[i][sp[i]>=0].tolist() for i in range(sp.shape[0])]
            stm_list = stm.tolist() if hasattr(stm, 'tolist') else list(stm)
            l=crit(model2.forward(sl,de,stm_list).squeeze(),va.squeeze())
            if not torch.isnan(l): val+=l.item(); vb+=1
    val/=max(1,vb)
    if val<best_val: best_val=val
    print(f'  ep {ep+1:>2d}/50  loss={val:.4f}')

torch.save({'model_state':model2.state_dict(),'val_loss':best_val}, 'models/nnue_trained/nnue_finetuned.pt')
print(f'\nSaved! val_loss={best_val:.4f}')
