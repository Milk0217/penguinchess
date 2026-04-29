"""Random-play data + AB score labels."""
import json, random, torch, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import ffi_ab_create
from penguinchess.ai.sparse_features import state_to_features
from penguinchess.ai.nnue_train import NNUEDataset, collate_nnue

CFG = '{"max_depth":4,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2}'

def c2j(c):
    sm={'active':'active','occupied':'occupied','used':'used','eliminated':'eliminated'}
    h=[{'coord':{'q':x.q,'r':x.r,'s':x.s},'state':sm.get(x.state,'active'),'points':x.points if x.state=='active' else 0} for x in c.hexes]
    p=[{'id':x.id,'alive':x.alive,'hex_idx':c.hexes.index(x.hex) if x.alive and x.hex else None,'hex_value':x.hex_value} for x in c.pieces]
    return json.dumps({'board':{'cells':h,'hex_map':{},'neighbors':[]},'pieces':p,'scores':c.players_scores,'phase':'placement' if c.phase=='placement' else 'movement','current_player':c.current_player,'placement_count':c._placement_count,'episode_steps':c._episode_steps,'terminated':False,'last_action':None})

# Load model
m=NNUE(); sd=torch.load('models/nnue/nnue_gen_2.pt',map_location='cpu',weights_only=False)
sd=sd.get('model_state',sd) if isinstance(sd,dict) and 'model_state' in sd else sd
m.load_state_dict(sd)
handle=ffi_ab_create(CFG); handle.set_weights(m.state_dict())

# Generate from random games
N=500; all_data=[]; t0=time.time()
for g in range(N):
    core=PenguinChessCore(seed=g*7919+42).reset(seed=g*7919+42)
    for step in range(200):
        sparse,dense=state_to_features(core)
        player=core.current_player
        legal=core.get_legal_actions()
        if not legal: break
        sj=c2j(core)
        r=handle.search(sj,max_depth=4)
        all_data.append({'sparse':sparse,'dense':dense.tolist(),'value':r.get('score',0.0),'player':player})
        core.step(random.choice(legal))
        if core._terminated: break
    if (g+1)%100==0: print(f'  [{g+1}/{N}] pos={len(all_data)} t={time.time()-t0:.0f}s'); sys.stdout.flush()

handle.free()
print(f'Data: {len(all_data)} pos in {time.time()-t0:.0f}s')

# Train
device='cuda' if torch.cuda.is_available() else 'cpu'
model=NNUE().to(device).float()
split=int(len(all_data)*0.8)
tl=DataLoader(NNUEDataset(all_data[:split]),batch_size=8192,shuffle=True,collate_fn=collate_nnue)
vl=DataLoader(NNUEDataset(all_data[split:]),batch_size=16384,shuffle=False,collate_fn=collate_nnue)
opt=optim.Adam(model.parameters(),lr=5e-4,weight_decay=1e-4)
sched=optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.5,patience=5)
crit=nn.MSELoss()
best_val=float('inf'); best_state=model.state_dict().copy()
for ep in range(80):
    model.train(); tr=0
    for sb,db,vb,stmb in tl:
        db,vb=db.to(device),vb.to(device)
        p=model(sb,db,stm_players=stmb)
        l=crit(p,vb); opt.zero_grad(); l.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tr+=l.item()
    model.eval(); vr=0
    with torch.no_grad():
        for sb,db,vb,stmb in vl:
            db,vb=db.to(device),vb.to(device)
            vr+=crit(model(sb,db,stm_players=stmb),vb).item()
    tr/=max(1,len(tl)); vr/=max(1,len(vl)); sched.step(vr)
    if vr<best_val: best_val=vr; best_state=model.state_dict().copy()
    if ep%10==0: print(f'  ep{ep+1:3d} trn={tr:.4f} val={vr:.4f} lr={opt.param_groups[0]["lr"]:.1e}')

model.load_state_dict(best_state)
torch.save({'model_state':model.state_dict(),'val_loss':best_val},'models/nnue/nnue_gen_random.pt')
print(f'Saved (val_loss={best_val:.4f})')

# Eval
print('Eval 30 games...')
eh=ffi_ab_create(CFG); eh.set_weights(model.state_dict())
w=l=d=0
for g in range(30):
    core=PenguinChessCore(seed=g*9973+42).reset(seed=g*9973+42)
    for _ in range(6):
        leg=core.get_legal_actions()
        if leg: core.step(random.choice(leg))
    while not core._terminated and core._episode_steps<200:
        leg=core.get_legal_actions()
        if not leg: break
        if core.current_player==0:
            a=eh.search(c2j(core),max_depth=4).get('best_action',0)
        else:
            a=random.choice(leg)
        if a in leg: core.step(a)
        elif leg: core.step(leg[0])
    s1,s2=core.players_scores
    if s1>s2: w+=1
    elif s1==s2: d+=1
    else: l+=1
print(f'gen_random vs Random: W={w} D={d} L={l} WR={w/30*100:.0f}%')
eh.free()
