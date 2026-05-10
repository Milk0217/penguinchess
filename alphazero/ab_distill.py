"""Train AZ model on Rust-generated AB expert data (fast, no Python serialization)."""
import sys, time, json, torch, numpy as np, random as rnd, struct
from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
from penguinchess.ai.alphazero_net import AlphaZeroResNet, AlphaZeroResNetLarge
from penguinchess.rust_ffi import get_engine, ffi_ab_create, POINTER, c_float, c_int32, c_char_p, create_string_buffer
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
DEVICE='cuda'

REC_BYTES = 272 * 4 + 4 + 4 + 4
def load_az_data(path):
    raw=open(path,'rb').read()
    n=struct.unpack('<Q',raw[:8])[0]; off=8; recs=[]
    for _ in range(n):
        obs=np.frombuffer(raw[off:off+1088],dtype=np.float32,count=272).copy(); off+=1088
        act=struct.unpack('<i',raw[off:off+4])[0]; off+=4
        out=struct.unpack('<f',raw[off:off+4])[0]; off+=4
        stm=struct.unpack('<i',raw[off:off+4])[0]; off+=4
        recs.append((obs,act,out,stm))
    return recs

def train(data,net,ep=50,bs=4096,lr=3e-4):
    n=len(data); print(f'Train {n} pos {ep} ep',flush=True)
    obss=np.array([d[0] for d in data]); acts=np.array([d[1] for d in data],dtype=np.int64)
    vals=np.array([d[2] for d in data],dtype=np.float32)
    ot=torch.from_numpy(obss).float(); at=torch.from_numpy(acts).long()
    vt=torch.from_numpy(vals).float().unsqueeze(1)
    opt=optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4); net.train()
    for e in range(ep):
        perm=torch.randperm(n);tl=0.;tc=0
        for i in range(0,n,bs):
            idx=perm[i:i+bs]; o=ot[idx].to(DEVICE); a=at[idx].to(DEVICE); v=vt[idx].to(DEVICE)
            logits,val=net(o); pl=F.cross_entropy(logits,a); vl=F.mse_loss(val,v); loss=pl+vl
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
            tl+=loss.item(); tc+=1
        if e%10==0: print(f'  ep {e+1:>3d}/{ep}  loss={tl/tc:.4f}',flush=True)
    return net

def evaluate(net,n=50,sims=200):
    from penguinchess.rust_ffi import mcts_search_rust_handle_az,get_engine; from penguinchess.rust_core import RustCore
    from alphazero.train import create_az_handle
    import random as rnd
    az=create_az_handle(net,device='cpu'); w=d=l=0
    for g in range(n):
        core=RustCore(engine=get_engine()).reset(seed=g*9973+42)
        for _ in range(6):
            leg=core.get_legal_actions()
            if leg: core.step(rnd.choice(leg))
        while True:
            leg=core.get_legal_actions()
            if not leg: break
            raw=mcts_search_rust_handle_az(core.handle,az._handle,num_simulations=sims,c_puct=3.,batch_size=128)
            cnt={int(k):v for k,v in raw.items() if v>0 and not np.isnan(v)}
            if not cnt: break
            a=max(cnt,key=cnt.__getitem__); _,_,term,_=core.step(a)
            if term: break
        s0,s1=core.players_scores
        if s0>s1: w+=1
        elif s0==s1: d+=1
        else: l+=1
    az.free(); print(f'  vs Random: {w}/{n} ({100*w//n}%)',flush=True); return w/n

if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser()
    p.add_argument('--games',type=int,default=2000); p.add_argument('--ep',type=int,default=50)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--large',action='store_true')
    p.add_argument('--depth',type=int,default=6)
    args=p.parse_args()
    path=gen_az_data('models/ab_nnue/nnue_gen_2.pt',args.games,args.depth,workers=8)
    data=load_az_data(path); print(f'{len(data)} positions',flush=True)
    Net=AlphaZeroResNetLarge if args.large else AlphaZeroResNet
    net=Net().to(DEVICE); train(data,net,ep=args.ep,bs=4096,lr=args.lr)
    tag='large' if args.large else 'base'
    torch.save(net.state_dict(),f'models/alphazero/az_expert_{tag}.pth')
    evaluate(net,n=50,sims=200)

