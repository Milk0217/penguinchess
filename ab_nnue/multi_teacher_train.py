"""Multi-teacher AB+NNUE training: mix data from gen_2 (diverse) + strong teacher (sharp)."""
import sys, time, json, shutil
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from penguinchess.rust_ffi import (
    ffi_ab_create, ffi_ab_generate_selfplay_data, ffi_ab_generate_random_data,
    get_engine, POINTER, c_float, c_int32, c_char_p, create_string_buffer)
from penguinchess.ai.nnue import NNUE

OUT_DIR = Path('models/nnue_trained')
R_FD,R_HD,R_HD2 = 128,512,256
R_IDIM = R_FD*2+66

def pad_teacher(path):
    sd = torch.load(path,map_location='cpu',weights_only=False)
    sd = sd.get('model_state',sd) if isinstance(sd,dict) and 'model_state' in sd else sd
    fd,hd = sd['ft.weight'].shape[0],sd['fc1.weight'].shape[0]
    if fd==R_FD and hd==R_HD:
        m=NNUE(ft_dim=R_FD,hidden_dim=R_HD); m.load_state_dict(sd,strict=False)
        return {k:v.cpu() for k,v in m.state_dict().items()}
    ms=NNUE(ft_dim=fd,hidden_dim=hd); ms.load_state_dict(sd,strict=False)
    mb=NNUE(ft_dim=R_FD,hidden_dim=R_HD)
    with torch.no_grad():
        mb.ft.weight.data[:fd,:]=ms.ft.weight; mb.ft.bias.data[:fd]=ms.ft.bias
        si=fd*2+66; mb.fc1.weight.data[:hd,:si]=ms.fc1.weight; mb.fc1.bias.data[:hd]=ms.fc1.bias
        hd2=ms.fc2.weight.shape[0]; mb.fc2.weight.data[:hd2,:hd]=ms.fc2.weight; mb.fc2.bias.data[:hd2]=ms.fc2.bias
        mb.fc3.weight.data[0,:hd2]=ms.fc3.weight[0]; mb.fc3.bias.data[0]=ms.fc3.bias[0]
    return {k:v.cpu() for k,v in mb.state_dict().items()}

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--teacher',default='models/nnue_trained/d6_gen3.pt')
parser.add_argument('--teachers',nargs='+',default=['models/nnue_trained/d6_gen3.pt'],help='Multiple teacher paths for data mix')
parser.add_argument('--start-gen',type=int,default=7)
parser.add_argument('--generations',type=int,default=3)
parser.add_argument('--games-teacher',type=int,default=200,help='Self-play games per teacher')
parser.add_argument('--random',type=int,default=200)
parser.add_argument('--epochs',type=int,default=50)
args=parser.parse_args()
OUT_DIR.mkdir(exist_ok=True)
best_wr=0.0

def gen(handle,tag,n,desc):
    p=str(OUT_DIR/f'{desc}_{tag}.bin'); Path(p).unlink(missing_ok=True)
    t0=time.time()
    if desc=='selfplay': cnt=ffi_ab_generate_selfplay_data(handle,num_games=n,workers=8,output_path=p)
    else: cnt=ffi_ab_generate_random_data(handle,num_games=n,workers=8,output_path=p)
    return p,cnt

def train_on_paths(paths,tag):
    from penguinchess.rust_ffi import get_engine as ge
    eng=ge()
    fresh=NNUE(ft_dim=R_FD,hidden_dim=R_HD)
    w=fresh.state_dict()
    flat=np.concatenate([w['ft.weight'].T.ravel(),w['ft.bias'].ravel(),
        w['fc1.weight'].ravel(),w['fc1.bias'].ravel(),
        w['fc2.weight'].ravel(),w['fc2.bias'].ravel(),
        w['fc3.weight'].ravel(),w['fc3.bias'].ravel()]).astype(np.float32)
    ptr=flat.ctypes.data_as(POINTER(c_float))
    t0=time.time()
    eng._lib.ffi_nnue_train(ptr,c_int32(len(flat)),
        c_char_p(','.join(paths).encode()),
        c_char_p(json.dumps({'lr':3e-4,'wd':1e-4,'batch_size':4096,'epochs':args.epochs}).encode()),
        create_string_buffer(4096),4096)
    m=NNUE(ft_dim=R_FD,hidden_dim=R_HD); off=0
    m.ft.weight.data=torch.from_numpy(flat[off:off+360*R_FD].reshape(R_FD,360)); off+=360*R_FD
    m.ft.bias.data=torch.from_numpy(flat[off:off+R_FD]); off+=R_FD
    m.fc1.weight.data=torch.from_numpy(flat[off:off+R_HD*R_IDIM].reshape(R_HD,R_IDIM)); off+=R_HD*R_IDIM
    m.fc1.bias.data=torch.from_numpy(flat[off:off+R_HD]); off+=R_HD
    m.fc2.weight.data=torch.from_numpy(flat[off:off+R_HD2*R_HD].reshape(R_HD2,R_HD)); off+=R_HD2*R_HD
    m.fc2.bias.data=torch.from_numpy(flat[off:off+R_HD2]); off+=R_HD2
    m.fc3.weight.data=torch.from_numpy(flat[off:off+R_HD2].reshape(1,R_HD2)); off+=R_HD2
    m.fc3.bias.data=torch.from_numpy(flat[off:off+1])
    return m

def evaluate(m,n=100):
    import random as rnd
    from penguinchess.core import PenguinChessCore
    cfg=json.dumps({'max_depth':4,'tt_size':65536})
    h=ffi_ab_create(cfg); h.set_weights({k:v.cpu() for k,v in m.state_dict().items()})
    sm={'active':'active','occupied':'occupied','used':'used','eliminated':'eliminated'}
    w=0
    for g in range(n):
        c=PenguinChessCore(seed=g*9973).reset(seed=g*9973)
        for _ in range(6):
            leg=c.get_legal_actions()
            if leg: c.step(rnd.choice(leg))
        while not c._terminated and c._episode_steps<200:
            leg=c.get_legal_actions()
            if not leg: break
            if c.current_player==0:
                hx=[{'coord':{'q':x.q,'r':x.r,'s':x.s},'state':sm.get(x.state,'active'),'points':x.points if x.state=='active' else 0} for x in c.hexes]
                px=[{'id':p.id,'alive':p.alive,'hex_idx':c.hexes.index(p.hex) if p.alive and p.hex else None,'hex_value':p.hex_value} for p in c.pieces]
                sj=json.dumps({'board':{'cells':hx},'pieces':px,'scores':c.players_scores,'phase':'movement','current_player':c.current_player,'placement_count':c._placement_count,'episode_steps':c._episode_steps,'terminated':False})
                ac=h.search(sj,max_depth=4).get('best_action',0)
            else: ac=rnd.choice(leg)
            if ac in leg: c.step(ac)
            elif leg: c.step(leg[0])
        if c.players_scores[0]>c.players_scores[1]: w+=1
    del h; return w/n

total_t0=time.time()

for gen in range(args.start_gen,args.start_gen+args.generations):
    tag=f'gen{gen}'
    print(f'\n{"="*50}\n  Generation {gen}\n{"="*50}',flush=True)
    t0=time.time()
    
    cfg=json.dumps({'max_depth':6,'tt_size':65536,'lmr_moves':3,'lmr_depth':1,'nnue_order_depth':2,'null_move':True})
    all_paths=[]
    
    # Data from gen_2 (diverse, soft labels, always included)
    d2=pad_teacher('models/ab_nnue/nnue_gen_2.pt')
    h2=ffi_ab_create(cfg); h2.set_weights(d2)
    rp,_=gen(h2,tag,args.random,'random'); all_paths.append(rp)
    sp,_=gen(h2,tag,100,'selfplay_diverse'); all_paths.append(sp)
    del h2
    
    # Data from each teacher (sharp labels, strong play)
    for i,tp in enumerate(args.teachers):
        d=pad_teacher(tp)
        h=ffi_ab_create(cfg); h.set_weights(d)
        sp,_=gen(h,tag,args.games_teacher//len(args.teachers),f'selfplay_t{i}')
        all_paths.append(sp)
        del h
    
    print(f'  Training on {len(all_paths)} data files...',flush=True)
    m=train_on_paths(all_paths,tag)
    wr=evaluate(m,100)
    print(f'  vs Random: {100*wr:.0f}%',flush=True)
    
    mp=OUT_DIR/f'd6_{tag}.pt'
    torch.save({'model_state':m.state_dict(),'win_rate':wr},str(mp))
    if wr>best_wr:
        best_wr=wr; shutil.copy2(str(mp),str(OUT_DIR/'d6_best.pt'))
        print(f'  [NEW BEST] {100*wr:.0f}%',flush=True)
    if wr>best_wr*0.95: args.teachers=[str(mp)]
    print(f'  Time: {time.time()-t0:.0f}s',flush=True)

print(f'\nDone! {time.time()-total_t0:.0f}s  Best: {100*best_wr:.0f}%',flush=True)
