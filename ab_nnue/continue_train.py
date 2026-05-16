"""Continue AB+NNUE multi-gen training from last generation.
Usage:
    python ab_nnue/continue_train.py --teacher models/nnue_trained/d6_gen3.pt --start-gen 4 --generations 3
"""
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
R_FT_DIM = 128
R_HD_DIM = 512
R_HD2_DIM = 256
R_INPUT_DIM = R_FT_DIM * 2 + 66

def teacher_weights(teacher_path):
    """Load and convert teacher to Rust-compatible 128/512 arch."""
    sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state", sd) if isinstance(sd, dict) and "model_state" in sd else sd
    ft_dim, hd = sd["ft.weight"].shape[0], sd["fc1.weight"].shape[0]
    if ft_dim == R_FT_DIM and hd == R_HD_DIM:
        m = NNUE(ft_dim=R_FT_DIM, hidden_dim=R_HD_DIM)
        m.load_state_dict(sd, strict=False)
        return {k: v.cpu() for k, v in m.state_dict().items()}
    print(f"  Padding teacher {ft_dim}/{hd} -> {R_FT_DIM}/{R_HD_DIM}", flush=True)
    ms = NNUE(ft_dim=ft_dim, hidden_dim=hd); ms.load_state_dict(sd, strict=False)
    mb = NNUE(ft_dim=R_FT_DIM, hidden_dim=R_HD_DIM)
    with torch.no_grad():
        mb.ft.weight.data[:ft_dim, :] = ms.ft.weight
        mb.ft.bias.data[:ft_dim] = ms.ft.bias
        si = ft_dim * 2 + 66
        mb.fc1.weight.data[:hd, :si] = ms.fc1.weight
        mb.fc1.bias.data[:hd] = ms.fc1.bias
        hd2 = ms.fc2.weight.shape[0]
        mb.fc2.weight.data[:hd2, :hd] = ms.fc2.weight
        mb.fc2.bias.data[:hd2] = ms.fc2.bias
        mb.fc3.weight.data[0, :hd2] = ms.fc3.weight[0]
        mb.fc3.bias.data[0] = ms.fc3.bias[0]
    return {k: v.cpu() for k, v in mb.state_dict().items()}

def gen_data(handle, tag, n_games, desc, worker_cb):
    p = str(OUT_DIR / f'{desc}_{tag}.bin')
    Path(p).unlink(missing_ok=True)
    t0 = time.time()
    cnt = worker_cb(handle, num_games=n_games, workers=8, output_path=p)
    print(f'  {desc}: {cnt} pos in {time.time()-t0:.0f}s ({time.time()-t0:.0f}s/{n_games:.0f}s/game)', flush=True)
    return p, cnt

def rust_train(flat, paths, tag):
    eng = get_engine()
    ptr = flat.ctypes.data_as(POINTER(c_float))
    tcfg = json.dumps({'lr':3e-4,'wd':1e-4,'batch_size':4096,'epochs':50})
    eng._lib.ffi_nnue_train(ptr, c_int32(len(flat)),
        c_char_p(','.join(paths).encode()), c_char_p(tcfg.encode()),
        create_string_buffer(4096), 4096)

def evaluate_model(model, n=100):
    import random as rnd
    from penguinchess.core import PenguinChessCore
    cfg = json.dumps({'max_depth':4,'tt_size':65536})
    h = ffi_ab_create(cfg)
    h.set_weights({k:v.cpu() for k,v in model.state_dict().items()})
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
                hx=[{'coord':{'q':x.q,'r':x.r,'s':x.s},'state':sm.get(x.state,'active'),
                     'points':x.points if x.state=='active' else 0} for x in c.hexes]
                px=[{'id':p.id,'alive':p.alive,
                     'hex_idx':c.hexes.index(p.hex) if p.alive and p.hex else None,
                     'hex_value':p.hex_value} for p in c.pieces]
                sj=json.dumps({'board':{'cells':hx},'pieces':px,'scores':c.players_scores,
                               'phase':'movement','current_player':c.current_player,
                               'placement_count':c._placement_count,
                               'episode_steps':c._episode_steps,'terminated':False})
                ac=h.search(sj,max_depth=4).get('best_action',0)
            else: ac=rnd.choice(leg)
            if ac in leg: c.step(ac)
            elif leg: c.step(leg[0])
        if c.players_scores[0]>c.players_scores[1]: w+=1
    del h
    return w/n

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--teacher',default='models/nnue_trained/d6_gen3.pt')
parser.add_argument('--start-gen',type=int,default=4)
parser.add_argument('--generations',type=int,default=3)
parser.add_argument('--games',type=int,default=300)
parser.add_argument('--random',type=int,default=150)
args=parser.parse_args()

OUT_DIR.mkdir(exist_ok=True)
best_wr=0.0
total_t0=time.time()

for gen in range(args.start_gen, args.start_gen+args.generations):
    tag=f'gen{gen}'
    print(f'\n{"="*50}\n  Generation {gen}\n{"="*50}', flush=True)
    t0=time.time()
    
    d=teacher_weights(args.teacher)
    cfg=json.dumps({'max_depth':6,'tt_size':65536,'lmr_moves':3,'lmr_depth':1,'nnue_order_depth':2,'null_move':True})
    handle=ffi_ab_create(cfg)
    handle.set_weights(d)
    
    sp_path,_=gen_data(handle,tag,args.games,'selfplay',ffi_ab_generate_selfplay_data)
    # Random data from gen_2 (diverse source, preventing collapse)
    d2 = teacher_weights('models/ab_nnue/nnue_gen_2.pt')
    h2 = ffi_ab_create(cfg)
    h2.set_weights(d2)
    r_path, _ = gen_data(h2, tag, args.random, 'random', ffi_ab_generate_random_data)
    del h2
    
    # Train from scratch
    fresh=NNUE(ft_dim=R_FT_DIM,hidden_dim=R_HD_DIM)
    w=fresh.state_dict()
    flat=np.concatenate([w['ft.weight'].T.ravel(),w['ft.bias'].ravel(),
        w['fc1.weight'].ravel(),w['fc1.bias'].ravel(),
        w['fc2.weight'].ravel(),w['fc2.bias'].ravel(),
        w['fc3.weight'].ravel(),w['fc3.bias'].ravel()]).astype(np.float32)
    rust_train(flat,[sp_path,r_path],tag)
    
    # Unflatten
    m=NNUE(ft_dim=R_FT_DIM,hidden_dim=R_HD_DIM)
    off=0
    m.ft.weight.data=torch.from_numpy(flat[off:off+360*R_FT_DIM].reshape(R_FT_DIM,360)); off+=360*R_FT_DIM
    m.ft.bias.data=torch.from_numpy(flat[off:off+R_FT_DIM]); off+=R_FT_DIM
    m.fc1.weight.data=torch.from_numpy(flat[off:off+R_HD_DIM*R_INPUT_DIM].reshape(R_HD_DIM,R_INPUT_DIM)); off+=R_HD_DIM*R_INPUT_DIM
    m.fc1.bias.data=torch.from_numpy(flat[off:off+R_HD_DIM]); off+=R_HD_DIM
    m.fc2.weight.data=torch.from_numpy(flat[off:off+R_HD2_DIM*R_HD_DIM].reshape(R_HD2_DIM,R_HD_DIM)); off+=R_HD2_DIM*R_HD_DIM
    m.fc2.bias.data=torch.from_numpy(flat[off:off+R_HD2_DIM]); off+=R_HD2_DIM
    m.fc3.weight.data=torch.from_numpy(flat[off:off+R_HD2_DIM].reshape(1,R_HD2_DIM)); off+=R_HD2_DIM
    m.fc3.bias.data=torch.from_numpy(flat[off:off+1])
    
    wr=evaluate_model(m,100)
    print(f'  vs Random: {100*wr:.0f}% (gen {gen})', flush=True)
    mp=OUT_DIR/f'd6_{tag}.pt'
    torch.save({'model_state':m.state_dict(),'win_rate':wr},str(mp))
    if wr>best_wr:
        best_wr=wr
        shutil.copy2(str(mp),str(OUT_DIR/'d6_best.pt'))
        print(f'  [NEW BEST] {100*wr:.0f}%', flush=True)
    if wr>0.7:
        shutil.copy2(str(mp),str(OUT_DIR/f'd6_{tag}_best.pt'))
    args.teacher=str(mp)
    print(f'  Time: {time.time()-t0:.0f}s', flush=True)

print(f'\nDone! {time.time()-total_t0:.0f}s  Best: {100*best_wr:.0f}%', flush=True)
