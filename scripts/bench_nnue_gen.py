"""Compare NNUE generations vs Random"""
import json, random, torch
from penguinchess.core import PenguinChessCore
from penguinchess.ai.nnue import NNUE
from penguinchess.rust_ffi import ffi_ab_create

def c2j(c):
    sm={'active':'active','occupied':'occupied','used':'used','eliminated':'eliminated'}
    h=[{'coord':{'q':x.q,'r':x.r,'s':x.s},'state':sm.get(x.state,'active'),'points':x.points if x.state=='active' else 0} for x in c.hexes]
    p=[{'id':x.id,'alive':x.alive,'hex_idx':c.hexes.index(x.hex) if x.alive and x.hex else None,'hex_value':x.hex_value} for x in c.pieces]
    return json.dumps({'board':{'cells':h,'hex_map':{},'neighbors':[]},'pieces':p,'scores':c.players_scores,'phase':'placement' if c.phase=='placement' else 'movement','current_player':c.current_player,'placement_count':c._placement_count,'episode_steps':c._episode_steps,'terminated':False,'last_action':None})

CFG='{"max_depth":4,"tt_size":65536,"lmr_moves":3,"lmr_depth":1,"nnue_order_depth":2}'

for name,pt in [('gen_1','models/nnue/nnue_gen_1.pt'),('gen_2','models/nnue/nnue_gen_2.pt'),('gen_3','models/nnue/nnue_gen_3.pt')]:
    m=NNUE(); sd=torch.load(pt,map_location='cpu',weights_only=False)
    sd=sd.get('model_state',sd) if isinstance(sd,dict) and 'model_state' in sd else sd
    m.load_state_dict(sd)
    h=ffi_ab_create(CFG); h.set_weights(m.state_dict())
    w=l=d=0; N=50
    for g in range(N):
        core=PenguinChessCore(seed=g*9973+42).reset(seed=g*9973+42)
        for _ in range(6):
            leg=core.get_legal_actions()
            if leg: core.step(random.choice(leg))
        while not core._terminated and core._episode_steps<200:
            leg=core.get_legal_actions()
            if not leg: break
            if core.current_player==0:
                a=h.search(c2j(core),max_depth=4).get('best_action',0)
            else:
                a=random.choice(leg)
            if a in leg: core.step(a)
            elif leg: core.step(leg[0])
        s1,s2=core.players_scores
        if s1>s2: w+=1
        elif s1==s2: d+=1
        else: l+=1
    print(f'{name}: W={w} D={d} L={l} WR={w/N*100:.0f}%')
    h.free()
