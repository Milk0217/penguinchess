"""Train AZ model on Rust-generated AB expert data (fast, no Python serialization)."""
import sys, time, json, torch, numpy as np, random as rnd, struct
from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
from penguinchess.ai.alphazero_net import AlphaZeroResNet1M, AlphaZeroResNet3M, AlphaZeroResNet2M
...
    Net=AlphaZeroResNet3M if args.large else AlphaZeroResNet2M
    net=Net().to(DEVICE); train(data,net,ep=args.ep,bs=4096,lr=args.lr)
    tag='large' if args.large else 'base'
    torch.save(net.state_dict(),f'models/alphazero/az_expert_{tag}.pth')
    evaluate(net,n=50,sims=200)

