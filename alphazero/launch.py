"""Launch AZ+MCTS training with all optimizations (50 iters, 500 games, 800 sims, 16 workers)."""
import sys; sys.path.insert(0, '.')
from pathlib import Path
import alphazero.train as t
import penguinchess.ai.alphazero_net as n

# Ensure data directory
Path('data').mkdir(exist_ok=True)

# Use 2M model with best existing config
t.AlphaZeroResNet = n.AlphaZeroResNet2M  
t.OBS_DIM = 272
t.arch_name = t.AlphaZeroResNet(obs_dim=272).arch_name

# Override argparse defaults
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='2m')
parser.add_argument('--iterations', type=int, default=50)
parser.add_argument('--games', type=int, default=500)
parser.add_argument('--simulations', type=int, default=800)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=4096)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--eval-interval', type=int, default=5)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--random-open-moves', type=int, default=10)
parser.add_argument('--temp-threshold', type=int, default=30)
parser.add_argument('--log-file', type=str, default='')
parser.add_argument('--auto-eval', action='store_true', default=False)
args, _ = parser.parse_known_args()

args.resume = 'models/alphazero/alphazero_resnet_2m_best.pth'

t.args = args
t.main()
