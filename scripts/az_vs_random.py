"""Quick comparison: AZ raw vs MCTS vs random."""
import time
from penguinchess.eval_utils import compete, AlphaZeroAgent, AlphaZeroMCTSAgent
from penguinchess.ai.alphazero_net import detect_net_arch
import torch

t0 = time.time()

state = torch.load('models/alphazero/alphazero_resnet_xl_best.pth', map_location='cpu', weights_only=False)
NetClass = detect_net_arch(state)
net = NetClass()
net.load_state_dict(state)
net.eval()
print(f'Model: {NetClass.__name__} ({sum(p.numel() for p in net.parameters()):,} params)')

raw_agent = AlphaZeroAgent(net)
mcts_agent = AlphaZeroMCTSAgent(net, num_simulations=400, batch_size=32)

print('vs Random (10 games, 4 workers):')
res_raw = compete(raw_agent, None, 10, use_rust=True, seed_offset=0, game_workers=4)
print(f'  Raw:    win={res_raw["p1_win"]:.0%} draw={res_raw["draw"]:.0%}')

res_mcts = compete(mcts_agent, None, 10, use_rust=True, seed_offset=10, game_workers=4)
print(f'  MCTS:   win={res_mcts["p1_win"]:.0%} draw={res_mcts["draw"]:.0%}')

print(f'Time: {time.time()-t0:.0f}s')
