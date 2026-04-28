"""Debug: compare AZ MCTS vs Raw vs Random."""
from penguinchess.eval_utils import compete, AlphaZeroMCTSAgent, AlphaZeroAgent
from penguinchess.ai.alphazero_net import AlphaZeroResNetXL
import torch

state = torch.load('models/alphazero/alphazero_resnet_xl_best.pth', map_location='cpu', weights_only=False)
net = AlphaZeroResNetXL()
net.load_state_dict(state)
net.eval()

# MCTS agent
mcts = AlphaZeroMCTSAgent(net, num_simulations=800, c_puct=3.0, batch_size=32)
res1 = compete(mcts, None, 2, use_rust=True, seed_offset=0, game_workers=2)
print(f'MCTS: win={res1["p1_win"]:.0%} draw={res1["draw"]:.0%}')

# Raw agent
raw = AlphaZeroAgent(net)
res2 = compete(raw, None, 2, use_rust=True, seed_offset=10, game_workers=2)
print(f'Raw:  win={res2["p1_win"]:.0%} draw={res2["draw"]:.0%}')
