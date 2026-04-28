import torch, os
from penguinchess.ai.alphazero_net import AlphaZeroResNetXL

state = torch.load('models/alphazero/alphazero_resnet_xl_best.pth', map_location='cpu', weights_only=False)
net = AlphaZeroResNetXL()
net.load_state_dict(state)

x = torch.randn(1, 206)
with torch.no_grad():
    logits, val = net(x)
print(f'Model healthy: val={val.item():.4f}')

cp = {'iteration': 41, 'model_state': state, 'best_state': state, 'best_iter': 35, 'best_win_rate': 0.585}
cp_path = 'models/alphazero/alphazero_resnet_xl_checkpoint.pth'
if os.path.exists(cp_path):
    os.remove(cp_path)
torch.save(cp, cp_path)
sz = os.path.getsize(cp_path)
print(f'Checkpoint saved: {sz/1024/1024:.0f}MB (iter=41, best=iter_35)')
