import numpy as np, torch
from penguinchess.core import PenguinChessCore
from penguinchess.ai.sparse_features import state_to_features, extract_sparse, compute_sparse_diff
from penguinchess.ai.nnue import NNUE
from penguinchess.ai.nnue_agent import NNUEAgent

core = PenguinChessCore(seed=42); core.reset(seed=42)
for step in range(6):
  legal = core.get_legal_actions()
  if not legal: break
  core.step(legal[0])

n = sum(1 for p in core.pieces if p.alive and p.hex is not None)
print(f'Alive: {n}/6')

sparse, dense = state_to_features(core)
print(f'Sparse: {len(sparse)} Dense: {dense.shape}')
assert len(sparse) == n
assert dense.shape == (66,)

model = NNUE(); model.eval()
val = model.evaluate(core)
print(f'NNUE eval: {val:.4f}')
assert -1.0 <= val <= 1.0

acc = model.create_accumulator()
acc.reset(sparse, core.current_player)
assert acc.get_crelu().shape == (128,)

if core.get_legal_actions():
  old_s = extract_sparse(core)
  core.step(core.get_legal_actions()[0])
  new_s = extract_sparse(core)
  r, a = compute_sparse_diff(old_s, new_s)
  acc.apply_diff(r, a)
  acc2 = model.create_accumulator(); acc2.reset(new_s, core.current_player)
  assert np.allclose(acc.get_crelu(), acc2.get_crelu(), atol=1e-5)
  print('Incremental: OK')

agent = NNUEAgent(model, max_depth=2)
legal = core.get_legal_actions()
if legal:
  action = agent.select_action(core, legal)
  print(f'Alpha-Beta: action={action} nodes={agent._nodes_searched}')

print('ALL NNUE TESTS PASSED')
