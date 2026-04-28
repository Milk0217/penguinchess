"""Debug: check piece states after placement."""
from penguinchess.core import PenguinChessCore

core = PenguinChessCore(seed=42)
core.reset(seed=42)

for step in range(6):
    legal = core.get_legal_actions()
    action = legal[0]
    _, reward, terminated, info = core.step(action)
    print(f'Step {step}: P{core.current_player} action={action} reward={reward} term={terminated} scores={core.players_scores}')
    if info.get('piece_eliminated'):
        print(f'  >> Piece eliminated!')
    if info.get('hexes_eliminated', 0) > 0:
        print(f'  >> {info["hexes_eliminated"]} hexes eliminated')

print(f'\nAfter 6 placements:')
print(f'  Phase: {core.phase}')
print(f'  Current player: {core.current_player}')
for i, p in enumerate(core.pieces):
    hex_idx = core.hexes.index(p.hex) if p.hex else -1
    print(f'  Piece[{i}] id={p.id} alive={p.alive} hex_idx={hex_idx}')
