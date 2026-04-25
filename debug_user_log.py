"""Simulate the exact game from user log and check piece states."""
import sys
sys.path.insert(0, '.')

from penguinchess.core import PenguinChessCore

core = PenguinChessCore()
core.reset(seed=None)

# The actions from user log (hex indices)
actions = [52, 53, 38, 37, 27, 39]

print("=== Simulating Exact Game ===")
print("Actions: hex indices from user log")
print()

for step, action in enumerate(actions):
    player = core.current_player
    h = core.hexes[action]
    print(f"Step {step+1}: Player {player+1} places at hex {action} ({h.q},{h.r},{h.s})")

    obs, reward, terminated, info = core.step(action)

    print(f"  Pieces remaining: P1={sum(1 for p in core.pieces if p.alive and p.id%2==0)}/3, P2={sum(1 for p in core.pieces if p.alive and p.id%2==1)}/3")

    for p in core.pieces:
        status = "ALIVE" if p.alive else "DEAD"
        hex_info = f"({p.hex.q},{p.hex.r},{p.hex.s})" if p.hex else "NONE"
        if not p.alive or step >= 2:  # Show all after step 3
            print(f"    Piece {p.id}: {status} {hex_info}")
    print()

print("=== Final State ===")
for p in core.pieces:
    status = "ALIVE" if p.alive else "DEAD"
    hex_info = f"({p.hex.q},{p.hex.r},{p.hex.s})" if p.hex else "NONE"
    print(f"  Piece {p.id}: {status} at {hex_info}")

# User says final state should be:
# 4 dead, 5 alive, 6 dead, 7 alive, 8 alive, 9 alive
print()
print("=== Expected (user claims) ===")
print("Piece 4: DEAD")
print("Piece 5: ALIVE")
print("Piece 6: DEAD")
print("Piece 7: ALIVE")
print("Piece 8: ALIVE")
print("Piece 9: ALIVE")
