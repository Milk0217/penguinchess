"""Debug: print all hex coordinates in the board."""
import sys
sys.path.insert(0, '.')

from penguinchess.core import PenguinChessCore, create_board, generate_sequence
import random

def coord(h):
    return f"({h.q},{h.r},{h.s})"

# Create a board with seed=None (same as server)
rng = random.Random(None)
seq = generate_sequence(rng=rng)
hexes = create_board(seq)

print(f"Total hexes: {len(hexes)}")
print("\nAll hex coordinates (first 60):")
for i, h in enumerate(hexes[:60]):
    print(f"  idx={i}: {coord(h)} state={h.state} pts={h.points}")

# Build a map to check
hex_map = {}
for idx, h in enumerate(hexes):
    hex_map[(h.q, h.r, h.s)] = idx

# Check specific coordinates from the log
test_coords = [
    (4, -6, 2),   # hex=52 from log
    (4, -5, 1),   # hex=53 from log
    (3, -5, 2),   # hex=38 from log
    (2, -5, 3),   # hex=37 from log
    (2, -4, 2),   # hex=27 from log
    (4, -4, 0),   # hex=54 from log
]

print("\n\nLooking up coordinates from server log:")
for q, r, s in test_coords:
    idx = hex_map.get((q, r, s))
    print(f"  ({q},{r},{s}) -> idx={idx}")

# Check what hex 52 actually is
print(f"\nhexes[52] = {coord(hexes[52])}")
