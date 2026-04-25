"""Debug the exact scenario from user's log.

Log shows:
#001: P1 placed hex=52 -> piece 4
#002: P2 placed hex=53 -> piece 5
#003: P1 placed hex=38 -> piece 6 (but piece 4 dies!)
#004: P2 placed hex=37 -> piece 7
#005: P1 placed hex=27 -> piece 8
#006: P2 placed hex=54 -> piece 9

But user says piece 6 ended up at piece 8's position (hex 27).
"""
import sys
sys.path.insert(0, '.')

from penguinchess.core import PenguinChessCore, create_board, generate_sequence
import random

# Create board with seed=None (same as server used)
rng = random.Random(None)
seq = generate_sequence(rng=rng)
hexes = create_board(seq)

print("Hex coordinates from create_board(seed=None):")
for idx in [52, 53, 38, 37, 27, 54]:
    h = hexes[idx]
    print(f"  idx={idx}: q={h.q}, r={h.r}, s={h.s}, state={h.state}, pts={h.points}")

# Check neighbors of hex 52 (where piece 4 should be placed)
print("\nNeighbors of idx=52 (where piece 4 is placed):")
h52 = hexes[52]
# Build hex_map
hex_map = {}
for i, h in enumerate(hexes):
    hex_map[(h.q, h.r, h.s)] = i

for dq, dr, ds in [(0,-1,1), (-1,0,1), (-1,1,0), (0,1,-1), (1,0,-1), (1,-1,0)]:
    key = (h52.q + dq, h52.r + dr, h52.s + ds)
    if key in hex_map:
        neighbor_idx = hex_map[key]
        nh = hexes[neighbor_idx]
        print(f"  idx={neighbor_idx}: q={nh.q}, r={nh.r}, s={nh.s}, state={nh.state}")

# Now simulate the placement
print("\n\n=== Simulating placement ===")
core = PenguinChessCore()
core.reset(seed=None)

# Track which piece is placed at each step
placement_sequence = [52, 53, 38, 37, 27, 54]

for step, hex_idx in enumerate(placement_sequence):
    h = core.hexes[hex_idx]
    print(f"\nStep {step+1}: Player {core.current_player+1} places at hex {hex_idx} ({h.q},{h.r},{h.s})")

    obs, reward, terminated, info = core.step(hex_idx)

    print(f"  After placement:")
    for p in core.pieces:
        if p.alive and p.hex:
            print(f"    Piece {p.id}: at ({p.hex.q},{p.hex.r},{p.hex.s})")
        elif p.alive:
            print(f"    Piece {p.id}: ALIVE but no hex!")

    # Check if any pieces died
    for p in core.pieces:
        if not p.alive:
            print(f"    Piece {p.id}: DIED")
