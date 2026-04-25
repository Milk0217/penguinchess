import json
from penguinchess.core import create_board, generate_sequence, Q_ADJUSTMENTS

# Load JSON board coordinates
with open('backend_data/boards/default.json') as f:
    data = json.load(f)

print('=== JSON Board (RAW coords from file) ===')
print('First hex:', data['hexes'][0])
print('Last hex:', data['hexes'][-1])

# Generate Python default board
seq = generate_sequence()
hexes = create_board(seq)

print('\n=== Python create_board() (adjusted coords) ===')
print('First hex:', {'q': hexes[0].q, 'r': hexes[0].r, 's': hexes[0].s, '_q_raw': hexes[0]._q_raw})
print('Last hex:', {'q': hexes[-1].q, 'r': hexes[-1].r, 's': hexes[-1].s, '_q_raw': hexes[-1]._q_raw})

print('\n=== Q_ADJUSTMENTS mapping ===')
for k, v in Q_ADJUSTMENTS.items():
    print(f'  q={k}: adjustment={v}')

# Now show the transformation relationship
print('\n=== Transformation Analysis ===')
print('JSON first hex: q=-4, r=-4, s=8')
print('This means: raw_q = -4, raw_r = -4, raw_s = -raw_q - raw_r = 8')
print()
print('Python first hex: q=-2, r=6, s=-4 (stored), _q_raw=-4')
print('Where did q=-2 come from? r + adjustment(-4) = -4 + 2 = -2')
print('And s=-4 is the original raw_q value')
print()
print('So the JSON "s" field STORES the raw_q!')