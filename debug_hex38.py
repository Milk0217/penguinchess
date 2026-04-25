"""Debug the specific test case: seed with pieces 4,5,6,7,8,9 at specific hexes."""
import sys
sys.path.insert(0, '.')

from penguinchess.core import PenguinChessCore

def coord(h):
    return f"({h.q},{h.r},{h.s})" if h else "DEAD"

def hex_info(core, idx):
    """Get hex info by index."""
    h = core.hexes[idx]
    return f"idx={idx} {coord(h)} state={h.state} pts={h.points}"

def trace_game(seed, actions):
    """Play through a game with specific actions."""
    core = PenguinChessCore()
    core.reset(seed=seed)

    print(f"\n{'='*60}")
    print(f"SEED {seed}, ACTIONS {actions}")
    print(f"{'='*60}")

    for step, action in enumerate(actions):
        legal = core.get_legal_actions()
        print(f"\nStep {step+1}: Player {core.current_player+1}")
        print(f"  Phase: {core.phase}")
        print(f"  Action: {action} -> {hex_info(core, action)}")

        obs, reward, terminated, info = core.step(action)

        print(f"  After step:")
        print(f"    Phase: {core.phase}")
        print(f"    Pieces:")
        for p in core.pieces:
            alive = "ALIVE" if p.alive else "DEAD  "
            h = coord(p.hex) if p.hex else "DEAD"
            print(f"      Piece {p.id}: {alive} at {h}")
        print(f"    Active hexes: {sum(1 for h in core.hexes if h.is_active())}")

    print(f"\nFinal state:")
    for p in core.pieces:
        alive = "ALIVE" if p.alive else "DEAD  "
        h = coord(p.hex) if p.hex else "DEAD"
        print(f"  Piece {p.id}: {alive} at {h}")

    return core


if __name__ == "__main__":
    # The hexes from the log:
    # #001: P1 placed hex=52 (+4,-6,+2) = piece 4
    # #002: P2 placed hex=53 (+4,-5,+1) = piece 5
    # #003: P1 placed hex=38 (+3,-5,+2) = piece 6 (but piece 4 died!)
    # #004: P2 placed hex=37 (+2,-5,+3) = piece 7
    # #005: P1 placed hex=27 (+2,-4,+2) = piece 8
    # #006: P2 placed hex=54 (+4,-4,+0) = piece 9

    # Let's trace what happens after each placement
    # First, let's see what hexes 38 and 27 look like

    core = PenguinChessCore()
    core.reset(seed=None)  # No seed, use default board

    print("=== Hexes around hex 38 (+3,-5,+2) ===")
    hex_38_idx = core._hex_map.get((3, -5, 2))
    print(f"Hex 38 index: {hex_38_idx}")
    if hex_38_idx is not None:
        h38 = core.hexes[hex_38_idx]
        print(f"Hex 38: {coord(h38)} state={h38.state} pts={h38.points}")
        print(f"Neighbors of hex 38:")
        for nidx in core._neighbors[hex_38_idx]:
            nh = core.hexes[nidx]
            print(f"  {coord(nh)} state={nh.state} pts={nh.points}")

    print("\n=== Hexes around hex 27 (+2,-4,+2) ===")
    hex_27_idx = core._hex_map.get((2, -4, 2))
    print(f"Hex 27 index: {hex_27_idx}")
    if hex_27_idx is not None:
        h27 = core.hexes[hex_27_idx]
        print(f"Hex 27: {coord(h27)} state={h27.state} pts={h27.points}")
        print(f"Neighbors of hex 27:")
        for nidx in core._neighbors[hex_27_idx]:
            nh = core.hexes[nidx]
            print(f"  {coord(nh)} state={nh.state} pts={nh.points}")

    print("\n=== Hexes around hex 37 (+2,-5,+3) ===")
    hex_37_idx = core._hex_map.get((2, -5, 3))
    print(f"Hex 37 index: {hex_37_idx}")
    if hex_37_idx is not None:
        h37 = core.hexes[hex_37_idx]
        print(f"Hex 37: {coord(h37)} state={h37.state} pts={h37.points}")
        print(f"Neighbors of hex 37:")
        for nidx in core._neighbors[hex_37_idx]:
            nh = core.hexes[nidx]
            print(f"  {coord(nh)} state={nh.state} pts={nh.points}")

    # Now let's trace the actual game
    print("\n\n=== TRACING GAME ===")
    actions = [52, 53, 38, 37, 27, 54]  # From the log
    core2 = trace_game(None, actions)
