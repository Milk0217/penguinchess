"""Reproduce the exact bug: pieces being placed at wrong hexes."""
import sys
sys.path.insert(0, '.')

from penguinchess.core import PenguinChessCore

def run_test():
    """Run the exact scenario from user's log."""
    core = PenguinChessCore()
    core.reset(seed=None)  # Same as server

    print("=== Reproducing Bug Scenario ===")
    print("Placement sequence from log: [52, 53, 38, 37, 27, 54]")
    print("Expected: P1 places 4,6,8 and P2 places 5,7,9")
    print()

    actions = [52, 53, 38, 37, 27, 54]

    for step, action in enumerate(actions):
        player_before = core.current_player
        phase_before = core.phase
        total_placed = sum(p.hex is not None for p in core.pieces)

        # Check what hex we're trying to place at
        h = core.hexes[action]
        print(f"Step {step+1}: Player {player_before+1} (phase={phase_before})")
        print(f"  Action: hex {action} at ({h.q},{h.r},{h.s})")
        print(f"  Hex state: {h.state}, hex_active={h.is_active()}")

        obs, reward, terminated, info = core.step(action)

        print(f"  After step:")
        for p in core.pieces:
            if p.alive and p.hex:
                print(f"    Piece {p.id}: at ({p.hex.q},{p.hex.r},{p.hex.s})")

        # Check if any pieces died
        for p in core.pieces:
            if not p.alive:
                print(f"    Piece {p.id}: DIED")
        print()

    # Final state
    print("=== Final State ===")
    print("All pieces:")
    for p in core.pieces:
        status = "ALIVE" if p.alive else "DEAD"
        hex_info = f"({p.hex.q},{p.hex.r},{p.hex.s})" if p.hex else "NONE"
        print(f"  Piece {p.id}: {status} at {hex_info}")

    print()
    print("Expected placements:")
    print("  Piece 4: hex 52")
    print("  Piece 5: hex 53")
    print("  Piece 6: hex 38")
    print("  Piece 7: hex 37")
    print("  Piece 8: hex 27")
    print("  Piece 9: hex 54")

    # Verify
    expected = {
        4: 52, 5: 53, 6: 38, 7: 37, 8: 27, 9: 54
    }

    print()
    print("=== Verification ===")
    all_correct = True
    for piece_id, expected_hex in expected.items():
        piece = next(p for p in core.pieces if p.id == piece_id)
        if not piece.alive:
            print(f"FAIL: Piece {piece_id} is dead (expected hex {expected_hex})")
            all_correct = False
        elif piece.hex is None:
            print(f"FAIL: Piece {piece_id} has no hex (expected hex {expected_hex})")
            all_correct = False
        else:
            actual_idx = core._hex_map.get((piece.hex.q, piece.hex.r, piece.hex.s))
            if actual_idx != expected_hex:
                print(f"FAIL: Piece {piece_id} at hex {actual_idx} (expected hex {expected_hex})")
                all_correct = False
            else:
                print(f"OK: Piece {piece_id} at hex {actual_idx}")

    if all_correct:
        print("\nAll pieces placed correctly!")
    else:
        print("\nBUG: Some pieces placed incorrectly!")

if __name__ == "__main__":
    run_test()
