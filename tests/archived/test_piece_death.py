#!/usr/bin/env python3
"""
Test piece death logic with a specific game scenario.
"""
from penguinchess.core import PenguinChessCore

def test_piece_death():
    """Test that pieces with no valid moves are correctly marked as dead."""
    # Use a seed that produces a specific board layout
    # Try different seeds to find one that creates a trapped piece scenario
    for seed in range(100):
        core = PenguinChessCore(seed=seed)
        core.reset(seed=seed)

        print(f"\n=== Testing seed={seed} ===")
        print(core.render())

        # Complete placement phase
        print("\n--- Placement Phase ---")
        legal = core.get_legal_actions()
        for i in range(6):  # 6 placements total
            action = legal[i % len(legal)]
            obs, reward, terminated, info = core.step(action)
            print(f"Step {i+1}: placed at hex {action}, reward={reward:.3f}")
            legal = core.get_legal_actions()

        print(f"\n--- Movement Phase Start ---")
        print(core.render())

        # Simulate some movement turns
        for turn in range(20):
            if core._terminated:
                print(f"\nGame terminated at turn {turn}")
                break

            legal = core.get_legal_actions()
            if not legal:
                print(f"\nNo legal actions at turn {turn} - checking for immobile pieces...")
                # Check piece status
                for p in core.pieces:
                    if p.alive:
                        moves = core._get_piece_moves(p)
                        if not moves:
                            print(f"  Piece {p.id} (Player {p.owner()}) is ALIVE but has NO valid moves!")
                            print(f"    hex={p.hex}")
                            print(f"    This is the BUG - piece should be destroyed!")
                break

            # Pick a legal action
            action = legal[0]
            obs, reward, terminated, info = core.step(action)
            print(f"Turn {turn+1}: action={action}, reward={reward:.3f}, terminated={terminated}")

            # Check piece status after each step
            for p in core.pieces:
                if not p.alive:
                    print(f"  Piece {p.id} destroyed correctly")
        else:
            print("\nReached max turns without game ending")

        print(f"\nFinal state:")
        print(core.render())

        # Check for bug: alive piece with no valid moves
        bug_found = False
        for p in core.pieces:
            if p.alive and p.hex is not None:
                moves = core._get_piece_moves(p)
                if not moves:
                    print(f"\n*** BUG: Piece {p.id} is alive but has NO valid moves! ***")
                    bug_found = True

        if bug_found:
            print(f"\nSeed {seed} reproduces the bug!")
            return seed

    print("\nNo bug found in seeds 0-99")
    return None

if __name__ == "__main__":
    test_piece_death()
