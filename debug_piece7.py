"""Debug script to trace piece 7's moves after P1's step 6 move."""
import sys
sys.path.insert(0, '.')

from penguinchess.core import PenguinChessCore

def coord(h):
    """Format hex coordinate as string."""
    return f"({h.q},{h.r},{h.s})" if h else "DEAD"

def trace_game(seed):
    """Play through a game and check piece 7's moves at key moments."""
    core = PenguinChessCore()
    core.reset(seed=seed)

    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    # Get legal placement actions for each step
    for step in range(6):
        legal = core.get_legal_actions()
        print(f"\nStep {step}: Player {core.current_player + 1}, legal placements: {legal[:6]}...")

        if not legal:
            print("No legal actions - should not happen in placement")
            break

        # Just pick the first legal action
        action = legal[0]
        print(f"  Choosing action {action}")
        core.step(action)

    print("\n--- End of Placement Phase ---")
    print_pieces_state(core)
    print(f"Phase: {core.phase}")
    print(f"Current player: {core.current_player + 1}")
    print(f"Current player: {core.current_player + 1}")

    # Now we're in movement phase
    # P1 (player 0) moves first
    # Need to find what piece 4's valid moves are

    # Get piece 4's current position and valid moves
    piece_4 = core.pieces[0]  # piece 4 is index 0
    piece_7 = core.pieces[4]  # piece 7 is index 4 (P2 pieces are indices 3,4,5)

    print(f"\nPiece 4: id={piece_4.id}, alive={piece_4.alive}, hex={coord(piece_4.hex)}")
    print(f"Piece 7: id={piece_7.id}, alive={piece_7.alive}, hex={coord(piece_7.hex)}")

    # Get valid moves for piece 4 (P1's turn)
    moves_4 = core._get_piece_moves(piece_4)
    print(f"\nPiece 4 valid moves: {len(moves_4)}")
    for m in moves_4[:5]:
        print(f"  -> {coord(m)} (state={m.state}, points={m.points})")
    if len(moves_4) > 5:
        print(f"  ... and {len(moves_4) - 5} more")

    if len(moves_4) == 0:
        print("  ** NO VALID MOVES FOR PIECE 4 - will be destroyed!")
        return

    # Execute P1's move (just pick the first valid move)
    target_hex = moves_4[0]
    target_idx = core._hex_map.get((target_hex.q, target_hex.r, target_hex.s))
    print(f"\nP1 (piece 4) moving to {coord(target_hex)} (idx={target_idx})")

    obs, reward, terminated, info = core.step(target_idx)
    print(f"Reward: {reward}, Terminated: {terminated}")
    print(f"Info: {info}")

    print("\n--- After P1's Move ---")
    print_pieces_state(core)

    # Now check piece 7's valid moves (P2's turn)
    piece_7 = core.pieces[4]  # refresh reference
    print(f"\nPiece 7: id={piece_7.id}, alive={piece_7.alive}, hex={coord(piece_7.hex)}")

    if piece_7.alive:
        moves_7 = core._get_piece_moves(piece_7)
        print(f"Piece 7 valid moves: {len(moves_7)}")
        for m in moves_7[:10]:
            print(f"  -> {coord(m)} (state={m.state}, points={m.points})")
        if len(moves_7) > 10:
            print(f"  ... and {len(moves_7) - 10} more")

        if len(moves_7) == 0:
            print("  ** NO VALID MOVES FOR PIECE 7!")
            print("  ** PIECE 7 SHOULD BE DESTROYED!")
    else:
        print("Piece 7 is already dead")

    # Check what hexes piece 7 could theoretically reach (same axis)
    print("\n--- Theoretical destinations for piece 7 (same axis) ---")
    if piece_7.hex:
        cur = piece_7.hex
        for h in core.hexes:
            if h.q == cur.q or h.r == cur.r or h.s == cur.s:
                occupied = core._hex_occupied(h)
                active = h.is_active()
                path_clear = core._path_clear(cur, h)
                print(f"  {coord(h)}: active={active}, occupied={occupied}, path_clear={path_clear}")


def print_pieces_state(core):
    print("All pieces:")
    for p in core.pieces:
        print(f"  Piece {p.id}: alive={p.alive}, hex={coord(p.hex)}")
    print(f"Active hexes count: {sum(1 for h in core.hexes if h.is_active())}")
    print(f"Occupied set: {core._occupied_set}")


if __name__ == "__main__":
    for seed in range(3):
        trace_game(seed)
