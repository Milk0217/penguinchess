#!/usr/bin/env python3
"""
Test corner piece death - piece surrounded by 0 and -1 hexes should die.
"""
from penguinchess.core import PenguinChessCore, Hex, Piece, PLAYER_1_PIECES, PLAYER_2_PIECES

def setup_corner_game():
    """创建一个角落棋子的场景"""
    print("=== 角落棋子阵亡测试 ===\n")

    core = PenguinChessCore(seed=999)
    core.reset(seed=999)

    # 1. 先完成正常的放置阶段
    print("--- 放置阶段 ---")
    legal = core.get_legal_actions()

    # P1 放置 3 个棋子
    for i in range(3):
        action = legal[i % len(legal)]
        core.step(action)
        print(f"P1 放置: hex {action}")

    # P2 放置 3 个棋子
    for i in range(3, 6):
        action = legal[i % len(legal)]
        core.step(action)
        print(f"P2 放置: hex {action}")

    print(f"\n放置完成，当前 phase: {core.phase}")

    # 2. 找到棋盘边缘的格子（角落）
    print("\n--- 寻找角落格子 ---")
    corner_hexes = []
    for h in core.hexes:
        # 角落格子：在边界上，周围同轴的格子不多
        # 检查是否是边缘格子
        is_corner = False
        for other in core.hexes:
            if other is h:
                continue
            # 如果在同一条线上但距离很远，可能是在边缘
            if h.q == other.q or h.r == other.r or h.s == other.s:
                # 检查是否还有其他格子在更远的地方
                dist = max(abs(other.q - h.q), abs(other.r - h.r), abs(other.s - h.s))
                if dist > 3:  # 距离较远，可能是边缘
                    is_corner = True
                    break
        if is_corner and h.value > 0:
            corner_hexes.append(h)

    print(f"找到 {len(corner_hexes)} 个角落格子")
    for h in corner_hexes[:5]:
        print(f"  ({h.q}, {h.r}, {h.s}) value={h.value}")

    # 3. 把一个棋子移动到角落
    print("\n--- 移动棋子到角落 ---")

    # 找一个还有棋子的位置
    piece_to_move = None
    for p in core.pieces:
        if p.alive and p.hex and p.id in PLAYER_1_PIECES:
            piece_to_move = p
            break

    if piece_to_move:
        print(f"选择棋子 {piece_to_move.id} 从 ({piece_to_move.hex.q}, {piece_to_move.hex.r}, {piece_to_move.hex.s})")

        # 找一个角落格子作为目标
        target_corner = None
        for corner in corner_hexes:
            if corner.value > 0 and not core._hex_occupied(corner):
                # 检查是否能移动到这个角落
                moves = core._get_piece_moves(piece_to_move)
                if corner in moves:
                    target_corner = corner
                    break

        if target_corner:
            print(f"目标角落: ({target_corner.q}, {target_corner.r}, {target_corner.s})")
            # 找到这个角落的 index
            target_idx = core._hex_to_index(target_corner)
            print(f"目标 index: {target_idx}")

            # 手动执行移动（直接操作）
            old_hex = piece_to_move.hex
            old_hex.value = -1  # 旧格子标记为已使用
            piece_to_move.hex = target_corner
            piece_to_move.hex_value = target_corner.value
            target_corner.value = 0  # 新格子被占据
            core._rebuild_occupied()

            print(f"移动后: 棋子 {piece_to_move.id} 在 ({piece_to_move.hex.q}, {piece_to_move.hex.r}, {piece_to_move.hex.s})")

    # 4. 现在把角落周围的其他格子都变成 0 或 -1
    print("\n--- 包围角落 ---")

    # 找到所有与角落同轴的格子
    corner = piece_to_move.hex
    neighbors_2 = []  # 两步以内的格子

    for h in core.hexes:
        if h is corner:
            continue
        # 同轴
        if h.q == corner.q or h.r == corner.r or h.s == corner.s:
            dist = max(abs(h.q - corner.q), abs(h.r - corner.r), abs(h.s - corner.s))
            if dist <= 2:  # 两步以内
                neighbors_2.append((h, dist))

    # 先把所有邻居都标记为可占据
    for h, dist in neighbors_2:
        if h.value > 0:
            h.value = -1  # 设为已使用

    # 然后把除了角落本身外的所有格子设为已使用
    for h in core.hexes:
        if h is not corner and h.value > 0:
            h.value = -1

    core._rebuild_occupied()

    print(f"角落格子 ({corner.q}, {corner.r}, {corner.s}) 周围都是 -1 或 0")
    print(f"角落格子 value: {corner.value}")

    # 5. 检查棋子的合法移动
    print("\n--- 检查棋子状态 ---")
    moves = core._get_piece_moves(piece_to_move)
    print(f"棋子 {piece_to_move.id} 的合法移动数: {len(moves)}")

    # 6. 调用销毁
    print("\n--- 调用 _destroy_immobile_pieces() ---")
    core._destroy_immobile_pieces()

    # 7. 检查结果
    print("\n--- 结果 ---")
    for p in core.pieces:
        if p.id == piece_to_move.id:
            print(f"角落棋子 {p.id}: alive={p.alive}, hex={p.hex.q if p.hex else None}")
            if p.alive:
                print("  *** BUG: 角落棋子应该阵亡但还是 alive=True ***")
            else:
                print("  正确: 角落棋子已阵亡")
        else:
            print(f"其他棋子 {p.id}: alive={p.alive}, hex={p.hex.q if p.hex else None}")

def manual_corner_test():
    """手动创建一个完美的角落场景"""
    print("\n\n=== 手动创建完美角落场景 ===\n")

    core = PenguinChessCore()
    core.reset()

    # 找一个棋子
    piece = core.pieces[0]  # P1 第一个棋子
    print(f"棋子 {piece.id} 初始: ({piece.hex.q}, {piece.hex.r}, {piece.hex.s})")

    # 找到这个棋子的所有同轴格子
    same_axis = []
    for h in core.hexes:
        if h.q == piece.hex.q or h.r == piece.hex.r or h.s == piece.hex.s:
            same_axis.append(h)

    print(f"同轴格子数: {len(same_axis)}")
    for h in sorted(same_axis, key=lambda x: (x.q, x.r, x.s)):
        print(f"  ({h.q}, {h.r}, {h.s}): value={h.value}")

    # 把棋子移动到边缘的同轴格子
    # 找一个边缘位置的同轴格子
    edge_hex = None
    for h in same_axis:
        if h.value > 0 and h is not piece.hex:
            # 检查是否在边缘
            is_edge = False
            for other in core.hexes:
                if other is h:
                    continue
                if h.q == other.q or h.r == other.r or h.s == other.s:
                    dist = max(abs(other.q - h.q), abs(other.r - h.r), abs(other.s - h.s))
                    if dist > 3:
                        is_edge = True
                        break
            if is_edge:
                edge_hex = h
                break

    if edge_hex:
        print(f"\n移动棋子到边缘格子 ({edge_hex.q}, {edge_hex.r}, {edge_hex.s})")

        # 移动棋子
        old_hex = piece.hex
        old_hex.value = -1
        piece.hex = edge_hex
        piece.hex_value = edge_hex.value
        edge_hex.value = 0
        core._rebuild_occupied()

        # 把棋盘上除了这个角落外的所有格子都设为 -1
        for h in core.hexes:
            if h is not edge_hex:
                h.value = -1

        # 角落格子本身设为 0（被占据）
        edge_hex.value = 0

        print(f"现在角落格子 ({edge_hex.q}, {edge_hex.r}, {edge_hex.s}) 周围全是 -1")

        # 检查合法移动
        moves = core._get_piece_moves(piece)
        print(f"\n棋子 {piece.id} 合法移动数: {len(moves)}")

        # 调用销毁
        print("\n调用 _destroy_immobile_pieces()...")
        core._destroy_immobile_pieces()

        # 检查
        print(f"棋子 {piece.id}: alive={piece.alive}, hex={piece.hex}")
        if piece.alive:
            print("*** BUG: 棋子应该阵亡但还是 alive ***")
        else:
            print("正确: 棋子已阵亡")

if __name__ == "__main__":
    setup_corner_game()
    manual_corner_test()
