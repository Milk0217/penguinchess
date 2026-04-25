#!/usr/bin/env python3
"""
Test piece death - create a trapped piece scenario manually.
"""
from penguinchess.core import PenguinChessCore, Hex, Piece

def create_trapped_piece_scenario():
    """手动创建一个棋子被困住的场景"""
    print("=== 手动创建困住场景 ===\n")

    core = PenguinChessCore()
    core.reset()

    # 完成放置阶段
    legal = core.get_legal_actions()
    for i in range(6):
        core.step(legal[i % len(legal)])

    print(f"放置阶段完成: {core.phase}")
    print(f"活跃格子: {len([h for h in core.hexes if h.is_active()])}")

    # 找一个棋子，检查它的状态
    piece = core.pieces[0]  # P1 棋子 ID=4
    print(f"\n棋子 {piece.id} 位置: ({piece.hex.q}, {piece.hex.r}, {piece.hex.s})")
    print(f"棋子状态: alive={piece.alive}")

    # 分析这个位置的所有同轴格子
    print("\n同轴格子分析:")
    same_axis_hexes = []
    for h in core.hexes:
        if h.q == piece.hex.q or h.r == piece.hex.r or h.s == piece.hex.s:
            same_axis_hexes.append(h)

    # 按 q/r/s 轴分类
    q_axis = [h for h in same_axis_hexes if h.q == piece.hex.q]
    r_axis = [h for h in same_axis_hexes if h.r == piece.hex.r]
    s_axis = [h for h in same_axis_hexes if h.s == piece.hex.s]

    for axis_name, hex_list in [("q轴", q_axis), ("r轴", r_axis), ("s轴", s_axis)]:
        print(f"\n{axis_name} (棋子坐标={getattr(piece.hex, axis_name[0])}):")
        for h in sorted(hex_list, key=lambda x: (x.q, x.r, x.s)):
            # 检查这个格子是否可达
            is_occupied = core._hex_occupied(h)
            path_clear = core._path_clear(piece.hex, h) if not is_occupied else False
            in_moves = h in core._get_piece_moves(piece)

            status = []
            if not h.is_active(): status.append(f"state={h.state}")
            if is_occupied: status.append("被占据")
            if not path_clear and not is_occupied: status.append("路径不通")
            if in_moves: status.append("[可达]")

            print(f"  ({h.q}, {h.r}, {h.s}): points={h.points:2d} {', '.join(status) if status else '空闲'}")

    moves = core._get_piece_moves(piece)
    print(f"\n棋子 {piece.id} 合法移动数: {len(moves)}")

    # 现在人为把棋子周围的所有活跃格子都变成 -1 或 0
    print("\n\n=== 人为创造困住场景 ===")
    print("将所有非占据格子设为已使用...")

    # 标记所有非目标格子为已使用
    for h in core.hexes:
        if h.is_active() and not core._hex_occupied(h):
            h.mark_used()  # 设为已使用

    # 重新计算合法移动
    moves_after = core._get_piece_moves(piece)
    print(f"修改后棋子 {piece.id} 合法移动数: {len(moves_after)}")

    # 现在手动调用 _destroy_immobile_pieces
    print("\n调用 _destroy_immobile_pieces():")
    core._destroy_immobile_pieces()

    # 检查棋子状态
    piece_after = core.pieces[0]
    print(f"棋子 {piece_after.id} 状态: alive={piece_after.alive}, hex={piece_after.hex}")

    # 再检查所有棋子
    print("\n所有棋子状态:")
    for p in core.pieces:
        moves = core._get_piece_moves(p)
        print(f"  棋子 {p.id}: alive={p.alive}, hex={p.hex.q if p.hex else None}, 可移动={len(moves)}")

def test_scenario_with_elimination():
    """测试消除后棋子被困的场景"""
    print("\n\n=== 测试消除后被困场景 ===\n")

    core = PenguinChessCore(seed=123)
    core.reset(seed=123)

    # 完成放置
    legal = core.get_legal_actions()
    for i in range(6):
        core.step(legal[i % len(legal)])

    print("放置阶段完成")

    # 移动几回合
    for i in range(5):
        legal = core.get_legal_actions()
        if legal:
            core.step(legal[0])

    print(f"\n移动 5 回合后:")
    print(f"活跃格子: {len([h for h in core.hexes if h.is_active()])}")

    # 检查所有棋子
    for p in core.pieces:
        if p.alive and p.hex:
            moves = core._get_piece_moves(p)
            print(f"  棋子 {p.id}: 在 ({p.hex.q},{p.hex.r},{p.hex.s}), 可移动={len(moves)}")

    # 执行消除
    eliminated = core._eliminate_disconnected_hexes()
    print(f"\n消除 {eliminated} 个格子")
    print(f"消除后活跃格子: {len([h for h in core.hexes if h.is_active()])}")

    # 再次检查
    print("\n消除后棋子状态:")
    for p in core.pieces:
        if p.alive and p.hex:
            moves = core._get_piece_moves(p)
            print(f"  棋子 {p.id}: 可移动={len(moves)}")
            if not moves:
                print(f"    *** 应该被消除！ ***")

    # 调用销毁
    print("\n调用 _destroy_immobile_pieces():")
    core._destroy_immobile_pieces()

    print("\n销毁后棋子状态:")
    for p in core.pieces:
        print(f"  棋子 {p.id}: alive={p.alive}, hex={p.hex.q if p.hex else None}")

if __name__ == "__main__":
    create_trapped_piece_scenario()
    test_scenario_with_elimination()
