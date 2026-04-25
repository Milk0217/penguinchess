#!/usr/bin/env python3
"""
Test piece death logic with specific scenarios.
"""
from penguinchess.core import PenguinChessCore, Hex

def test_piece_death_scenario():
    """测试棋子被困在死胡同的情况"""
    print("=== 测试棋子阵亡逻辑 ===\n")

    # 创建一个简化的测试场景
    core = PenguinChessCore(seed=42)
    core.reset(seed=42)

    # 放置阶段完成
    legal = core.get_legal_actions()
    for i in range(6):
        action = legal[i % len(legal)]
        obs, reward, terminated, info = core.step(action)

    print(f"放置阶段完成，当前 phase: {core.phase}")
    print(f"P1 棋子: {[f'ID={p.id}, alive={p.alive}, hex={p.hex.q if p.hex else None}' for p in core.pieces if p.owner() == 0]}")
    print(f"P2 棋子: {[f'ID={p.id}, alive={p.alive}, hex={p.hex.q if p.hex else None}' for p in core.pieces if p.owner() == 1]}")

    # 手动创建一个困住的棋子场景
    print("\n=== 手动创建困住场景 ===")
    core2 = PenguinChessCore()
    core2.reset()

    # 先完成放置阶段
    legal = core2.get_legal_actions()
    for i in range(6):
        core2.step(legal[i % len(legal)])

    # 假设某个棋子被完全包围
    # 找一个棋子，手动设置周围格子状态
    piece = core2.pieces[0]  # P1 的第一个棋子
    if piece.hex is None:
        print(f"棋子 {piece.id} 还没有放置")
        return

    print(f"棋子 {piece.id} 初始位置: q={piece.hex.q}, r={piece.hex.r}, s={piece.hex.s}")
    print(f"周围格子状态:")

    # 检查所有格子
    for h in core2.hexes:
        if h.is_active():
            # 检查是否与 piece 同轴
            if h.q == piece.hex.q or h.r == piece.hex.r or h.s == piece.hex.s:
                moves = core2._get_piece_moves(piece)
                is_target = h in moves
                print(f"  ({h.q}, {h.r}, {h.s}) points={h.points} - {'可移动' if is_target else '不可移动'}")

    print(f"\n棋子 {piece.id} 的合法移动: {len(core2._get_piece_moves(piece))} 个")
    print(f"棋子状态: alive={piece.alive}")

    # 现在模拟移动几步，创造一个困住的场景
    print("\n=== 模拟移动 ===")
    for turn in range(10):
        if core2._terminated:
            print(f"游戏在第 {turn} 回合结束")
            break

        legal = core2.get_legal_actions()
        if not legal:
            print(f"第 {turn} 回合: 没有合法动作！")
            # 检查所有棋子状态
            for p in core2.pieces:
                if p.alive and p.hex:
                    moves = core2._get_piece_moves(p)
                    print(f"  棋子 {p.id}: alive={p.alive}, hex=({p.hex.q},{p.hex.r},{p.hex.s}), 可移动数={len(moves)}")
                    if not moves:
                        print(f"    *** BUG: 这个棋子应该被消除！***")
            break

        action = legal[0]
        obs, reward, terminated, info = core2.step(action)
        print(f"第 {turn} 回合: 动作={action}, reward={reward:.3f}")

        # 每步后检查棋子状态
        for p in core2.pieces:
            if p.alive and p.hex:
                moves = core2._get_piece_moves(p)
                if not moves:
                    print(f"  棋子 {p.id} 困住了！alive={p.alive}, 应该被消除")

    print(f"\n最终状态:")
    print(core2.render())

def test_isolated_piece():
    """测试完全孤立的棋子"""
    print("\n\n=== 测试孤立棋子 ===")

    core = PenguinChessCore()
    core.reset()

    # 放置所有棋子
    legal = core.get_legal_actions()
    for i in range(6):
        core.step(legal[i % len(legal)])

    # 人为创造一个孤立场景：让所有格子都变成 -1 或 0
    # 先看看当前的 hex 状态
    print("活跃格子数量:", len([h for h in core.hexes if h.is_active()]))

    # 找一个棋子，检查它到底有没有合法移动
    for p in core.pieces:
        if p.alive and p.hex:
            moves = core._get_piece_moves(p)
            print(f"棋子 {p.id} 在 ({p.hex.q},{p.hex.r},{p.hex.s}), 合法移动数: {len(moves)}")
            if moves:
                print(f"  可移动目标: {[(m.q, m.r, m.s, m.points) for m in moves[:5]]}")

if __name__ == "__main__":
    test_piece_death_scenario()
    test_isolated_piece()
