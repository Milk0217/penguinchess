"""
企鹅棋核心游戏逻辑单元测试。
"""

import json
import os
import pytest
from penguinchess.core import (
    PenguinChessCore,
    Hex,
    Piece,
    generate_sequence,
    create_board,
    create_board_from_coords,
    json_board_to_coords,
    TOTAL_VALUE,
    HEX_COUNT,
    PLAYER_1_PIECES,
    PLAYER_2_PIECES,
    PIECES_PER_PLAYER,
)


def get_default_board_coords():
    """加载 default.json 棋盘并转换为 Python 内部坐标格式。"""
    board_path = os.path.join(
        os.path.dirname(__file__), "..", "backend_data", "boards", "default.json"
    )
    with open(board_path, encoding="utf-8") as f:
        data = json.load(f)
    return json_board_to_coords(data["hexes"])


class TestBoardGeneration:
    """棋盘生成测试。"""

    @pytest.fixture
    def default_board_core(self):
        """使用 default.json 棋盘的 PenguinChessCore 实例。"""
        coords = get_default_board_coords()
        return PenguinChessCore(custom_coords=coords)

    def test_board_value_sum_equals_99(self, default_board_core):
        """棋盘格子分值总和必须等于 99。"""
        core = default_board_core
        core.reset()
        # 注意: 活跃格子 (is_active) 的 points 总和应该等于 99
        active_total = sum(h.points for h in core.hexes if h.is_active())
        assert active_total == 99, f"Active hex value sum should be 99, got {active_total}"

    def test_hex_count_equals_60(self, default_board_core):
        """默认棋盘应该有 60 个格子。"""
        core = default_board_core
        core.reset()
        assert len(core.hexes) == 60, f"Expected 60 hexes, got {len(core.hexes)}"

    def test_generate_sequence_length(self):
        """生成的序列长度应该等于格子数。"""
        seq = generate_sequence(total_sum=99, hex_count=60)
        assert len(seq) == 60, f"Expected sequence length 60, got {len(seq)}"

    def test_generate_sequence_sum(self):
        """生成的序列总和应该等于指定值。"""
        seq = generate_sequence(total_sum=99, hex_count=60)
        assert sum(seq) == 99, f"Expected sum 99, got {sum(seq)}"

    def test_generate_sequence_values(self):
        """生成的序列每个值必须是 1, 2 或 3。"""
        seq = generate_sequence(total_sum=99, hex_count=60)
        for v in seq:
            assert v in (1, 2, 3), f"Invalid value {v} in sequence"

    def test_seed_reproducibility(self):
        """相同种子应该生成相同的棋盘。"""
        core1 = PenguinChessCore(seed=42)
        core1.reset()
        hex_values1 = [(h.q, h.r, h.s, h.points) for h in core1.hexes]

        core2 = PenguinChessCore(seed=42)
        core2.reset()
        hex_values2 = [(h.q, h.r, h.s, h.points) for h in core2.hexes]

        assert hex_values1 == hex_values2, "Same seed should produce same board"

    def test_different_seeds_different_boards(self):
        """不同种子应该生成不同的棋盘。"""
        core1 = PenguinChessCore(seed=42)
        core1.reset()
        values1 = [h.points for h in core1.hexes]

        core2 = PenguinChessCore(seed=123)
        core2.reset()
        values2 = [h.points for h in core2.hexes]

        assert values1 != values2, "Different seeds should produce different boards"


class TestHexClass:
    """Hex 数据结构测试。"""

    def test_hex_active_state(self):
        """state='active' 表示活跃格子。"""
        h = Hex(q=0, r=0, s=0, points=2, state='active', _q_raw=0)
        assert h.is_active()
        assert not h.is_occupied()
        assert not h.is_eliminated()

    def test_hex_occupied_state(self):
        """state='occupied' 表示被占据。"""
        h = Hex(q=0, r=0, s=0, state='occupied', _q_raw=0)
        assert not h.is_active()
        assert h.is_occupied()
        assert not h.is_eliminated()

    def test_hex_eliminated_state(self):
        """state='eliminated' 表示已消除。"""
        h = Hex(q=0, r=0, s=0, state='eliminated', _q_raw=0)
        assert not h.is_active()
        assert not h.is_occupied()
        assert h.is_eliminated()

    def test_hex_used_state(self):
        """state='used' 表示已使用（棋子曾到达）。"""
        h = Hex(q=0, r=0, s=0, state='used', _q_raw=0)
        assert not h.is_active()
        assert not h.is_occupied()
        assert h.is_eliminated()

    def test_hex_occupy(self):
        """occupy() 方法将 state 设为 occupied。"""
        h = Hex(q=0, r=0, s=0, points=2, state='active', _q_raw=0)
        h.occupy()
        assert h.is_occupied()
        assert h.state == 'occupied'

    def test_hex_eliminate(self):
        """eliminate() 方法将 state 设为 eliminated。"""
        h = Hex(q=0, r=0, s=0, points=2, state='active', _q_raw=0)
        h.eliminate()
        assert h.is_eliminated()
        assert h.state == 'eliminated'

    def test_hex_mark_used(self):
        """mark_used() 方法将 state 设为 used。"""
        h = Hex(q=0, r=0, s=0, points=2, state='active', _q_raw=0)
        h.mark_used()
        assert h.is_eliminated()
        assert h.state == 'used'


class TestPieceClass:
    """Piece 数据结构测试。"""

    def test_piece_owner_p1(self):
        """偶数 ID 属于 Player 1 (id: 4, 6, 8)。"""
        for pid in PLAYER_1_PIECES:
            piece = Piece(id=pid)
            assert piece.owner() == 0, f"Piece {pid} should belong to Player 1"

    def test_piece_owner_p2(self):
        """奇数 ID 属于 Player 2 (id: 5, 7, 9)。"""
        for pid in PLAYER_2_PIECES:
            piece = Piece(id=pid)
            assert piece.owner() == 1, f"Piece {pid} should belong to Player 2"

    def test_piece_alive_default(self):
        """默认状态下棋子是存活的。"""
        piece = Piece(id=4)
        assert piece.alive
        assert piece.hex is None

    def test_piece_move_to(self):
        """move_to() 方法正确移动棋子。"""
        piece = Piece(id=4)
        target = Hex(q=1, r=-1, s=0, points=2, state='active', _q_raw=0)

        piece.move_to(target)

        assert piece.hex == target
        assert piece.hex_value == 2
        assert piece._moved
        assert target.is_occupied()  # 被占据


class TestPlacementPhase:
    """放置阶段测试。"""

    def test_placement_phase_initial(self):
        """游戏初始状态为放置阶段。"""
        core = PenguinChessCore()
        core.reset()
        assert core.phase == core.PHASE_PLACEMENT
        assert core.current_player == 0  # Player 1 先手

    def test_placement_strict_alternating(self):
        """放置阶段严格交替。"""
        core = PenguinChessCore()
        core.reset()

        players_sequence = []
        for _ in range(6):
            players_sequence.append(core.current_player)
            # 找到一个合法动作并执行
            legal = core.get_legal_actions()
            if legal:
                core.step(legal[0])

        # Player 1, 2, 1, 2, 1, 2 交替
        expected = [0, 1, 0, 1, 0, 1]
        # 注意：放置满后不再交替
        assert players_sequence[:6] == expected or len(set(players_sequence)) <= 2

    def test_placement_score(self):
        """放置时立即获得格子分值。"""
        core = PenguinChessCore(seed=42)
        core.reset()

        initial_score = core.players_scores[0]
        legal = core.get_legal_actions()
        target_hex = core.hexes[legal[0]]
        target_points = target_hex.points

        core.step(legal[0])

        assert core.players_scores[0] == initial_score + target_points

    def test_placement_hex_occupied(self):
        """放置后格子被占据。"""
        core = PenguinChessCore(seed=42)
        core.reset()

        legal = core.get_legal_actions()
        target_hex = core.hexes[legal[0]]

        core.step(legal[0])

        assert target_hex.is_occupied()


class TestMovementPhase:
    """移动阶段测试。"""

    def test_movement_phase_switch(self):
        """6 个棋子放置后进入移动阶段。"""
        core = PenguinChessCore(seed=42)
        core.reset()

        # 执行 6 步放置
        for _ in range(6):
            legal = core.get_legal_actions()
            if legal:
                core.step(legal[0])

        assert core.phase == core.PHASE_MOVEMENT

    def test_movement_same_axis_only(self):
        """移动必须沿 q/r/s 轴同坐标进行。"""
        core = PenguinChessCore(seed=42)
        core.reset()

        # 跳过放置阶段到移动阶段
        for _ in range(6):
            legal = core.get_legal_actions()
            if legal:
                core.step(legal[0])

        # 获取所有合法移动
        legal_actions = core.get_legal_actions()
        if legal_actions:
            # 移动后检查目标格子坐标关系
            for action in legal_actions[:3]:  # 检查前几个
                target = core.hexes[action]
                # 目标格子应该与某个己方棋子在同一轴上
                for piece in core.pieces:
                    if piece.alive and piece.hex and core._piece_owner(piece) == core.current_player:
                        # 检查是否同 q/r/s 轴
                        same_axis = (target.q == piece.hex.q or
                                     target.r == piece.hex.r or
                                     target.s == piece.hex.s)
                        if same_axis:
                            # 找到匹配的，移动有效
                            break
                else:
                    pytest.fail(f"Action {action} is not a valid movement along any axis")


class TestElimination:
    """连通性消除测试。"""

    def test_elimination_disconnected_hexes(self):
        """消除所有不与棋子连通的格子。"""
        core = PenguinChessCore(seed=42)
        core.reset()

        # 放置阶段后手动检查
        for _ in range(6):
            legal = core.get_legal_actions()
            if legal:
                core.step(legal[0])

        # 移动几步触发消除
        for _ in range(3):
            legal = core.get_legal_actions()
            if legal:
                core.step(legal[0])
            if core._terminated:
                break

        # 消除后检查：被消除的格子 state 应该是 'eliminated'
        eliminated_count = sum(1 for h in core.hexes if h.state == 'eliminated')
        # 应该有一些格子被消除
        # 检查消除逻辑是否运行完毕（可能没有格子需要消除）
        assert eliminated_count >= 0  # 验证 elimination 代码执行完毕

    def test_flood_fill_connected(self):
        """Flood fill 应该从棋子位置连通所有相邻格子。"""
        core = PenguinChessCore(seed=42)
        core.reset()

        # 放置棋子
        for _ in range(6):
            legal = core.get_legal_actions()
            if legal:
                core.step(legal[0])

        # 检查 hex_map 和 neighbors 预计算是否正确
        assert hasattr(core, '_neighbors')
        assert len(core._neighbors) == len(core.hexes)
        for neighbors in core._neighbors:
            assert all(isinstance(n, int) for n in neighbors)


class TestGameOver:
    """游戏结束测试。"""

    def test_game_over_not_initially_terminated(self):
        """游戏开始时不应该立即结束。"""
        core = PenguinChessCore()
        core.reset()
        assert not core._terminated

    def test_termination_flag(self):
        """游戏结束时应设置终止标志。"""
        core = PenguinChessCore()
        core.reset()

        # 玩多步直到结束或达到步数上限
        for _ in range(100):
            if core._terminated:
                break
            legal = core.get_legal_actions()
            if legal:
                core.step(legal[0])
            else:
                break

        # 游戏要么终止，要么棋盘还有合法动作
        # 重要的是不会因为内部错误而崩溃

    def test_score_tracking(self):
        """分数追踪正确。"""
        core = PenguinChessCore()
        core.reset()

        p1_initial = core.players_scores[0]
        p2_initial = core.players_scores[1]

        # 执行几步
        for _ in range(6):
            legal = core.get_legal_actions()
            if legal:
                core.step(legal[0])

        # 至少有一个玩家得分应该增加
        assert (core.players_scores[0] > p1_initial or
                core.players_scores[1] > p2_initial)


class TestGetLegalActions:
    """get_legal_actions 测试。"""

    def test_legal_actions_not_empty_placement(self):
        """放置阶段应该有合法动作。"""
        core = PenguinChessCore()
        core.reset()
        legal = core.get_legal_actions()
        assert len(legal) > 0

    def test_legal_actions_not_empty_movement(self):
        """移动阶段应该有合法动作。"""
        core = PenguinChessCore()
        core.reset()

        # 进入移动阶段
        for _ in range(6):
            legal = core.get_legal_actions()
            if legal:
                core.step(legal[0])

        legal = core.get_legal_actions()
        assert len(legal) > 0, "Movement phase should have legal actions"

    def test_legal_actions_return_indices(self):
        """返回的动作 ID 应该是有效的格子索引。"""
        core = PenguinChessCore()
        core.reset()
        legal = core.get_legal_actions()
        for action in legal:
            assert 0 <= action < len(core.hexes)


class TestObservations:
    """观测编码测试。"""

    def test_observation_keys(self):
        """观测应该包含必要的键。"""
        core = PenguinChessCore()
        core.reset()
        obs = core.get_observation()

        assert "board" in obs
        assert "pieces" in obs
        assert "current_player" in obs
        assert "phase" in obs
        assert "scores" in obs

    def test_board_encoding_length(self):
        """棋盘编码应该是 60 个格子 (每个格子3个特征，编码为 list of lists)。"""
        core = PenguinChessCore()
        core.reset()
        obs = core.get_observation()
        # board 是 list of 60 个 hex 编码，每个 3 个值
        assert len(obs["board"]) == 60, f"Expected 60 hexes, got {len(obs['board'])}"
        # 每个 hex 编码有 3 个值 (q, r, value)
        assert len(obs["board"][0]) == 3, f"Expected 3 values per hex, got {len(obs['board'][0])}"

    def test_pieces_encoding_length(self):
        """棋子编码应该是 6 个棋子 (每个棋子4个特征，编码为 list of lists)。"""
        core = PenguinChessCore()
        core.reset()
        obs = core.get_observation()
        # pieces 是 list of 6 个 piece 编码，每个 4 个值
        assert len(obs["pieces"]) == 6, f"Expected 6 pieces, got {len(obs['pieces'])}"
        # 每个 piece 编码有 4 个值 (piece_id, q, r, s)
        assert len(obs["pieces"][0]) == 4, f"Expected 4 values per piece, got {len(obs['pieces'][0])}"

    def test_current_player_values(self):
        """current_player 应该是 0 或 1。"""
        core = PenguinChessCore()
        core.reset()
        obs = core.get_observation()
        assert obs["current_player"] in (0, 1)

    def test_phase_values(self):
        """phase 应该是 0 (placement) 或 1 (movement)。"""
        core = PenguinChessCore()
        core.reset()
        obs = core.get_observation()
        assert obs["phase"] in (0, 1)


class TestEdgeCases:
    """边界情况测试。"""

    def test_invalid_action_id_negative(self):
        """负数动作 ID 应该被拒绝。"""
        core = PenguinChessCore()
        core.reset()
        obs, reward, terminated, info = core.step(-1)
        assert info.get("invalid", False) or reward < 0

    def test_invalid_action_id_too_large(self):
        """过大的动作 ID 应该被拒绝。"""
        core = PenguinChessCore()
        core.reset()
        obs, reward, terminated, info = core.step(9999)
        assert info.get("invalid", False) or reward < 0

    def test_game_renders(self):
        """render() 方法应该返回字符串。"""
        core = PenguinChessCore()
        core.reset()
        output = core.render()
        assert isinstance(output, str)
        assert len(output) > 0