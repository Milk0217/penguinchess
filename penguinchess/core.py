"""
企鹅棋核心游戏逻辑（Python 版）
与 Web JS 版本严格对齐，用作 Gymnasium 环境后端和未来 Web API 核心。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# =============================================================================
# 常量（与 statics/config.js 对齐）
# =============================================================================

TOTAL_VALUE = 99  # 10×3 + 20×2 + 30×1 = 99
HEX_COUNT = 60  # 与 JS CONFIG.HEX_COUNT 一致
COUNT_OF_THREE_MIN = 8
COUNT_OF_THREE_MAX = 10
THREE_VALUE = 3

# 玩家棋子 ID
PLAYER_1_PIECES = [4, 6, 8]
PLAYER_2_PIECES = [5, 7, 9]
PIECES_PER_PLAYER = 3

# 棋盘 q 值有效范围（对应 JS qAdjustments 的键）
VALID_Q_RANGE = list(range(-4, 4))  # -4, -3, -2, -1, 0, 1, 2, 3

# q 值调整表（对应 JS qAdjustments）
# 键为字符串，与 JS 行为一致
Q_ADJUSTMENTS = {
    "-4": 2, "-3": 1, "-2": 0, "-1": 0,
    "0": 0, "1": -1, "2": -2, "3": -2,
}

# 6 个立方体方向偏移量
HEX_DIRECTIONS = [
    (1, -1, 0),
    (1, 0, -1),
    (0, 1, -1),
    (-1, 1, 0),
    (-1, 0, 1),
    (0, -1, 1),
]


# =============================================================================
# 数据结构
# =============================================================================

@dataclass(slots=True)
class Hex:
    """
    六边形格子。坐标使用立方体坐标 (q, r, s)，满足 q + r + s = 0。

    q/r/s 存储的是经过 row_range 调整后的值（与 JS createBoard 一致），
    额外记录 _q_raw = 原始 q 值（用于邻居查找中对齐 q 轴）。

    状态机:
      state: 'active'    - 可放置/可移动，points 有效
             'occupied'  - 有棋子占据
             'used'      - 棋子曾到达，不可再占据
             'eliminated' - 断连被消除，不可再占据
      points: 1/2/3 分值，仅 state='active' 时有效
    """
    q: int       # adjusted_r: r + qAdjustments[q_raw]
    r: int       # adjusted_s: -q_raw - adjusted_r
    s: int       # s_raw: q_raw + r_raw（原始 q + 原始 r，用于合法性检查）
    points: int = 0   # 分值（1/2/3），仅 active 时有效
    state: str = 'active'  # 'active' | 'occupied' | 'used' | 'eliminated'
    _q_raw: int = field(default=0)  # 原始 q 值，用于邻居 q 轴偏移

    def __hash__(self):
        return hash((self.q, self.r, self.s))

    def __eq__(self, other):
        if not isinstance(other, Hex):
            return False
        return self.q == other.q and self.r == other.r and self.s == other.s

    def is_active(self) -> bool:
        """是否活跃（可放置/可移动）。"""
        return self.state == 'active'

    def is_occupied(self) -> bool:
        """是否被棋子占据。"""
        return self.state == 'occupied'

    def is_eliminated(self) -> bool:
        """是否已消除或已使用。"""
        return self.state in ('used', 'eliminated')

    def occupy(self) -> None:
        """占据此格子。"""
        self.state = 'occupied'
        self.points = 0

    def eliminate(self) -> None:
        """消除此格子（断连导致）。"""
        self.state = 'eliminated'
        self.points = 0

    def mark_used(self) -> None:
        """标记为已使用（棋子曾到达）。"""
        self.state = 'used'
        self.points = 0

    def activate(self, points: int) -> None:
        """激活格子并设置分值。"""
        self.points = points
        self.state = 'active'


@dataclass(slots=True)
class Piece:
    """
    棋子。ID 决定归属: 偶数→Player1, 奇数→Player2。

    移动后，旧格子被标记为已使用（-1），新格子被占据（0）。
    如果棋子被销毁（从未移动），则恢复原格子的分值。
    """
    id: int
    hex: Optional[Hex] = None
    hex_value: int = 0  # 棋子当前占据格子的原始分值
    alive: bool = True
    _moved: bool = field(default=False)  # 是否已移动过（用于销毁时判断）

    def owner(self) -> int:
        """返回棋子归属: 0=Player1, 1=Player2。"""
        return self.id % 2

    def is_at_hex(self, hex_obj: Hex) -> bool:
        """是否在指定格子上。"""
        return self.alive and self.hex is hex_obj

    def is_mobile(self, valid_moves: list) -> bool:
        """是否有合法移动。"""
        return self.alive and self.hex is not None and len(valid_moves) > 0

    def move_to(self, target_hex: Hex) -> int:
        """
        将棋子移动到目标格子。返回目标格子的分值（用于计分）。

        注意：此方法不检查移动合法性，调用者需确保 move 合法。
        旧格子会被标记为已使用，新格子被占据。
        """
        # 旧格子标记为已使用
        if self.hex is not None:
            self.hex.mark_used()
        # 记录新格子原始分值
        self.hex_value = target_hex.points
        # 占据新格子
        target_hex.occupy()
        # 移动棋子
        self.hex = target_hex
        self._moved = True


# =============================================================================
# 棋盘生成（与 JS generateSequence + createBoard 对齐）
# =============================================================================

def generate_sequence(
    total_sum: int = TOTAL_VALUE,
    hex_count: int = None,
    rng: random.Random | None = None,
) -> List[int]:
    """
    生成随机数列，总和为 total_sum，包含 COUNT_OF_THREE_MIN~MAX 个 3。
    与 JS generateSequence() 完全对齐。

    Args:
        rng: 可选的随机数生成器。如果为 None，则使用全局 random。
    """
    if rng is None:
        rng = random.Random()

    if hex_count is None:
        hex_count = HEX_COUNT

    for _ in range(10000):  # 防死循环
        sequence: List[int] = []
        remaining_sum = total_sum
        remaining_length = hex_count

        # 随机决定 3 的个数
        count3 = rng.randint(COUNT_OF_THREE_MIN, COUNT_OF_THREE_MAX)

        # 填入 3
        for _ in range(count3):
            sequence.append(THREE_VALUE)
            remaining_sum -= THREE_VALUE
            remaining_length -= 1

        # 用 1 和 2 填满剩余位置
        success = True
        while remaining_length > 0:
            next_num = rng.choice([1, 2])
            if remaining_sum - next_num >= 0:
                sequence.append(next_num)
                remaining_sum -= next_num
                remaining_length -= 1
            elif remaining_sum - 2 >= 0:
                sequence.append(2)
                remaining_sum -= 2
                remaining_length -= 1
            else:
                success = False
                break

        if success and sum(sequence) == total_sum and len(sequence) == hex_count:
            # Fisher-Yates 洗牌
            rng.shuffle(sequence)
            return sequence

    raise RuntimeError("generateSequence 死循环，无法生成合法序列")


def create_board(value_sequence: List[int]) -> List[Hex]:
    """
    根据值序列创建棋盘。
    对应 JS createBoard()，包含完全相同的坐标逻辑。

    qAdjustments 键为字符串（与 JS 一致），
    因此 qAdjustments[2] 查不到会当作 0（与 JS 行为一致）。
    """
    # 行范围（对应 JS rowRanges）
    row_ranges = {
        "even": (-4, 3),  # q 为偶数
        "odd": (-3, 3),   # q 为奇数
    }

    hexes: List[Hex] = []
    idx = 0

    for q in range(-4, 4):  # -4, -3, -2, -1, 0, 1, 2, 3
        is_even_row = q % 2 == 0
        start, end = row_ranges["even"] if is_even_row else row_ranges["odd"]

        for r in range(start, end + 1):
            # JS s 约束: if (Math.abs(s) <= radius) 其中 s = q (原始) + r (原始)
            s_raw = q + r
            if abs(s_raw) > 8:
                continue

            adjustment = Q_ADJUSTMENTS.get(str(q), 0)
            adjusted_r = r + adjustment
            # JS: s 用 adjusted_r: Hex(q=adjusted_r, r=-q-adjusted_r, s=q)
            adjusted_s = -q - adjusted_r

            value = value_sequence[idx]
            idx += 1

            # 与 JS: new Hex(adjusted_r, adjusted_s, q_raw, value) 对齐
            hex_obj = Hex(q=adjusted_r, r=adjusted_s, s=q, points=value, state='active', _q_raw=q)
            hexes.append(hex_obj)

    return hexes


def create_board_from_coords(coords: List[dict], value_sequence: List[int]) -> List[Hex]:
    """
    根据自定义坐标列表和值序列创建棋盘。

    Args:
        coords: 格子坐标列表，每个元素可以是 {"q": int, "r": int, "s": int}（无 value）
                或 {"q": int, "r": int, "s": int, "value": int}（带固定 value）
        value_sequence: 对应的值序列，总和必须为 TOTAL_VALUE (99)。
                       如果 coords 中每个元素已包含 value，则此参数被忽略。

    Returns:
        Hex 对象列表
    """
    hexes: List[Hex] = []
    for i, coord in enumerate(coords):
        q = coord["q"]
        r = coord["r"]
        s = coord["s"]
        # 优先使用 coord 内嵌的固定 value（用于自定义固定分数棋盘）
        if "value" in coord:
            value = coord["value"]
        else:
            value = value_sequence[i]
        hex_obj = Hex(q=q, r=r, s=s, points=value, state='active', _q_raw=q)
        hexes.append(hex_obj)
    return hexes


def json_board_to_coords(json_board: List[dict]) -> List[dict]:
    """
    将 JSON 棋盘格式转换为 Python 内部坐标格式。

    JSON 棋盘使用原始立方体坐标 (q, r, s)，需要转换为 Python 内部调整后坐标。

    Args:
        json_board: JSON 棋盘的 hexes 数组，每个元素为 {"q": int, "r": int, "s": int, ...}

    Returns:
        Python 格式的坐标列表，可直接传给 create_board_from_coords()
    """
    coords = []
    for h in json_board:
        raw_q = h["q"]
        raw_r = h["r"]
        # 应用与 create_board() 相同的调整逻辑
        adj = Q_ADJUSTMENTS.get(str(raw_q), 0)
        py_q = raw_r + adj      # adjusted_r
        py_r = -raw_q - py_q    # adjusted_s
        py_s = raw_q            # s 存储原始 q 值
        coords.append({
            "q": py_q,
            "r": py_r,
            "s": py_s,
        })
    return coords


# =============================================================================
# 核心游戏类
# =============================================================================

class PenguinChessCore:
    """
    企鹅棋核心规则引擎。

    游戏分两个阶段：
    1. 放置阶段（placement）：双方轮流放置棋子，Player1 先手，各放 3 个
    2. 移动阶段（movement）：双方轮流移动棋子，Player1 先手

    胜负规则：游戏结束时，分数更高者获胜（分数相同则平局）。
    """

    PHASE_PLACEMENT = "placement"
    PHASE_MOVEMENT = "movement"

    def __init__(self, *, seed: Optional[int] = None, custom_coords: Optional[List[dict]] = None):
        """
        初始化游戏核心。

        Args:
            seed: 随机种子
            custom_coords: 可选，自定义格子坐标列表。
                          如果为 None，使用默认平行四边形 60 格。
                          格式: [{"q": int, "r": int, "s": int}, ...]
        """
        self._rng = random.Random(seed)
        self._seed = seed
        self._custom_coords = custom_coords  # 保存自定义坐标

        self.hexes: List[Hex] = []       # 当前所有格子
        self.pieces: List[Piece] = []    # 所有棋子
        self.players_scores: List[int] = [0, 0]  # [Player1, Player2] 分数

        self.phase: str = self.PHASE_PLACEMENT
        self.current_player: int = 0  # 0 = Player1, 1 = Player2
        self._placement_count: int = 0  # 已放置棋子总数

        self._episode_steps: int = 0
        self._terminated: bool = False

        # 内部状态
        self._hex_map: dict = {}      # (q,r,s) → hex_index，快速查找
        self._occupied_set: set = set()  # 已占据格子的 hex 索引集合，O(1) 查询
        self._neighbors: List[List[int]] = []  # hex_index → [neighbor_indices]，预计算邻居

    def reset(self, *, seed: Optional[int] = None) -> None:
        """初始化/重置游戏到初始状态。"""
        if seed is None:
            seed = self._seed if self._seed is not None else None
        self._rng = random.Random(seed)
        self._seed = seed

        # 使用自定义坐标或默认坐标
        if self._custom_coords is not None:
            # 自定义棋盘：使用提供的坐标
            hex_count = len(self._custom_coords)
            # coords 中每个 hex 带 "value" 字段时直接用它（固定分数棋盘），
            # 否则随机生成值序列
            has_fixed_values = any("value" in c for c in self._custom_coords)
            if has_fixed_values:
                seq = [1] * hex_count  # placeholder，create_board_from_coords 会忽略
            else:
                seq = generate_sequence(rng=self._rng, hex_count=hex_count)
            self.hexes = create_board_from_coords(self._custom_coords, seq)
        else:
            # 默认棋盘：平行四边形
            seq = generate_sequence(rng=self._rng)
            self.hexes = create_board(seq)
        self._build_hex_map()

        self.pieces = []
        for pid in PLAYER_1_PIECES:
            self.pieces.append(Piece(id=pid))
        for pid in PLAYER_2_PIECES:
            self.pieces.append(Piece(id=pid))

        self.players_scores = [0, 0]
        self.phase = self.PHASE_PLACEMENT
        self.current_player = 0
        self._placement_count = 0
        self._episode_steps = 0
        self._terminated = False
        self._rebuild_occupied()

    def get_legal_actions(self) -> List[int]:
        """
        返回当前所有合法动作的 ID 列表。
        动作 ID = hexes 数组索引（0 ~ N-1）。

        放置阶段: 返回所有可放置的空格子索引
        移动阶段: 返回所有己方棋子可移动到的目标格子索引
        """
        if self.phase == self.PHASE_PLACEMENT:
            ids = []
            for i, h in enumerate(self.hexes):
                if h.is_active() and not self._hex_occupied(h):
                    ids.append(i)
            return ids

        # 移动阶段: 收集所有合法的目标格子索引
        ids = set()
        for piece in self.pieces:
            if not piece.alive or piece.hex is None:
                continue
            if self._piece_owner(piece) != self.current_player:
                continue
            moves = self._get_piece_moves(piece)
            for target in moves:
                idx = self._hex_map.get((target.q, target.r, target.s))
                if idx is not None:
                    ids.add(idx)  # O(1) - idx is already the index
        return sorted(ids)

    def step(self, action: int, piece_id: Optional[int] = None) -> Tuple[List, float, bool, dict]:
        """
        执行动作。
        返回 (observation, reward, terminated, info)。
        注意: Python Gymnasium 环境由 env.py 调用，此处返回原生格式。

        piece_id: 可选参数，指定要移动的棋子 ID（移动阶段）。
                 如果提供，后端会验证该棋子是否真的能移动到目标 hex。
                 如果不提供或验证失败，回退到原有逻辑（取第一个能移动到的棋子）。
        """
        if self._terminated:
            return self.get_observation(), 0.0, True, {}

        hex_obj = self._action_id_to_hex(action)
        if hex_obj is None:
            return self.get_observation(), -1.0, self._terminated, {"invalid": True}

        reward = 0.0
        info = {}
        # 保存验证结果：piece_id 是否通过验证
        info["piece_id_valid"] = True

        # 保存当前玩家（在动作执行前）
        acting_player = self.current_player

        if self.phase == self.PHASE_MOVEMENT:
            reward, info = self._do_movement(hex_obj, piece_id=piece_id)
        else:
            # 放置阶段守卫：当 6 个棋子全部放完后，切换到移动阶段并执行移动
            # 对应 JS: pieces.length >= 6 时 resolve() 跳过 handleClick → 进入移动阶段
            total_placed = sum(p.hex is not None for p in self.pieces)
            if total_placed < PIECES_PER_PLAYER * 2:
                # 验证放置动作合法性
                if not hex_obj.is_active() or self._hex_occupied(hex_obj):
                    return self.get_observation(), -1.0, self._terminated, {"invalid": True}
                reward = self._do_placement(hex_obj)
            else:
                self.phase = self.PHASE_MOVEMENT
                reward, info = self._do_movement(hex_obj, piece_id=piece_id)

        self._episode_steps += 1

        # 每回合后：
        # 1. 消除断连格子（不能通过非-1路径连接到任何存活棋子的格子）
        # 2. 棋子离场检查（消除后可能产生新的无路棋子）
        # 注意：放置阶段也可能触发格子消除和棋子销毁
        if not self._terminated:
            self._eliminate_disconnected_hexes()
            # 消除格子后，可能有棋子因断连而失去所有合法移动，再次销毁
            self._destroy_immobile_pieces()

        self._check_game_over()

        # 玩家切换：放置阶段严格交替，双方都放满后才进入移动阶段
        if not self._terminated:
            if self.phase == self.PHASE_PLACEMENT:
                # 放置阶段：严格交替，直到双方都放满 3 个
                P0_placed = sum(
                    1 for p in self.pieces
                    if self._piece_owner(p) == 0 and p.hex
                )
                P1_placed = sum(
                    1 for p in self.pieces
                    if self._piece_owner(p) == 1 and p.hex
                )

                if self.current_player == 0 and P0_placed >= PIECES_PER_PLAYER:
                    # P1 放满，检查是否双方都放满
                    if P1_placed >= PIECES_PER_PLAYER:
                        self.phase = self.PHASE_MOVEMENT
                        self.current_player = 0
                    else:
                        self._switch_player()  # P2 继续
                elif self.current_player == 1 and P1_placed >= PIECES_PER_PLAYER:
                    # P2 放满，检查是否双方都放满
                    if P0_placed >= PIECES_PER_PLAYER:
                        self.phase = self.PHASE_MOVEMENT
                        self.current_player = 0
                    else:
                        self._switch_player()  # P1 继续
                else:
                    # 当前玩家未放满，正常交替到对方
                    self._switch_player()
            else:
                # 移动阶段：正常切换
                self._switch_player()

        obs = self.get_observation()
        return obs, reward, self._terminated, info

    def get_observation(self) -> dict:
        """返回当前观测（Dict 结构）。"""
        return {
            "board": self._encode_board(),
            "pieces": self._encode_pieces(),
            "current_player": self.current_player,
            "phase": 0 if self.phase == self.PHASE_PLACEMENT else 1,
            "scores": list(self.players_scores),
        }

    def get_info(self) -> dict:
        """返回额外信息（用于 RL info dict）。"""
        legal = self.get_legal_actions()
        return {
            "valid_actions": legal,
            "current_player": self.current_player,
            "phase": self.phase,
            "scores": list(self.players_scores),
            "pieces_remaining": [
                self._count_alive_pieces(0),
                self._count_alive_pieces(1),
            ],
            "episode_steps": self._episode_steps,
        }

    def render(self) -> str:
        """返回棋盘文本表示（调试用）。"""
        lines = []
        lines.append(f"Phase: {self.phase}, Player: {self.current_player + 1}")
        lines.append(f"Scores: P1={self.players_scores[0]}, P2={self.players_scores[1]}")
        pieces_info = []
        for p in self.pieces:
            if p.hex:
                pieces_info.append((p.id, p.alive, p.hex.q, p.hex.r))
            else:
                pieces_info.append((p.id, p.alive, None, None))
        lines.append(f"Pieces: {pieces_info}")
        lines.append(f"Episode steps: {self._episode_steps}")
        lines.append(f"Terminated: {self._terminated}")
        active = [h for h in self.hexes if h.is_active()]
        lines.append(f"Active hexes: {len(active)}/60")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # 内部方法：放置阶段
    # -------------------------------------------------------------------------

    def _do_placement(self, hex_obj: Hex) -> float:
        """执行放置动作，返回 reward。"""
        # 守卫：检查当前玩家是否已经放满 3 个棋子
        current_player_placed = sum(
            1 for p in self.pieces
            if self._piece_owner(p) == self.current_player and p.hex is not None
        )
        if current_player_placed >= PIECES_PER_PLAYER:
            raise RuntimeError(
                f"Placement guard: cp={self.current_player} placed={current_player_placed}"
            )

        # 分配棋子：使用 _placement_count 确定当前是第几步放置
        # 放置顺序严格交替: P1(step0) P2(step1) P1(step2) P2(step3) P1(step4) P2(step5)
        # 每个玩家在第 N 次放置时使用第 N 个棋子（即使之前放置的棋子已阵亡）
        player_pieces = PLAYER_1_PIECES if self.current_player == 0 else PLAYER_2_PIECES
        piece_index = self._placement_count // 2
        piece_id = player_pieces[piece_index]
        piece = next(p for p in self.pieces if p.id == piece_id)
        piece.hex = hex_obj
        piece.hex_value = hex_obj.points  # 记录格子原始分值

        # 计分
        score = hex_obj.points
        self.players_scores[self.current_player] += score
        hex_obj.occupy()  # 被占据

        self._rebuild_occupied()  # 更新占据集合缓存
        self._placement_count += 1

        # 6 个棋子放完，进入移动阶段
        if self._placement_count >= PIECES_PER_PLAYER * 2:
            self.phase = self.PHASE_MOVEMENT

        return float(score) / TOTAL_VALUE  # 归一化 reward

    # -------------------------------------------------------------------------
    # 内部方法：移动阶段
    # -------------------------------------------------------------------------

    def _do_movement(self, target_hex: Hex, dry_run: bool = False,
                      piece_id: Optional[int] = None) -> Tuple[float, dict]:
        """
        执行移动动作。
        target_hex: 目标格子（用户选择的动作 ID 对应的 Hex）。
        piece_id: 可选参数，指定要移动的棋子 ID。如果提供，优先使用该棋子；
                  如果该棋子无法移动到目标 hex，返回失败。
        dry_run: True 时只检查不执行。

        Returns: (reward, info)
        """
        info = {}

        # 优先使用用户指定的棋子
        piece = None
        piece_moves = None

        if piece_id is not None:
            # 找到指定棋子
            for p in self.pieces:
                if not p.alive or p.hex is None:
                    continue
                if p.id != piece_id:
                    continue
                if self._piece_owner(p) != self.current_player:
                    # 棋子不属于当前玩家
                    info["piece_id_valid"] = False
                    return -0.5, info
                moves = self._get_piece_moves(p)
                if target_hex in moves:
                    piece = p
                    piece_moves = moves
                else:
                    # 指定棋子无法移动到目标 hex
                    info["piece_id_valid"] = False
                    return -0.5, info
                break

        if piece is None:
            # 未指定 piece_id 或未找到，回退到原有逻辑：取第一个能移动到的棋子
            for p in self.pieces:
                if not p.alive or p.hex is None:
                    continue
                if self._piece_owner(p) != self.current_player:
                    continue
                moves = self._get_piece_moves(p)
                if target_hex in moves:
                    piece = p
                    piece_moves = moves
                    break

        if piece is None:
            # 无合法移动
            return -0.5, info

        if not piece_moves:
            if not dry_run:
                self._destroy_piece(piece)
            return -1.0, info

        # 目标格子的分值（移动到新格子获得该格子分值）
        score_gain = target_hex.points
        if not dry_run:
            self.players_scores[self.current_player] += score_gain
            # 棋子离开后，旧格子标记为已使用，不可再占据
            if piece.hex is not None:
                piece.hex.mark_used()
            # 记录新格子的原始分值（用于未来可能的还原）
            piece.hex_value = target_hex.points
            # 设置新格子为被占据状态
            target_hex.occupy()
            # 移动棋子到新格子
            piece.hex = target_hex

            self._rebuild_occupied()  # 更新占据集合缓存

        reward = float(score_gain) / TOTAL_VALUE
        return reward, info

    def _get_piece_moves(self, piece: Piece) -> List[Hex]:
        """返回某棋子的所有合法目标格子（按 q/r/s 轴同坐标过滤）。"""
        if piece is None or piece.hex is None:
            return []

        cur = piece.hex
        candidates = []

        for h in self.hexes:
            if not h.is_active():
                continue
            if self._hex_occupied(h):
                continue
            # q 轴、r 轴、或 s 轴同坐标
            if not (h.q == cur.q or h.r == cur.r or h.s == cur.s):
                continue
            # 检查路径中间格子
            if not self._path_clear(cur, h):
                continue
            candidates.append(h)

        return candidates

    def _path_clear(self, from_hex: Hex, to_hex: Hex) -> bool:
        """检查从 from_hex 到 to_hex 的路径是否畅通（不含任何棋子/已消除格子）。"""
        dq = to_hex.q - from_hex.q
        dr = to_hex.r - from_hex.r
        ds = to_hex.s - from_hex.s

        steps = max(abs(dq), abs(dr), abs(ds))
        if steps <= 1:
            return True

        sign_q = _sign(dq)
        sign_r = _sign(dr)
        sign_s = _sign(ds)

        for i in range(1, steps):
            # 中间格子使用调整后坐标（与 JS calculatePossibleMoves 一致）
            key_q = from_hex.q + sign_q * i
            key_r = from_hex.r + sign_r * i
            key_s = from_hex.s + sign_s * i
            mid_idx = self._hex_map.get((key_q, key_r, key_s))
            if mid_idx is None:
                return False  # 中间格子不在棋盘上，路径无效
            mid_hex = self.hexes[mid_idx]
            # 中间格子被占据、已使用或已消除 → 阻挡
            if mid_idx in self._occupied_set or not mid_hex.is_active():
                return False

        return True

    # -------------------------------------------------------------------------
    # 内部方法：棋子销毁 & 格子消除
    # -------------------------------------------------------------------------

    def _destroy_piece(self, piece: Piece) -> None:
        """
        移除棋子：格子标记为已使用，不可再移动到。

        与移动后的旧格子处理一致：无论是否移动过，格子都变为 used。
        """
        if piece.hex is not None:
            piece.hex.mark_used()  # 标记为已使用（不可再占据）
            piece.hex = None
        piece.alive = False

    def _destroy_immobile_pieces(self) -> None:
        """
        销毁所有周围没有可移动格子的棋子（多轮迭代直到稳定）。

        判断逻辑：检查棋子周围一圈（6 个邻居）中是否有 active 格子。

        级联效应：棋子 A 被销毁后，它占据的格子变为 used，可能导致：
        1. 其他棋子的邻居格子突然可用（不再被 A 阻挡）
        2. 某些格子断连被 eliminate，导致新的死子
        因此需要多轮迭代直到没有新棋子被销毁。
        """
        changed = True
        while changed:
            changed = False
            for piece in self.pieces:
                if not piece.alive or piece.hex is None:
                    continue
                # 获取棋子所在格子的索引
                hex_idx = self._hex_map.get((piece.hex.q, piece.hex.r, piece.hex.s))
                if hex_idx is None:
                    self._destroy_piece(piece)
                    changed = True
                    continue
                # 检查邻居中是否有 active 格子
                has_valid_move = False
                for neighbor_idx in self._neighbors[hex_idx]:
                    if self.hexes[neighbor_idx].is_active():
                        has_valid_move = True
                        break
                if not has_valid_move:
                    self._destroy_piece(piece)
                    changed = True

    def _eliminate_disconnected_hexes(self) -> int:
        """
        消除所有不与任何棋子连通的格子。
        返回消除的格子数量。

        连通性定义：从棋子位置出发，通过相邻的 active 或 occupied 格子可以到达的格子都是连通的。
        eliminated 或 used 格子不可通过。
        """
        connected: set = set()

        def flood_fill(start_idx: int) -> None:
            """从给定索引开始 flood fill，标记所有连通的 hex 索引。"""
            stack = [start_idx]
            while stack:
                idx = stack.pop()
                if idx in connected:
                    continue
                h = self.hexes[idx]
                # 只允许遍历 active 或 occupied 格子（排除 eliminated 和 used）
                if h.state in ('eliminated', 'used'):
                    continue
                connected.add(idx)
                for neighbor_idx in self._neighbors[idx]:
                    if neighbor_idx not in connected:
                        neighbor = self.hexes[neighbor_idx]
                        if neighbor.state in ('active', 'occupied'):
                            stack.append(neighbor_idx)

        # 从所有存活棋子的位置开始 flood fill
        for piece in self.pieces:
            if piece.alive and piece.hex is not None:
                idx = self._hex_map.get((piece.hex.q, piece.hex.r, piece.hex.s))
                if idx is not None:
                    flood_fill(idx)

        eliminated = 0
        for idx, h in enumerate(self.hexes):
            # 格子满足以下条件时会被"消除"（eliminate）：
            # - 活跃格子（is_active）
            # - 无法通过 active/occupied 格子路径连接到任何存活棋子
            if h.is_active() and idx not in connected:
                h.eliminate()
                eliminated += 1

        return eliminated

    # -------------------------------------------------------------------------
    # 内部方法：回合切换 & 游戏结束检查
    # -------------------------------------------------------------------------

    def _switch_player(self) -> None:
        """切换当前玩家。"""
        if not self._terminated:
            self.current_player = 1 - self.current_player

    def _check_game_over(self) -> None:
        """检查游戏是否结束，更新 self._terminated。"""
        p1_alive = self._count_alive_pieces(0)
        p2_alive = self._count_alive_pieces(1)
        # 情况1: 一方棋子全灭
        if not p1_alive or not p2_alive:
            survivor = 1 if not p1_alive else 0
            # 幸存者获得所有剩余格子
            for h in self.hexes:
                if h.is_active():
                    self.players_scores[survivor] += h.points
                    h.mark_used()
            self._terminated = True
            return

        # 情况2: 所有活跃格子都被消除（双方都有棋子但无路可走）
        has_active = any(h.is_active() for h in self.hexes)
        if not has_active:
            self._terminated = True
            return

    # -------------------------------------------------------------------------
    # 内部方法：辅助函数
    # -------------------------------------------------------------------------

    def _build_hex_map(self) -> None:
        """构建 (q,r,s) → hex_index 映射表，并预计算邻居关系。"""
        self._hex_map = {}
        self._neighbors = [[] for _ in range(len(self.hexes))]
        for idx, h in enumerate(self.hexes):
            self._hex_map[(h.q, h.r, h.s)] = idx

        # 预计算每个 hex 的邻居索引
        for idx, h in enumerate(self.hexes):
            # 将 adjusted 坐标转换为 raw 坐标
            raw_q = h.s
            raw_r = h.q - Q_ADJUSTMENTS.get(str(raw_q), 0)

            for dq, dr, ds in HEX_DIRECTIONS:
                # 在 raw 坐标空间添加方向偏移
                new_raw_q = raw_q + dq
                new_raw_r = raw_r + dr

                # 将 raw 坐标转换回 adjusted 坐标
                adj = Q_ADJUSTMENTS.get(str(new_raw_q), 0)
                key_q = new_raw_r + adj
                key_r = -new_raw_q - key_q
                key_s = new_raw_q

                neighbor_idx = self._hex_map.get((key_q, key_r, key_s))
                if neighbor_idx is not None:
                    self._neighbors[idx].append(neighbor_idx)

    def _hex_is_legal_target(self, hex_obj: Hex) -> bool:
        """格子是否可作为动作目标（任何用途）。"""
        return hex_obj.is_active()

    def _hex_occupied(self, hex_obj: Hex) -> bool:
        """格子上是否有棋子。O(1) 查询。"""
        idx = self._hex_map.get((hex_obj.q, hex_obj.r, hex_obj.s))
        if idx is None:
            return False
        return idx in self._occupied_set

    def _rebuild_occupied(self) -> None:
        """重建占据格子索引集合。从棋子状态重建，O(pieces)。"""
        self._occupied_set = set()
        for p in self.pieces:
            if p.alive and p.hex is not None:
                idx = self._hex_map.get((p.hex.q, p.hex.r, p.hex.s))
                if idx is not None:
                    self._occupied_set.add(idx)

    def _hex_has_player_piece(self, hex_obj: Hex, player_idx: int) -> bool:
        """格子上是否有指定玩家的棋子。"""
        for p in self.pieces:
            if p.alive and p.hex is hex_obj and self._piece_owner(p) == player_idx:
                return True
        return False

    def _get_piece_at_hex(self, hex_obj: Hex) -> Optional[Piece]:
        """获取指定格子上的棋子。"""
        for p in self.pieces:
            if p.alive and p.hex is hex_obj:
                return p
        return None

    def _piece_owner(self, piece: Piece) -> int:
        """棋子归属: 0=Player1 (id 4,6,8), 1=Player2 (id 5,7,9)。"""
        return piece.id % 2

    def _count_alive_pieces(self, player_idx: int) -> int:
        """统计指定玩家存活棋子数。"""
        return sum(1 for p in self.pieces if p.alive and self._piece_owner(p) == player_idx)

    def _action_id_to_hex(self, action_id: int) -> Optional[Hex]:
        """动作 ID → Hex 对象。"""
        if 0 <= action_id < len(self.hexes):
            return self.hexes[action_id]
        return None

    def _hex_to_index(self, hex_obj: Hex) -> Optional[int]:
        """Hex 对象 → 动作 ID（数组索引）。"""
        try:
            return self.hexes.index(hex_obj)
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    # 观测编码
    # -------------------------------------------------------------------------

    def _encode_board(self) -> list:
        """
        编码棋盘状态为扁平 list。
        每格: [q, r, points]（归一化）
        """
        result = []
        for h in self.hexes:
            result.append([
                float(h.q) / 8.0,
                float(h.r) / 8.0,
                float(h.points) / 3.0 if h.is_active() else 0.0,
            ])
        return result

    def _encode_pieces(self) -> list:
        """
        编码棋子状态。
        每棋子: [piece_id/10, q/8, r/8, s/8]（归一化）
        棋子被移除时坐标为 0, id 为 -1。
        """
        result = []
        for p in self.pieces:
            if p.alive and p.hex is not None:
                result.append([
                    float(p.id) / 10.0,
                    float(p.hex.q) / 8.0,
                    float(p.hex.r) / 8.0,
                    float(p.hex.s) / 8.0,
                ])
            else:
                result.append([-1.0, 0.0, 0.0, 0.0])
        return result


# =============================================================================
# 工具函数
# =============================================================================

def _sign(x: int) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0
