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

TOTAL_VALUE = 99
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

@dataclass
class Hex:
    """
    六边形格子。坐标使用立方体坐标 (q, r, s)，满足 q + r + s = 0。

    q/r/s 存储的是经过 row_range 调整后的值（与 JS createBoard 一致），
    额外记录 _q_raw = 原始 q 值（用于邻居查找中对齐 q 轴）。
    """
    q: int       # adjusted_r: r + qAdjustments[q_raw]
    r: int       # adjusted_s: -q_raw - adjusted_r
    s: int       # s_raw: q_raw + r_raw（原始 q + 原始 r，用于合法性检查）
    value: int   # 格子分值: 1, 2, 3（活跃）或 0（被占据）或 -1（已消除）
    _q_raw: int = field(default=0)  # 原始 q 值，用于邻居 q 轴偏移

    def __hash__(self):
        return hash((self.q, self.r, self.s))

    def __eq__(self, other):
        if not isinstance(other, Hex):
            return False
        return self.q == other.q and self.r == other.r and self.s == other.s


@dataclass
class Piece:
    """棋子。ID 决定归属: 偶数→Player1, 奇数→Player2。"""
    id: int
    hex: Optional[Hex] = None
    hex_value: int = 0  # 棋子占据格子前的原始分值（用于还原）
    alive: bool = True


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
    # 键为字符串，与 JS 的 qAdjustments 行为一致
    q_adjustments = {
        "-4": 2, "-3": 1, "-2": 0, "-1": 0,
        "0": 0, "1": -1, "2": -2, "3": -2,
    }

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

            adjustment = q_adjustments.get(str(q), 0)
            adjusted_r = r + adjustment
            # JS: s 用 adjusted_r: Hex(q=adjusted_r, r=-q-adjusted_r, s=q)
            adjusted_s = -q - adjusted_r

            value = value_sequence[idx]
            idx += 1

            # 与 JS: new Hex(adjusted_r, adjusted_s, q_raw, value) 对齐
            hex_obj = Hex(q=adjusted_r, r=adjusted_s, s=q, value=value, _q_raw=q)
            hexes.append(hex_obj)

    return hexes


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

    def __init__(self, *, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._seed = seed

        self.hexes: List[Hex] = []       # 当前所有格子
        self.pieces: List[Piece] = []    # 所有棋子
        self.players_scores: List[int] = [0, 0]  # [Player1, Player2] 分数

        self.phase: str = self.PHASE_PLACEMENT
        self.current_player: int = 0  # 0 = Player1, 1 = Player2
        self._placement_count: int = 0  # 已放置棋子总数

        self._episode_steps: int = 0
        self._terminated: bool = False

        # 内部状态
        self._hex_map: dict = {}  # (q,r,s) → Hex，快速查找

    # -------------------------------------------------------------------------
    # 公共 API（与 Gymnasium 接口对齐）
    # -------------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None) -> None:
        """初始化/重置游戏到初始状态。"""
        if seed is None:
            seed = self._seed if self._seed is not None else None
        self._rng = random.Random(seed)
        self._seed = seed

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
                if h.value > 0 and not self._hex_occupied(h):
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
                    ids.add(self.hexes.index(target))
        return sorted(ids)

    def step(self, action: int) -> Tuple[List, float, bool, dict]:
        """
        执行动作。
        返回 (observation, reward, terminated, info)。
        注意: Python Gymnasium 环境由 env.py 调用，此处返回原生格式。
        """
        if self._terminated:
            return self.get_observation(), 0.0, True, {}

        hex_obj = self._action_id_to_hex(action)
        if hex_obj is None:
            return self.get_observation(), -1.0, self._terminated, {"invalid": True}

        reward = 0.0
        info = {}

        # 保存当前玩家（在动作执行前）
        acting_player = self.current_player

        if self.phase == self.PHASE_MOVEMENT:
            reward, info = self._do_movement(hex_obj)
        else:
            # 放置阶段守卫：当 6 个棋子全部放完后，切换到移动阶段并执行移动
            # 对应 JS: pieces.length >= 6 时 resolve() 跳过 handleClick → 进入移动阶段
            total_placed = sum(p.hex is not None for p in self.pieces)
            if total_placed < PIECES_PER_PLAYER * 2:
                reward = self._do_placement(hex_obj)
            else:
                self.phase = self.PHASE_MOVEMENT
                reward, info = self._do_movement(hex_obj)

        self._episode_steps += 1

        # 移动阶段每回合后执行连通性消除，再检查游戏是否结束
        if self.phase == self.PHASE_MOVEMENT and not self._terminated:
            self._eliminate_disconnected_hexes()

        self._check_game_over()

        # 玩家切换：任一玩家放满 3 个时跳过切换 → 放满者先进入移动阶段
        if not self._terminated:
            P0_placed = sum(1 for p in self.pieces if self._piece_owner(p) == 0 and p.hex)
            P1_placed = sum(1 for p in self.pieces if self._piece_owner(p) == 1 and p.hex)
            if not (P0_placed >= PIECES_PER_PLAYER and P1_placed >= PIECES_PER_PLAYER):
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
        active = [h for h in self.hexes if h.value > 0]
        lines.append(f"Active hexes: {len(active)}/60")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # 内部方法：放置阶段
    # -------------------------------------------------------------------------

    def _do_placement(self, hex_obj: Hex) -> float:
        """执行放置动作，返回 reward。"""
        # JS 守卫: pieces.length >= 6 后直接 resolve()，不再处理 place 动作
        current_player_placed = sum(
            1 for p in self.pieces
            if self._piece_owner(p) == self.current_player and p.hex is not None
        )
        if current_player_placed >= PIECES_PER_PLAYER:
            # 棋子放满了，拒绝放置（JS 会提前在 handleClick 里 resolve 跳过这里）
            raise RuntimeError(
                f"Placement guard: cp={self.current_player} placed={current_player_placed}"
            )

        # 分配棋子
        player_pieces = PLAYER_1_PIECES if self.current_player == 0 else PLAYER_2_PIECES
        piece_id = player_pieces[current_player_placed]

        piece = next(p for p in self.pieces if p.id == piece_id)
        piece.hex = hex_obj
        piece.hex_value = hex_obj.value  # 记录格子原始分值

        # 计分
        score = hex_obj.value
        self.players_scores[self.current_player] += score
        hex_obj.value = 0  # 被占据

        self._placement_count += 1

        # 6 个棋子放完，进入移动阶段
        if self._placement_count >= PIECES_PER_PLAYER * 2:
            self.phase = self.PHASE_MOVEMENT

        return float(score) / TOTAL_VALUE  # 归一化 reward

    # -------------------------------------------------------------------------
    # 内部方法：移动阶段
    # -------------------------------------------------------------------------

    def _do_movement(self, target_hex: Hex, dry_run: bool = False) -> Tuple[float, dict]:
        """
        执行移动动作。
        target_hex: 目标格子（用户选择的动作 ID 对应的 Hex）。
        找到己方棋子能移动到该格子的，验证并执行移动。

        Returns: (reward, info)
        """
        info = {}

        # 找到己方能移动到 target_hex 的棋子
        piece = None
        for p in self.pieces:
            if not p.alive or p.hex is None:
                continue
            if self._piece_owner(p) != self.current_player:
                continue
            # 检查该棋子是否可以将 target_hex 作为合法移动目标
            moves = self._get_piece_moves(p)
            if target_hex in moves:
                piece = p
                break

        if piece is None:
            # 无合法移动
            return -0.5, info

        moves = self._get_piece_moves(piece)
        if not moves:
            if not dry_run:
                self._destroy_piece(piece)
            return -1.0, info

        # 目标格子的分值（移动到新格子获得该格子分值）
        score_gain = target_hex.value
        if not dry_run:
            old_hex_value = piece.hex_value
            self.players_scores[self.current_player] += score_gain
            # 还原旧格子原分值
            if piece.hex is not None:
                piece.hex.value = piece.hex_value
            # 设置新格子为被占据状态（value=0）
            target_hex.value = 0
            piece.hex_value = old_hex_value
            piece.hex = None

        reward = float(score_gain) / TOTAL_VALUE
        return reward, info

    def _get_piece_moves(self, piece: Piece) -> List[Hex]:
        """返回某棋子的所有合法目标格子（按 q/r/s 轴同坐标过滤）。"""
        if piece is None or piece.hex is None:
            return []

        cur = piece.hex
        candidates = []

        for h in self.hexes:
            if h.value <= 0:
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
        """检查从 from_hex 到 to_hex 的路径是否畅通（不含任何棋子）。"""
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
            # q 轴偏移用 _q_raw，其他轴用存储值
            key_q = from_hex.q + sign_q * i  # 沿 q 轴移动
            key_r = from_hex.r + sign_r * i
            key_s = from_hex.s + sign_s * i
            # 若沿 q 轴移动，需要用 _q_raw 对齐
            if dq != 0:
                key_q = from_hex._q_raw + sign_q * i
            mid = self._hex_map.get((key_q, key_r, key_s))
            if mid is None:
                continue  # 棋盘外，跳过
            if self._hex_occupied(mid):
                return False

        return True

    # -------------------------------------------------------------------------
    # 内部方法：棋子销毁 & 格子消除
    # -------------------------------------------------------------------------

    def _destroy_piece(self, piece: Piece) -> None:
        """移除棋子：还原格子原值，清除棋子引用。"""
        if piece.hex is not None:
            piece.hex.value = piece.hex_value  # 还原格子原分值
            piece.hex = None
        piece.alive = False

    def _eliminate_disconnected_hexes(self) -> int:
        """
        消除所有不与任何棋子连通的格子。
        返回消除的格子数量。
        """
        connected: set = set()

        def flood_fill(hex_obj: Hex) -> None:
            stack = [hex_obj]
            while stack:
                h = stack.pop()
                if h in connected or h.value < 0:
                    continue
                connected.add(h)
                for dq, dr, ds in HEX_DIRECTIONS:
                    # q 轴邻居用 _q_raw 对齐，其他轴用存储坐标
                    key_q = h.q + dq if dq == 0 else h._q_raw + dq
                    key_r = h.r + dr
                    key_s = h.s + ds
                    neighbor = self._hex_map.get((key_q, key_r, key_s))
                    if neighbor and neighbor not in connected and neighbor.value > 0:
                        stack.append(neighbor)

        # 从所有存活棋子的位置开始 flood fill
        for piece in self.pieces:
            if piece.alive and piece.hex is not None:
                flood_fill(piece.hex)

        eliminated = 0
        for h in self.hexes:
            if h.value > 0 and h not in connected:
                h.value = -1
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
        has_active = any(h.value > 0 for h in self.hexes)

        # 情况1: 一方棋子全灭
        if not p1_alive or not p2_alive:
            survivor = 1 if not p1_alive else 0
            # 幸存者获得所有剩余格子
            for h in self.hexes:
                if h.value > 0:
                    self.players_scores[survivor] += h.value
                    h.value = -1
            self._terminated = True
            return

        # 情况2: 所有活跃格子都被消除（双方都有棋子但无路可走）
        has_active = any(h.value > 0 for h in self.hexes)
        if not has_active:
            self._terminated = True
            return

    # -------------------------------------------------------------------------
    # 内部方法：辅助函数
    # -------------------------------------------------------------------------

    def _build_hex_map(self) -> None:
        """构建 (q,r,s) → Hex 映射表。"""
        self._hex_map = {}
        for h in self.hexes:
            self._hex_map[(h.q, h.r, h.s)] = h

    def _hex_is_legal_target(self, hex_obj: Hex) -> bool:
        """格子是否可作为动作目标（任何用途）。"""
        return hex_obj.value >= 0  # value > 0（活跃）或 == 0（被占据但可作为选择）

    def _hex_occupied(self, hex_obj: Hex) -> bool:
        """格子上是否有棋子。"""
        return any(p.alive and p.hex is hex_obj for p in self.pieces)

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

    # -------------------------------------------------------------------------
    # 观测编码
    # -------------------------------------------------------------------------

    def _encode_board(self) -> list:
        """
        编码棋盘状态为扁平 list。
        每格: [q, r, value]（归一化）
        """
        result = []
        for h in self.hexes:
            result.append([
                float(h.q) / 8.0,
                float(h.r) / 8.0,
                float(max(h.value, 0)) / 3.0,
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
