"""
Reward shaping 函数。
提供稀疏 reward（胜负）和密集 reward（分值变化、棋子消除）两种模式。
"""

from __future__ import annotations
from dataclasses import dataclass


# =============================================================================
# Reward 配置
# =============================================================================

TOTAL_VALUE = 99  # 棋盘总分 99（与 core.py 保持一致）


@dataclass
class RewardConfig:
    """Reward shaping 配置。"""
    # 稀疏 reward
    win_reward: float = 1.0
    loss_reward: float = -1.0
    draw_reward: float = 0.0

    # 密集 reward（分值变化）
    score_change_weight: float = 0.0  # 分数变化是否计入 reward（0=关闭, 1=全额）

    # 棋子消除
    opponent_piece_eliminated: float = 0.0  # 对手棋子被移除
    own_piece_eliminated: float = 0.0  # 己方棋子被移除

    # 中间奖励
    hex_eliminated_bonus: float = 0.0  # 消除一个格子

    # 最大步数惩罚（truncation）
    step_penalty: float = 0.0  # 每步附加惩罚


# =============================================================================
# Reward 函数
# =============================================================================

def compute_reward(
    prev_scores: list[int],
    curr_scores: list[int],
    prev_pieces_alive: tuple[int, int],
    curr_pieces_alive: tuple[int, int],
    game_over: bool,
    winner: int | None,
    cfg: RewardConfig | None = None,
) -> tuple[float, float]:
    """
    计算当前玩家的 reward。

    Args:
        prev_scores:      前一步双方分数 [p1, p2]
        curr_scores:      当前双方分数 [p1, p2]
        prev_pieces_alive: 前一步存活棋子数 (p1_count, p2_count)
        curr_pieces_alive: 当前存活棋子数 (p1_count, p2_count)
        game_over:        游戏是否结束
        winner:           胜者: 0=P1, 1=P2, 2=平局, None=未结束
        cfg:              Reward shaping 配置

    Returns:
        (reward, info) where info is a dict with reward breakdown.
    """
    if cfg is None:
        cfg = RewardConfig()

    current_player_idx = 0  # 固定为 Player1 的视角
    reward = 0.0
    info = {}

    # 1. 分数变化
    if cfg.score_change_weight > 0:
        delta = curr_scores[current_player_idx] - prev_scores[current_player_idx]
        reward += delta / TOTAL_VALUE * cfg.score_change_weight
        info["score_change"] = delta

    # 2. 棋子消除
    if cfg.opponent_piece_eliminated != 0:
        opp_count_prev = prev_pieces_alive[1 - current_player_idx]
        opp_count_curr = curr_pieces_alive[1 - current_player_idx]
        if opp_count_curr < opp_count_prev:
            reward += cfg.opponent_piece_eliminated
            info["opp_eliminated"] = opp_count_prev - opp_count_curr

    if cfg.own_piece_eliminated != 0:
        own_prev = prev_pieces_alive[current_player_idx]
        own_curr = curr_pieces_alive[current_player_idx]
        if own_curr < own_prev:
            reward += cfg.own_piece_eliminated
            info["own_eliminated"] = own_prev - own_curr

    # 3. 游戏结束 reward
    if game_over:
        if winner == current_player_idx:
            reward += cfg.win_reward
            info["result"] = "win"
        elif winner == 2:  # 平局
            reward += cfg.draw_reward
            info["result"] = "draw"
        else:
            reward += cfg.loss_reward
            info["result"] = "loss"

    info["total"] = reward
    return reward, info


def sparse_reward(game_over: bool, winner: int | None) -> tuple[float, dict]:
    """最简单的稀疏 reward: 只有游戏结束时返回胜负。"""
    cfg = RewardConfig(win_reward=1.0, loss_reward=-1.0, draw_reward=0.0)
    return compute_reward([0, 0], [0, 0], (0, 0), (0, 0), game_over, winner, cfg)


def dense_reward(
    prev_scores: list[int],
    curr_scores: list[int],
    prev_pieces: tuple[int, int],
    curr_pieces: tuple[int, int],
    game_over: bool,
    winner: int | None,
) -> tuple[float, dict]:
    """密集 reward: 分数变化 + 棋子消除 + 胜负。"""
    cfg = RewardConfig(
        score_change_weight=1.0,
        opponent_piece_eliminated=2.0,
        own_piece_eliminated=-2.0,
        win_reward=10.0,
        loss_reward=-10.0,
        draw_reward=0.0,
    )
    return compute_reward(prev_scores, curr_scores, prev_pieces, curr_pieces, game_over, winner, cfg)
