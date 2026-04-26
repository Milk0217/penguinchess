"""
PenguinChess AI 模块。
包含 MCTS 搜索、AlphaZero 网络和训练工具。
"""
from penguinchess.ai.mcts_core import (
    mcts_search,
    mcts_search_batched,
    mcts_search_parallel,
    MCTSNode,
    select_action,
)
from penguinchess.ai.alphazero_net import AlphaZeroNet
