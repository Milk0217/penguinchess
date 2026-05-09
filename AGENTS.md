# PenguinChess — Agent 开发指南

## 一、项目目标

本项目有三个核心方向：

1. **Gymnasium 环境** — 构建严格遵循 Farama Gymnasium 接口的企鹅棋 RL 环境
2. **强 AI 训练** — 基于该环境训练高胜率 RL 智能体
3. **高平局率 AI** — 在达成高胜率后，进一步训练追求高平局率的策略 AI（防守型/和棋型 AI）

---

## 二、架构概览

```
penguinchess/
├── game_engine/                # Rust 引擎 (搜索+推理)
│   ├── src/
│   │   ├── board.rs            # 棋盘数据结构
│   │   ├── rules.rs            # 游戏规则
│   │   ├── mcts_rs.rs          # MCTS 搜索
│   │   ├── alphabeta_rs.rs     # Alpha-Beta 搜索 (PVS+TT+LMR+空着)
│   │   ├── nnue_rs.rs          # NNUE 推理 (AVX2)
│   │   ├── nnue_train.rs       # NNUE 训练 (buggy Adam)
│   │   ├── az_model.rs         # AlphaZero/PPO 推理
│   │   ├── nnue_candle.rs      # Candle 训练 (正确 Adam)
│   │   ├── ffi.rs              # C FFI 接口
│   │   └── lib.rs              # 模块入口
│   └── Cargo.toml
│
├── ab_nnue/                    # AB+NNUE 子系统
│   ├── train.py                # Rust 训练周期 (buggy Adam)
│   ├── pt_train.py             # PyTorch 微调 (正确 Adam)
│   ├── pt_scratch.py           # PyTorch scratch 训练
│   ├── pt_ft_full.py           # PyTorch 全参数微调
│   ├── ensemble.py             # 集成训练
│   ├── evaluate.py             # vs PPO/随机评估
│   ├── random_gen.py           # 随机数据生成
│   ├── rust_gen.py             # Rust 数据生成
│   ├── bench.py                # 模型对比
│   ├── nnue_fast.py            # 快速 NNUE 训练
│   ├── nnue_selfplay.py        # AB 自对弈训练
│   ├── nnue_mcts.py            # MCTS+NNUE 训练
│   └── nnue_rust.py            # Rust NNUE 训练
│
├── alphazero/                  # AlphaZero 子系统
│   ├── train.py                # AZ ResNet/MLP 训练
│   ├── evaluate.py             # ELO 评估
│   ├── random_ai.py            # 随机 AI 基准
│   ├── train_pipeline.py       # 训练管线
│   ├── train.cmd               # 训练脚本
│   ├── train_bg.ps1            # 后台训练
│   └── train_bg.bat            # 后台训练
│
├── ppo/                        # PPO 子系统
│   └── train.py                # PPO 训练
│
├── penguinchess/               # Python 游戏核心
│   ├── core.py                 # 游戏规则
│   ├── env.py                  # Gymnasium 环境
│   ├── ai/                     # AI 模型
│   │   ├── nnue.py             # NNUE 架构
│   │   ├── nnue_train.py       # NNUE 数据加载
│   │   ├── sparse_features.py  # 特征提取
│   │   ├── alphazero_net.py    # AZ 网络
│   │   └── nnue_agent.py       # NNUE Agent
│   └── rust_ffi.py             # Rust FFI 包装
│
├── models/
│   ├── ab_nnue/                # AB+NNUE 模型
│   ├── alphazero/              # AlphaZero 模型
│   └── ppo/                    # PPO 模型
│
├── data/                       # 训练数据 (bin)
├── server/                     # Flask HTTP
├── frontend/                   # React 前端
├── docs/                       # 文档
└── backend_data/               # 棋盘数据

### 12.0 技术文档约束（强制）

> **引入任何新技术或算法前，必须先补充 `docs/TECHNICAL_REFERENCE.md` 相关章节。**
> 内容包括：算法原理、架构图、参数说明、代码位置、参考链接。
