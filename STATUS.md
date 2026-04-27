# PenguinChess 项目状态文档

> 最后更新: 2026-04-27

## 项目概述

PenguinChess（企鹅棋）是一个双人对战六边形棋盘游戏，同时作为强化学习研究平台。项目包含：

1. **Rust 游戏引擎** — 有状态 FFI，11.5x 加速
2. **Gymnasium RL 环境** — 严格遵循 Farama Gymnasium 标准
3. **PPO 训练** — 10 个世代，最高 ELO 1210
4. **AlphaZero 自对弈训练** — MCTS + ResNet，GPU 加速
5. **ELO 评估系统** — 增量 Round-Robin，并行 workers
6. **Model Registry** — ELO 持久化，自动选择最优模型
7. **Web 人机对战** — React + Flask

---

## 一、当前实现状态

### ✅ 已完成

#### 1.1 Rust 游戏引擎 (`game_engine/`)

| 模块 | 文件 | 状态 |
|------|------|------|
| 棋盘数据结构 | `board.rs` | ✅ 完成 |
| 游戏规则引擎 | `rules.rs` (273 行) | ✅ 完成 |
| C FFI 导出 | `ffi.rs` | ✅ 完成 |
| 有状态游戏 API | `game_stateful_*` (144x 加速) | ✅ 完成 |
| MCTS 搜索 | `mcts_rs.rs` | ✅ 完成 |
| 根节点 Dirichlet 噪声 | `mcts_rs.rs` | ✅ 完成 |
| 随机 UCB 平局打断 | `mcts_rs.rs` | ✅ 完成 |
| ONNX 推理占位 | `net_infer.rs` | ✅ 架构预留 |

**性能**: Rust release 构建单步 0.014ms (Python 0.159ms 的 **11.5x**)

#### 1.2 Python FFI 桥接

| 文件 | 说明 |
|------|------|
| `rust_ffi.py` | ctypes 桥接 + `RustStatefulGame` |
| `rust_core.py` | `RustCore` duck-type 兼容 `PenguinChessCore` |

#### 1.3 Python 游戏核心 (`penguinchess/`)

| 文件 | 说明 |
|------|------|
| `core.py` | PenguinChessCore 游戏逻辑（双核心可选） |
| `env.py` | Gymnasium Env，暴露 `.core` 属性 |
| `spaces.py` | 观测/动作空间 |
| `reward.py` | Reward shaping |
| `random_ai.py` | 随机 AI 基准 |
| `model_registry.py` | ELO 持久化 JSON 注册表 |

#### 1.4 AlphaZero 训练 (`penguinchess/ai/`)

| 文件 | 说明 |
|------|------|
| `alphazero_net.py` | `AlphaZeroNet` (MLP) + `AlphaZeroResNet` (残差 MLP) |
| `train_alphazero.py` | 自对弈训练循环 + `best_net` 跟踪 |
| `mcts_core.py` | Python MCTS + 批处理 + 根并行 |

**训练特性**:
- Rust MCTS 自对弈（22x 加速 via 零拷贝观测）
- GPU AMP 混合精度
- 非阻塞 CUDA 传输
- flat obs 存储（零 CPU 解码）
- ResourceMonitor（GPU/CPU/RAM）
- Dirichlet 噪声 + 随机 UCB 平局打断
- ResNet 残差网络（自动检测架构）
- 每代 45+ 独特动作探索（原 1）

#### 1.5 PPO 训练 (`examples/`)

| 文件 | 说明 |
|------|------|
| `train_ppo.py` | 完整 PPO 训练 + ELO 评估 |
| `eval_elo.py` | 增量/全量 ELO，Python/Rust 引擎，PPO/AZ/MCTS Agent |

**ELO 特性**:
- 自动交换先后手消除偏差
- `--workers N` 并行评估
- `--incremental` 增量模式
- `--gpu` GPU 推理
- `--python-core` 回退 Python 引擎
- `--stochastic` PPO 随机采样
- `--mcts` Rust MCTS 评估

#### 1.6 Web 后端 (Flask)

| 文件 | 说明 |
|------|------|
| `server/app.py` | HTTP 路由、会话管理 |
| `server/game.py` | Game 会话封装 |
| `server/boards.py` | 棋盘存储 |
| `server/ai_player.py` | AI 玩家（PPO + AlphaZero，自动 ELO 选最优） |

**API 端点**:
- `POST /api/game` — 创建新游戏
- `GET /api/game/<id>` — 获取游戏状态
- `POST /api/game/<id>/action` — 提交动作
- `POST /api/game/<id>/ai_move` — AI 自动移动
- `POST /api/game/<id>/reset` — 重开一局
- `GET /api/models` — 获取所有模型及 ELO
- `GET /api/models/best` — 获取最优模型
- `GET /api/boards` — 已保存棋盘列表

#### 1.7 Web 前端 (React)

| 组件 | 状态 |
|------|------|
| `App.tsx` | ✅ 完成（含 AI 模型信息显示） |
| `board/BoardContainer.tsx` | ✅ 完成 |
| `board/HexCell.tsx` | ✅ 完成 |
| `board/Piece.tsx` | ✅ 完成 |
| `api.ts` | ✅ 完成（含模型 API） |
| 棋盘编辑器 | ✅ 完成 |

---

## 二、最新 ELO 排名（2026-04-27）

```
  排名  模型          ELO    vs随机
--------------------------------------
   1   ppo_gen_5    1210   50%
   2   ppo_gen_6    1210   51%
   3   ppo_gen_7    1210   53%
   4   ppo_gen_2    1210   52%
   5   ppo_gen_9    1206   50%
   6   ppo_gen_8    1205   51%
   7   ppo_gen_10   1204   48%
   8   ppo_gen_1    1203   50%
   9   ppo_gen_4    1201   45%
  10   ppo_gen_3    1200   46%
  11   az_best      1172   37%
  12   az_iter_10   1168   35%
```

**评估方法**: 1000 局/对，交换先后手，4 workers 并行，Rust 引擎。

---

## 三、项目结构

```
penguinchess/
├── game_engine/              # Rust cdylib 库
│   ├── Cargo.toml
│   └── src/
│       ├── board.rs          # 棋盘数据结构
│       ├── rules.rs          # 游戏规则引擎
│       ├── ffi.rs            # C FFI 导出
│       ├── mcts_rs.rs        # Rust MCTS 搜索
│       ├── net_infer.rs      # ONNX 推理（预留）
│       └── lib.rs
│
├── penguinchess/             # Python 核心
│   ├── core.py               # 游戏核心
│   ├── env.py                # Gymnasium 环境
│   ├── spaces.py             # Space 定义
│   ├── reward.py             # Reward 函数
│   ├── random_ai.py          # 随机 AI
│   ├── model_registry.py    # ELO 注册表
│   ├── rust_ffi.py           # Rust ctypes 桥接
│   ├── rust_core.py          # RustCore 包装
│   └── ai/
│       ├── alphazero_net.py  # AlphaZero 网络
│       ├── train_alphazero.py # 自对弈训练
│       ├── mcts_core.py      # Python MCTS
│       └── __init__.py
│
├── server/                    # Flask 后端
│   ├── app.py                # HTTP 路由
│   ├── game.py               # Game 会话
│   ├── ai_player.py          # AI 玩家
│   └── boards.py             # 棋盘存储
│
├── frontend/                  # React 前端
│   └── src/
│       ├── App.tsx           # 主应用
│       ├── api.ts            # API 客户端
│       ├── board/            # 棋盘可视化
│       └── editor/           # 棋盘编辑器
│
├── examples/                  # 训练脚本
│   ├── train_ppo.py          # PPO 训练
│   └── eval_elo.py           # ELO 评估
│
├── models/                    # 训练产出
│   ├── model_registry.json   # ELO 数据
│   ├── ppo_penguinchess_*.zip # PPO 模型
│   └── alphazero/            # AZ 模型
│
├── docs/                      # 文档
│   ├── ARCHITECTURE.md       # 架构设计
│   ├── RULES.md              # 游戏规则
│   ├── OPTIMIZATION.md       # 优化分析
│   └── BOARD_EDITOR.md       # 编辑器设计
│
├── STATUS.md                  # 本文档
├── IMPROVEMENTS.md           # 改进建议
├── AGENTS.md                 # Agent 开发指南
└── README.md                 # 项目概述
```

---

## 四、性能基准

| 测试 | Python | Rust (release) | 加速比 |
|------|--------|---------------|--------|
| `core.step()` 单步 | 0.159 ms | **0.014 ms** | **11.5x** |
| 完整一局（随机） | 6.9 ms | **1.0 ms** | **6.9x** |
| MCTS (200 sims, Python 回调) | — | **5 ms** | — |
| MCTS (200 sims, 零拷贝 obs) | — | **4 ms** | **22x vs JSON** |
| 全量 ELO 55×1000 (1 worker) | ~963s | — | — |
| 全量 ELO (4 workers) | — | **~327s** | **~3x** |

---

## 五、已知问题

详见 [IMPROVEMENTS.md](IMPROVEMENTS.md)。

| ID | 级别 | 问题 |
|----|------|------|
| C1-C5 | 🔴 | 历史遗留 Bug（env 双重消除、hex_value 为 0、reward 常量等） |
| AZ1 | 🟡 | AlphaZero 迭代 ELO 未超越 PPO（MLP 架构限制） |
| AZ2 | 🟡 | 训练时 GPU 利用率仅 40-50%（MCTS CPU 端开销） |

---

## 六、下一步方向

1. **ResNet 架构训练** — 当前 `az_best` 仍是旧 MLP 模型，需用 `AlphaZeroResNet` 重新训练
2. **根并行 MCTS** — `mcts_search_parallel` 已实现未使用
3. **Pipeline 整合** — 训练 → ELO 评估 → 最佳模型部署
4. **Candle Rust 推理** — `net_infer.rs` 待接入 candle 实现纯 Rust 推理
