# PenguinChess 架构文档

## 概述

企鹅棋（PenguinChess）是一款基于六边形格子的双人对战棋类游戏。

**核心设计理念**：
- **Python**：只负责 AI 模型训练
- **Rust**：游戏逻辑、搜索（Alpha-Beta + MCTS）、模型推理（NNUE + AlphaZero + PPO）
- **前端**：React + TypeScript，只负责渲染

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (React)                          │
│  http://localhost:5173 → Vite 开发服务器                      │
│  职责：棋盘渲染、用户交互、状态展示                             │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP JSON API
                      ▼
┌─────────────────────────────────────────────────────────────┐
│           后端 (Python Flask — 仅做路由/会话管理)              │
│                                                             │
│  server/app.py      HTTP 路由层（会话管理）                    │
│  server/game.py     Game 会话封装（调用 Rust 引擎）            │
│  server/ai_player.py AI 玩家 — 全部推理走 Rust FFI            │
│                                                             │
│  ┌─ Python 产线（训练专用）──┐                                │
│  │  examples/train_ppo.py  │  PPO 训练 → .zip               │
│  │  examples/train_*.py    │  AlphaZero/NNUE 训练 → .pth    │
│  │  penguinchess/env.py    │  Gymnasium 训练环境             │
│  └─────────────────────────┘                                │
└──────────────┬──────────────────────────────────────────────┘
               │ ctypes FFI
               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Rust 层 (game_engine.dll)                    │
│                                                             │
│  ┌─ 游戏引擎 ───────────────────────┐                        │
│  │  board.rs                        六边形棋盘数据结构       │
│  │  rules.rs                        游戏规则引擎             │
│  │  ffi.rs                          C FFI 导出入口          │
│  ├─ 搜索算法 ───────────────────────┤                        │
│  │  mcts_rs.rs                      MCTS（UCT + 批处理）    │
│  │  alphabeta_rs.rs                 Alpha-Beta（PVS+TT+LMR）│
│  │  nnue_rs.rs                      NNUE 推理（AVX2）       │
│  ├─ 模型推理 ───────────────────────┤                        │
│  │  az_model.rs                     AZ MLP/ResNet 推理      │
│  │                                  （BN 折叠 + AVX2 FMA）  │
│  │  nnue_rs.rs                      NNUE 推理               │
│  │  net_infer.rs                    ONNX 运行时（备选）      │
│  └──────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

```
训练 → Python PyTorch → .pth/.zip → 权重提取 + BN 折叠
                                       ↓
                                    Rust FFI
                                       ↓
                                    AZModelWeights / NNUEWeights
                                       ↓
                              ┌────────┴────────┐
                              │  推理             │
                              │  ffi_az_evaluate  │
                              │  mcts_search_core │
                              │  alphabeta_search │
                              └──────────────────┘
```

---

### API 端点

#### 游戏 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/game` | 创建新游戏，返回初始状态 |
| `GET` | `/api/game/<id>` | 获取指定游戏状态 |
| `POST` | `/api/game/<id>/action` | 提交动作（放置/移动） |
| `POST` | `/api/game/<id>/ai_move` | AI 自动执行一次移动 |
| `POST` | `/api/game/<id>/reset` | 重开一局（相同会话） |
| `GET` | `/api/health` | 健康检查 |

#### 模型 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/models` | 获取所有可用模型及评估数据（ELO/胜率） |
| `GET` | `/api/models/best` | 获取当前最优模型信息（基于 ELO） |

#### 棋盘 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/boards` | 获取所有已保存棋盘列表 |
| `POST` | `/api/boards` | 保存新棋盘（编辑器调用） |
| `GET` | `/api/boards/<id>` | 获取指定棋盘完整数据 |
| `DELETE` | `/api/boards/<id>` | 删除棋盘 |

所有 API 响应均为完整游戏状态，前端无状态。

### 目录结构

```
penguinchess/
├── server/                  # Flask HTTP 服务层（薄路由）
│   ├── app.py               # 路由、会话管理
│   ├── game.py              # Game 会话封装
│   ├── ai_player.py         # AI 玩家（Rust FFI 推理）
│   └── boards.py            # 棋盘存储管理
│
├── game_engine/              # Rust 运行态（全部逻辑 + 推理）
│   └── src/
│       ├── board.rs          # 棋盘
│       ├── rules.rs          # 规则引擎
│       ├── ffi.rs            # C FFI 导出
│       ├── mcts_rs.rs        # MCTS 搜索
│       ├── alphabeta_rs.rs   # Alpha-Beta 搜索
│       ├── nnue_rs.rs        # NNUE 推理（AVX2）
│       ├── az_model.rs       # AZ MLP/ResNet/PPO 推理
│       ├── net_infer.rs      # ONNX 推理（备选）
│       └── lib.rs            # 模块入口
│
├── penguinchess/            # Python 训练库（不参与线上推理）
│   ├── env.py               # Gymnasium 环境
│   ├── reward.py            # 奖励函数
│   ├── spaces.py            # Space 定义
│   ├── model_registry.py    # 模型注册表
│   ├── rust_ffi.py          # ctypes FFI 桥接
│   ├── rust_core.py         # RustCore 包装
│   └── ai/                  # AI 训练代码
│       ├── alphazero_net.py # AZ 网络定义（导权重用）
│       ├── nnue.py          # NNUE 定义（导权重用）
│       ├── nnue_train.py    # NNUE 训练
│       └── mcts_core.py     # Python MCTS（训练用）
│
├── examples/                # 训练脚本（唯一的 Python 推理场景）
│   ├── train_ppo.py         # PPO 训练
│   ├── train_alphazero.py   # AlphaZero 训练
│   ├── train_nnue_selfplay.py # NNUE 自对弈训练
│   └── eval_elo.py          # ELO 评估
│
├── models/                  # 模型文件（Python 训练 → Rust 加载）
│   ├── ppo_penguinchess_gen_*.zip
│   ├── alphazero/           # AlphaZero .pth
│   └── nnue/                # NNUE .pth
│
├── frontend/                # React 前端
├── docs/                    # 文档
├── statics/                 # 原始前端（Vanilla JS）
└── tests/                   # pytest 测试套件

---

## Rust 运行时（已实现）

`game_engine/` 是一个 Rust cdylib 库，编译为 `game_engine.dll`，通过 ctypes FFI 被 Python 调用。

```
┌──────────────────────────────────────────────────────────────────┐
│                     Python 层（仅训练 + 路由）                    │
│                                                                  │
│  server/app.py         HTTP 路由                                 │
│  server/ai_player.py   AI 玩家 — 全部推理走 Rust FFI             │
│  penguinchess/env.py   Gymnasium 环境（训练用）                   │
│  penguinchess/rust_ffi.py    ctypes FFI 桥接                     │
│  penguinchess/rust_core.py   RustCore 包装                      │
│                                                                  │
│  训练产线:                                                        │
│    examples/train_*.py → .pth / .zip → Rust FFI                  │
│                                                                  │
└───────────────────────┬──────────────────────────────────────────┘
                        │ ctypes FFI
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Rust 层 (game_engine.dll)                    │
│                                                                  │
│  游戏引擎                                                         │
│    board.rs        棋盘数据结构（Hex, Board, Piece）              │
│    rules.rs        规则引擎（GameState, step, get_legal_actions）│
│    ffi.rs          C FFI 导出                                    │
│                                                                  │
│  搜索算法                                                         │
│    mcts_rs.rs      MCTS（UCT + 批量回调 + Dirichlet 噪声）        │
│    alphabeta_rs.rs Alpha-Beta（PVS + TT + LMR + 期望窗口）       │
│                                                                  │
│  模型推理（AVX2 SIMD 加速）                                       │
│    nnue_rs.rs      NNUE（Alpha-Beta 评估函数）                    │
│    az_model.rs     AlphaZero MLP / ResNet / PPO 统一推理          │
│                    BN 折叠 + 残差连接 + 独立 policy/value 主干    │
│    net_infer.rs    ONNX Runtime（备选，feature=ort）              │
│                                                                  │
│  构建: cargo build --release → target/release/game_engine.dll    │
│  AVX2: RUSTFLAGS='-C target-cpu=native' cargo build --release    │
└──────────────────────────────────────────────────────────────────┘
```

### 推理性能（release + AVX2）

| 模型 | 架构 | 参数量 | 推理时间 |
|------|------|--------|---------|
| NNUE | FT(64) + FC(256→128) | 106K | **~3μs** |
| AlphaZero MLP | FC(512→256) + BN | 344K | **~10μs** |
| AlphaZero ResNet | FC(512→512→256) + BN + 残差 | 550K | **~15μs** |
| PPO MlpPolicy | FC(64→64) × 2 | ~30K | **~3μs** |

> 所有数值为 Rust CPU (AVX2 FMA) 单样本推理时间，不含 Python FFI 开销。

### 性能基准（release 构建）

使用 `--release` 优化编译后的基准测试结果（Intel CPU, 单线程）:

| 测试项 | Python | Rust (release) | 加速比 |
|--------|--------|---------------|--------|
| `core.step()` 单步 | 0.159 ms | **0.014 ms** | **11.5x** |
| 完整一局（随机走法） | 6.9 ms | **1.0 ms** | **6.9x** |
| PPO 推理单次 | 0.201 ms | — | — |
| ELO 评估 55对×50局 | ~19s* | **~3s*** | **~6.9x** |

> \* 仅含游戏模拟时间，模型推理时间另计。实际 ELO 评估受模型推理（~70%）和游戏模拟（~30%）共同影响。

| 构建配置 | 单步耗时 | 对比 Python |
|----------|---------|-------------|
| Python core.step() | 0.159 ms | 1.0x |
| Rust debug 构建 | 0.080 ms | **2.0x** |
| Rust release 构建 | 0.014 ms | **11.5x** |

### 有状态 FFI 设计（关键优化）

旧版 `game_step()` 每次调用需要完整序列化 GameState（~8KB JSON）：

```
Python 序列化 State → JSON → ctypes → Rust 反序列化 → step → 序列化 → JSON → 返回
```

新版有状态 API（`game_stateful_*`）将 GameState 保留在 Rust 内存中：

```
Python 传 (handle, action) → Rust 直接操作内存 → 返回小 JSON (~200B)
```

结果：单步耗时从 **2.018 ms**（旧版 FFI）降至 **0.014 ms**（有状态 release），**144x 改进**。

### 增量 ELO 模式

`--incremental` 跳过双方已有 ELO 的对局：

```bash
# 全量评估（首次）
uv run python examples/eval_elo.py --episodes 50 --rust-core

# 增量评估（训练新模型后）
uv run python examples/eval_elo.py --episodes 50 --incremental --rust-core
```

训练 gen_N 时只需跑 N-1 对新局（~10s），而非 N×(N-1)/2 对（~60s）。

### 组件与文件对应

| 职责 | 文件 | 说明 |
|------|------|------|
| 游戏核心（Rust 版） | `game_engine/src/rules.rs` + `board.rs` | 编译为 DLL |
| Rust FFI 桥接 | `penguinchess/rust_ffi.py` | ctypes 封装 |
| Rust MCTS 搜索 | `game_engine/src/mcts_rs.rs` | Rust + AZ 模型推理 |
| Rust Alpha-Beta 搜索 | `game_engine/src/alphabeta_rs.rs` | PVS + TT + LMR |
| NNUE 推理 | `game_engine/src/nnue_rs.rs` | AVX2 SIMD |
| AZ/PPO 模型推理 | `game_engine/src/az_model.rs` | AVX2 + BN 折叠 |
| AI 玩家（Rust 推理） | `server/ai_player.py` | Rust FFI 调用 |
| Model Registry | `penguinchess/model_registry.py` | JSON 持久化 |
| ELO 评估 | `examples/eval_elo.py` | 训练后评估 |
| PPO 训练 | `examples/train_ppo.py` | 训练产出 → Rust |
| AlphaZero 训练 | `penguinchess/ai/train_alphazero.py` | 训练产出 → Rust |
| AlphaZero 网络定义 | `penguinchess/ai/alphazero_net.py` | 导权重用 |
| Gymnasium 环境 | `penguinchess/env.py` | 训练用（不变） |

### AlphaZero 神经网络架构

（Python 训练定义，Rust 推理执行）

`penguinchess/ai/alphazero_net.py` 提供可配置的 ResNet 策略-价值网络：

| 类 | 参数量 | Rust 推理时间 |
|----|--------|---------------|
| `AlphaZeroNet` (MLP) | 344K | **~10μs** |
| `AlphaZeroResNet` | 550K | **~15μs** |
| `AlphaZeroResNetLarge` | 3.0M | **~30μs** |
| `AlphaZeroResNetXL` | 447M | ~20ms (CPU) |
| `AlphaZeroResNetConfigurable` | 自定义 | 取决于配置 |

**Rust 推理流程**：
1. Python 训练产出 `.pth` 文件（PyTorch state_dict）
2. `_build_az_model()` 加载权重 + BN 折叠 → 扁平 float32 数组
3. `ffi_az_create()` → `ffi_az_set_weights()` → `ffi_az_evaluate()`
4. Rust 端 AVX2 FMA 执行前向传播，输出 policy logits + value

### 构建 Rust

```bash
cd game_engine
cargo build --release                                    # 基础构建
RUSTFLAGS='-C target-cpu=native' cargo build --release   # AVX2 加速
```

---

## AI 开发进展

### 已完成里程碑

```
✅ 阶段 1: Python Gymnasium 环境          → 已完成
✅ 阶段 2: Rust 游戏核心 (FFI cdylib)     → 已完成
✅ 阶段 3: MCTS + AlphaZero 搜索          → Rust 原生
✅ 阶段 4: Alpha-Beta + NNUE 搜索         → Rust 原生 (AVX2)
✅ 阶段 5: 全模型 Rust 推理               → AZ/PPO/NNUE 全部 Rust
✅ 阶段 6: BN 折叠 + 残差网络             → az_model.rs 通用架构
⬜ 阶段 7: Rust ONNX 原生推理             → 预留，待 ort 稳定
```
├── reward.py                  # 奖励函数研究（不变）
├── spaces.py                  # Space 定义（不变）
├── model_registry.py          # Model Registry（ELO 持久化）
├── rust_ffi.py                # Rust ctypes FFI 桥接
├── rust_core.py               # RustCore 包装（duck-type 兼容 core.py）
├── rust_bridge.py             # Rust 统一桥接层
└── ai/                        # Python AI 算法模块
    ├── alphazero_net.py       # AlphaZero 神经网络
    ├── mcts_core.py           # MCTS 搜索核心
    └── train_alphazero.py     # AlphaZero 训练脚本
```

Rust 后端的 `mcts_rs.rs` 通过回调调用 Python 神经网络进行批量评估。`net_infer.rs` 预留用于未来 ONNX 原生推理，届时可绕过 Python 实现极致性能。

---

## 游戏规则

详见 [docs/RULES.md](RULES.md)。

---

## 快速开始

### 启动当前版本（Python 后端）

```bash
uv run python server/app.py
# 访问 http://localhost:8080
```

### 启动开发前端（Vite 热重载）

```bash
cd frontend && bun run dev
# 访问 http://localhost:5173
```

### 运行 AI 训练

```bash
uv run python -c "
from penguinchess.env import PenguinChessEnv
env = PenguinChessEnv()
obs, info = env.reset()
for _ in range(1000):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        obs, info = env.reset()
"
```

### 运行测试

```bash
uv run python -m pytest tests/ -q
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 游戏核心（Python 版） | Python 3.11 (`penguinchess/core.py`) |
| 游戏核心（Rust 加速） | Rust cdylib (`game_engine/`, `--release` 构建) |
| HTTP 框架 | Flask（当前）/ Axum（计划中） |
| 前端 | React 18 + TypeScript + Vite + Tailwind CSS v4 |
| AI 训练 | Python Gymnasium + PyTorch + Stable-Baselines3 |
| 测试 | pytest + Playwright |
