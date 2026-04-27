# 企鹅棋 — 改进建议文档

> 本文档记录当前发现的 **Bug**、**代码质量问题**和**架构改进建议**。包含 P0-P4 优先级分类。

---

## 🔴 P0 — 严重 Bug（必须立即修复）

### C1. `env.py` 双重消除调用

**位置:** `penguinchess/env.py:142-144`

`env.step()` 在调用 `_game.step()` 后重复调用了 `_eliminate_disconnected_hexes()`。
`core.step()` 内部已执行消除逻辑，导致每次移动步消除执行两次，可能造成状态损坏。

**修复:** 删除 `env.py` 中 142-144 行。

---

### C2. `server/game.py` 日志中 `hex_value` 始终为 0

**位置:** `server/game.py:138,155,171`

Hex 重构后使用 `points` 而非 `value`，但日志代码使用了 `getattr(hex_obj, "value", 0)`，
导致记录的分值始终为 0。

**修复:** 替换为 `hex_obj.points`。

---

### C3. `reward.py` 总分常量 `TOTAL_VALUE = 100` 错误

**位置:** `penguinchess/reward.py:14`

`core.py:TOTAL_VALUE = 99`，但 `reward.py` 为 100，导致奖励计算偏小。

**修复:** 改为 `TOTAL_VALUE = 99`。

---

### C4. 前端硬编码自定义棋盘 ID

**位置:** `frontend/src/App.tsx:151,166`

**影响:** 其他用户无此棋盘，首次加载游戏会失败。

**修复:** 改为 `"default"`。

---

## 🟡 P1 — 逻辑 Bug

### L1. 前端 `last_action.hex` 类型与后端不匹配

- 前端类型: `state` + `points`
- 后端返回: `q, r, s, value`（`value` 始终为 0）

### L2. E2E 测试端口 5000，实际服务器在 8080

### L3. `start_all.py` 使用 `bun`，且 README 指定 `npm`

### L4. `api.ts` 缺少 `getBoard(boardId)` 方法

后端有 `GET /api/boards/<id>` 端点，但前端 API 客户端未实现调用。

### L5. 废弃调试脚本未更新（10+ 文件）

Hex 重构后（`value` → `state`/`points`），以下脚本会报 `AttributeError`:

| 文件 | 问题 |
|------|------|
| `test_corner_death.py` | 多处 `h.value` → `h.points`/`is_active()` |
| `test_piece_death_debug.py` | 多处 `h.value` |
| `test_piece_death2.py` | 多处 `h.value` |
| `test_cascade.py` | `p.hex.qrs` 不存在 |
| `examples/random_ai.py:148` | `h.value > 0` → `h.is_active()` |

---

## 🟢 P2 — 代码质量

### Q1. 根目录脚本过多（10+ 个调试文件）

建议: 删除或移至 `tests/`。

### Q2. 重复的 `GameState` 类型

`api.ts` 和 `types.ts` 各定义了一份，不同步。

### Q3. `_check_game_over()` 死代码 (core.py:851)

`has_active` 赋值但从未使用。

### Q4. `env.py` 死代码

`self._prev_scores` / `self._prev_pieces` 定义赋值但从未读取。

### Q5. `tests/test_core.py:312` 无意义断言

```python
assert eliminated_count >= 0  # 永远为 True
```

---

## 🔵 P3 — 架构改进

### A1. 统一坐标系统

两个坐标系统: 原始坐标 (`default.json`) vs 调整后坐标 (`create_board()`)。`json_board_to_coords()` 负责转换，但增加了复杂度。

### A2. `__init__.py` 导出不完整

当前只导出 `PenguinChessEnv`，建议补充 `PenguinChessCore` 和工具函数。

### A3. 自定义 Space 类型兼容性

`PenguinChessFlatObs` 继承 `gym.Space`，但部分 Gymnasium wrapper 可能不支持。

---

## 🟠 AZ — AlphaZero 训练问题

### AZ1. MLP 架构表达能力不足

**位置**: `penguinchess/ai/alphazero_net.py`

**问题**: MLP (206→512→256) 无法有效学习棋盘空间结构。ResNet (AlphaZeroResNet) 已实现但旧模型 (az_iter_10) 使用旧 MLP。

**修复**: 使用 `AlphaZeroResNet` 重新训练（需 `--resume` 续训时自动切换）。

### AZ2. GPU 利用率仅 40-50%

**位置**: `penguinchess/ai/train_alphazero.py` 自对弈循环

**问题**: MCTS 模拟在 Rust 端运行，GPU 仅在 callback 时使用。400 sims × 100 games 中 GPU 前向传播仅占 ~2s/迭代（总时间 23s）。

**方向**: 增加 batch_size (当前 256)、根并行 MCTS。

### AZ3. best_net 评估使用 MCTS + 网络（非纯网络强度）

**位置**: `penguinchess/ai/train_alphazero.py` → `_evaluate_models()`

**问题**: 评估时双方都用 MCTS(800 sims) + 网络，MCTS 会平滑网络差异。纯网络强度指标缺失。

**建议**: 增加纯网络（无 MCTS）ELO 评估。

---

## ⚪ P4 — 功能待实现

| 方向 | 内容 | 状态 |
|------|------|------|
| AI 训练 | 根并行 MCTS (已实现未启用) | 🔶 待启用 |
| AI 训练 | Candle Rust 端推理（net_infer.rs 预留） | ⬜ 待实现 |
| AI 训练 | 分布式自对弈（多 GPU 生成数据） | ⬜ 待开始 |
| Web 对战 | 人类 vs AI 对战 | 🤖 后端已就绪 |
| Rust 核心 | HTTP API 迁移到 Rust (Axum) | ⬜ 待开始 |
| 评估工具 | 训练可视化 / 对局回放 | ⬜ 待开始 |

---

## 已完成项

### Phase 1: Gymnasium 环境核心 ✅

- `penguinchess/core.py` — 游戏核心逻辑
- `penguinchess/env.py` — Gymnasium 环境
- `penguinchess/spaces.py` — Space 定义
- `penguinchess/reward.py` — Reward 函数
- `penguinchess/random_ai.py` — 随机 AI 基准
- 91 个测试全部通过

### Phase 2: Web 前后端分离 ✅

- `server/app.py` — Flask HTTP 服务
- `server/game.py` — Game 会话封装
- `frontend/` — React + Vite 前端
- 棋盘编辑器 + 布局/主题策略模式
- `default.json` 棋盘作为默认地图

### Bug 修复历史

- 棋子归属判断 (`piece.id % 2`)
- 坐标系统不一致（新增 `_q_raw` 字段）
- flood_fill 起点包含被占据格子
- 放置阶段严格交替规则
- `_neighbors` 邻居计算（调整后坐标转换）
- 棋子阵亡判断简化（邻居 active 检查）

### ✅ P0.1 Python 游戏核心模块

**文件**: `penguinchess/core.py`（已完成）

实现内容：
- `PenguinChessCore` 类，包含所有游戏规则（放置、移动、消除、胜负判定）
- `get_legal_actions()` — 返回当前所有合法动作
- `step(action)` — 执行动作，返回 (next_state, reward, done, info)
- `get_observation()` — 返回当前观测
- `reset(seed=None)` — 初始化棋盘

### ✅ P0.2 观测空间与动作空间

**文件**: `penguinchess/spaces.py`（已完成）

- `ObservationSpace` — 棋盘状态编码（60 格子 × 3 特征 = 180维 Box）
- `ActionSpace` — Discrete(60)，60 个格子 ID

初期方案：简单扁平表示，稳定后升级为 Dict 结构。

### ✅ P0.3 Gymnasium 环境注册

**文件**: `penguinchess/env.py`（已完成）

实现 `gymnasium.Env` 接口：
- `reset(seed=None)` → `obs, info`
- `step(action)` → `obs, reward, terminated, truncated, info`
- `close()`
- `seed(seed)`

注册到 Gymnasium：`gymnasium.register(id="PenguinChess-v0", entry_point=...)`

### ✅ P0.4 Reward Shaping

**文件**: `penguinchess/reward.py`（已完成）

- 稀疏 reward（胜负 ±1）
- 密集 reward（分值变化、棋子消除、连通性奖励）

### ⬜ P0.5 环境验证测试

**文件**: `tests/test_env.py`（待实现）

待验证：
- 随机种子一致性
- 动作空间合法性
- step/reset 循环稳定性
- 棋盘值总和 = 99

---

## P1: 基础 RL 训练

### 🚧 P1.1 PPO 训练脚本

**文件**: `examples/train_ppo.py`（待实现）

目标：
- 使用 Stable-Baselines3 PPO
- 100k 步训练 + 评估
- 保存 model checkpoint 到 `models/`

验收：对随机策略胜率 > 80%（100k 步）/ > 95%（500k 步）

### ✅ P1.2 依赖项配置

已安装依赖：
```
gymnasium>=0.29.0
stable-baselines3>=2.0.0
torch>=2.0.0
numpy>=1.24.0
```

---

## P2: Web 与 RL 环境整合

### ✅ P2.1 Flask API 暴露游戏核心

**文件**: `server/app.py`, `server/game.py`（已完成）

已将 `PenguinChessCore` 暴露为 REST API，Web 前端通过 HTTP 调用 Python 核心。

路由：
```
POST /api/game           → 创建游戏
GET  /api/game/<id>     → 获取游戏状态
POST /api/game/<id>/action → 执行动作
POST /api/game/<id>/reset → 重置游戏
```

### ✅ P2.2 Web 前端改造

**文件**: `frontend/`（已完成）

- React + Vite 前端
- 游戏规则调用全部通过 `/api/` 请求
- 策略模式布局和主题系统

### 🚧 P2.3 人类 vs AI 对战

**状态**: 部分完成

Web UI 已支持选择对手类型，但 RL 智能体对战功能待训练脚本完成后可用。

---

## P3: Self-play 训练框架 ✅ 已完成

### P3.1 Self-Play + ELO 系统

**文件**:
- `penguinchess/ai/train_alphazero.py` — AlphaZero 自对弈训练
- `penguinchess/ai/alphazero_net.py` — AlphaZero 网络（MLP + ResNet）
- `penguinchess/ai/mcts_core.py` — Python MCTS 批处理 + 根并行
- `examples/eval_elo.py` — ELO 评估系统（增量、并行、多模型）

### P3.2 Rust MCTS + 优化

- Rust MCTS 搜索（`mcts_rs.rs`）— JSON 零拷贝，22x 加速
- Dirichlet 噪声 + 随机 UCB 平局打断
- GPU AMP 混合精度 + 非阻塞 CUDA 传输
- flat obs 存储消除 CPU 解码瓶颈
- ResourceMonitor GPU/CPU/RAM 监控

---

## P4: 评估与可视化 ⬜ 待开始

### P4.1 训练可视化

待实现：
- Weights & Biases 或 TensorBoard 集成
- 胜率曲线（vs 随机 / vs 贪心 / Self-play ELO）
- Loss 曲线（policy loss / value loss）
- 棋盘价值热力图

### P4.2 对局回放工具

待实现：
- 复用现有 Web 回放系统
- RL 决策标注（显示智能体的 Q 值估计）
- 多局统计（平均步数、各方胜率）

---

## 历史已完成项

### ✅ Phase 1: Gymnasium 环境核心 (2026-04)

- `penguinchess/core.py` — PenguinChessCore 游戏逻辑
- `penguinchess/env.py` — Gymnasium 环境
- `penguinchess/spaces.py` — Space 定义
- `penguinchess/reward.py` — Reward 函数
- `penguinchess/random_ai.py` — 随机 AI 基准

### ✅ Phase 2: Web 前后端分离 (2026-04)

- `server/app.py` — Flask HTTP 服务
- `server/game.py` — Game 会话封装
- `frontend/` — React + Vite 前端
- `docs/BOARD_EDITOR.md` — 棋盘编辑器设计

### ✅ Bug 修复

- `gameovercheck()` 棋子归属判断修复
- `replayHistory()` 全局变量遮蔽修复
- `aftergame()` 空函数实现
- 坐标系统不一致修复
- flood_fill 起点包含被占据格子修复
- 游戏结束检查时机错误修复
- 放置阶段规则修复（严格交替）

### ✅ 代码质量

- 魔法数字集中到配置
- ESLint / Prettier 配置
- Playwright E2E 测试
- 棋盘响应式布局

---

## 技术栈

| 层级 | 当前技术 | 未来技术 |
|------|----------|----------|
| 游戏核心 | Python 3.11 | Rust |
| HTTP 框架 | Flask | Axum / Actix-web |
| 前端 | React 18 + TypeScript + Vite | React 18 + TypeScript + Vite |
| RL 框架 | Gymnasium + Stable-Baselines3 | Gymnasium + Stable-Baselines3 |
