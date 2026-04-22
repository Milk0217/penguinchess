# 企鹅棋 — 改进建议文档

## 优先级总览

| 优先级 | 方向 | 核心改进项 |
|--------|------|-----------|
| **P0** | Gymnasium 环境 | Phase 1 全部内容 |
| **P1** | RL 训练基础 | Phase 3 全部内容 |
| **P2** | Web + RL 整合 | 共享核心规则、RL vs 人类对战 |
| **P3** | Self-play | Phase 4 全部内容 |
| **P4** | 评估工具 | Phase 5 全部内容 |

---

## P0: Gymnasium 环境核心（必须先行）

### P0.1 Python 游戏核心模块

将游戏规则逻辑统一在 Python 端实现，避免 Web JS 和 RL 环境各写一套。

**文件**: `penguinchess/core.py`（新建）

实现内容：
- `PenguinChessCore` 类，包含所有游戏规则（放置、移动、消除、胜负判定）
- `get_legal_actions()` — 返回当前所有合法动作
- `step(action)` — 执行动作，返回 (next_state, reward, done, info)
- `get_observation()` — 返回当前观测
- `reset(seed=None)` — 初始化棋盘

### P0.2 观测空间与动作空间

**文件**: `penguinchess/spaces.py`（新建）

- `ObservationSpace` — 棋盘状态编码（60 格子 × 3 特征 = 180维 Box）
- `ActionSpace` — Discrete(60)，60 个格子 ID

初期方案：简单扁平表示，稳定后升级为 Dict 结构。

### P0.3 Gymnasium 环境注册

**文件**: `penguinchess/env.py`（新建）

实现 `gymnasium.Env` 接口：
- `reset(seed=None)` → `obs, info`
- `step(action)` → `obs, reward, terminated, truncated, info`
- `close()`
- `seed(seed)`

注册到 Gymnasium：`gymnasium.register(id="PenguinChess-v0", entry_point=...)`

### P0.4 Reward Shaping

**文件**: `penguinchess/reward.py`（新建）

- 稀疏 reward（胜负 ±1）
- 密集 reward（分值变化、棋子消除、连通性奖励）

### P0.5 环境验证测试

**文件**: `tests/test_env.py`（新建）

- 随机种子一致性
- 动作空间合法性
- step/reset 循环稳定性
- 棋盘值总和 = 99

---

## P1: 基础 RL 训练

### P1.1 PPO 训练脚本

**文件**: `examples/train_ppo.py`（新建）

- 使用 Stable-Baselines3 PPO
- 100k 步训练 + 评估
- 保存 model checkpoint 到 `models/`

验收：对随机策略胜率 > 80%（100k 步）/ > 95%（500k 步）

### P1.2 依赖项配置

**文件**: `pyproject.toml`（更新）

添加依赖：
```
gymnasium>=0.29.0
stable-baselines3>=2.0.0
torch>=2.0.0
numpy>=1.24.0
```

---

## P2: Web 与 RL 环境整合

### P2.1 Flask API 暴露游戏核心

**文件**: `main.py`（改造）

将 `PenguinChessCore` 暴露为 REST API，使 Web 前端通过 HTTP 调用 Python 核心，不再需要 JS 重写游戏逻辑。

路由：
```
POST /api/reset          → 重置游戏
POST /api/step           → 执行动作
GET  /api/observation    → 获取当前观测
GET  /api/legal_actions  → 获取合法动作列表
POST /api/agent/act      → RL 智能体决策（可选，让人类 vs AI）
```

### P2.2 Web 前端改造

**文件**: `statics/main.js`（改造）

- 保留 UI 渲染和用户交互逻辑
- 游戏规则调用全部改为 `/api/` 请求
- 支持选择对手类型：人类 / RL 智能体

### P2.3 人类 vs AI 对战

在 Web 界面添加"vs AI"模式，用户可以选择与训练好的 RL 智能体对战。

---

## P3: Self-Play 训练框架

### P3.1 Self-Play 主循环

**文件**: `examples/selfplay.py`（新建）

- 维护策略种群（population）
- 每轮：当前策略 self-play → 数据收集 → 策略更新 → 评估
- ELO 评分跟踪

### P3.2 AlphaZero 风格训练

**文件**: `examples/train_alphazero.py`（新建）

- MCTS 树搜索
- 神经网络策略/价值评估
- 自对弈数据收集

---

## P4: 评估与可视化

### P4.1 训练可视化

集成 Weights & Biases 或 TensorBoard：
- 胜率曲线（vs 随机 / vs 贪心 / Self-play ELO）
- Loss 曲线（policy loss / value loss）
- 棋盘价值热力图

### P4.2 对局回放工具

复用现有 Web 回放系统，添加：
- RL 决策标注（显示智能体的 Q 值估计）
- 多局统计（平均步数、各方胜率）

---

## 已完成项

以下为 Web 版阶段已完成的改进项（保留参考）：

### ✅ P0 Web Bug 修复
- `gameovercheck()` 棋子归属判断修复
- `replayHistory()` 全局变量遮蔽修复
- `aftergame()` 空函数实现

### ✅ P1 Web 代码质量
- 魔法数字集中到 `config.js`
- ESLint / Prettier 配置
- Playwright E2E 测试

### ✅ P1 Web 响应式
- 棋盘 `min(600px, 90vw)` 自适应

---

## 旧版优先级（Web 对战为主）

以下为早期以 Web 为主时的优先级，现在已降级为辅助目标：

| 原优先级 | 内容 | 当前状态 |
|----------|------|----------|
| ~~AI 对战~~ | Web JS 简单 AI | 降级：优先做 RL 环境 |
| ~~局域网对战~~ | Flask-SocketIO | 降级：Phase 2 API 完成后用 REST 替代 |
| ~~多人扩展~~ | 3-4 人支持 | 搁置 |
| ~~构建工具化~~ | Vite / esbuild | 搁置：当前 ES module 够用 |
