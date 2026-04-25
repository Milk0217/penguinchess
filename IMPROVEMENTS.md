# 企鹅棋 — 改进建议文档

> ⚠️ **注意**: 本文档部分内容已过时。请参考 [STATUS.md](STATUS.md) 获取最新项目状态。

## 优先级总览

| 优先级 | 方向 | 核心改进项 | 状态 |
|--------|------|-----------|------|
| **P0** | Gymnasium 环境 | Phase 1 全部内容 | ✅ 已完成 |
| **P1** | RL 训练基础 | Phase 3 全部内容 | 🚧 部分完成 |
| **P2** | Web + RL 整合 | 共享核心规则、RL vs 人类对战 | 🚧 部分完成 |
| **P3** | Self-play | Phase 4 全部内容 | ⬜ 待开始 |
| **P4** | 评估工具 | Phase 5 全部内容 | ⬜ 待开始 |

---

## P0: Gymnasium 环境核心 ✅ 已完成

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

## P3: Self-play 训练框架 ⬜ 待开始

### P3.1 Self-Play 主循环

**文件**: `examples/selfplay.py`（待实现）

- 维护策略种群（population）
- 每轮：当前策略 self-play → 数据收集 → 策略更新 → 评估
- ELO 评分跟踪

### P3.2 AlphaZero 风格训练

**文件**: `examples/train_alphazero.py`（待实现）

- MCTS 树搜索
- 神经网络策略/价值评估
- 自对弈数据收集

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
