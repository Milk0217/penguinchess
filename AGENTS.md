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
├── server/                          # Flask HTTP 服务层
│   ├── app.py                      # HTTP 路由、会话管理
│   └── game.py                     # Game 会话封装
│
├── frontend/                        # React 前端
│   └── src/
│       ├── App.tsx                 # 主应用状态机
│       ├── board/                  # 棋盘可视化系统
│       │   ├── BoardContainer.tsx  # 主容器
│       │   ├── HexCell.tsx        # 六边形格子
│       │   ├── Piece.tsx          # 棋子组件
│       │   ├── Legend.tsx         # 图例
│       │   ├── types.ts           # 类型定义
│       │   ├── layouts/           # 布局策略
│       │   └── themes/            # 主题策略
│       └── editor/                # 棋盘编辑器
│
├── penguinchess/                    # Python 游戏核心 / RL 环境
│   ├── __init__.py
│   ├── core.py                     # PenguinChessCore 游戏逻辑
│   ├── env.py                      # Gymnasium 环境
│   ├── spaces.py                   # 观测/动作空间定义
│   ├── reward.py                   # Reward shaping 函数
│   └── random_ai.py               # 随机 AI 基准
│
├── statics/                         # 原始前端（Vanilla JS）
│
├── docs/                      # 文档

---

## 十一、Agent 工作流程与可用工具

### 11.1 调试与测试工具

Agent 在任务执行过程中可以使用以下工具：

| 工具 | 用途 |
|------|------|
| `uv run python ...` | 运行 Python 脚本（后端逻辑验证） |
| `uv run python server/app.py` | 启动 Flask 后端服务器 |
| `bun run dev` (前端目录) | 启动 Vite 前端开发服务器 |
| `uv run python start_all.py` | 一键启动前后端 |
| `gh` (GitHub CLI) | 查看 Issue、PR、代码审查 |
| `opencode debug` | 启动交互式调试会话 |
| `pytest` | 运行 Python 测试（如存在） |
| 后端日志 | 查看 Flask 输出的游戏状态、棋子死亡、阶段变化等信息 |

### 11.2 推荐工作流程

1. **分析阶段**：使用 `explore` agent 并行搜索代码库，理解相关模块
2. **调试阶段**：
   - 启动后端：`uv run python server/app.py`（终端 1）
   - 启动前端：`cd frontend && bun run dev`（终端 2）
   - 通过浏览器操作游戏，观察后端日志输出
   - 使用 `uv run python -c "..."` 直接测试后端逻辑
3. **验证阶段**：确认问题修复后，检查无回归问题
4. **提交**：使用 `git add/commit/push` 提交代码

### 11.3 后端调试日志解读

Flask 后端会输出详细的游戏状态日志：

```
[game_id] #001 | P1 placed | hex=57 (-6,+2,+4) | val=0 | +0.030 | P1:3/3 P2:3/3
[game_id] #002 | P2 placed | hex=56 (-5,+1,+4) | val=0 | +0.010 | P1:3/3 P2:3/3
[game_id] #003 | P1 placed | hex=55 (-7,+4,+3) | val=0 | +0.020 | P1:2/3 P2:3/3
[game_id] | P1 piece DEAD | piece=4 | (+0,+0,+0) | reason=eliminated
[game_id] MOVEMENT (seed=None)
[game_id] #007 | P1 moved | piece=6 | (+3,-5,+2) -> (+3,-5,+2) | hex=38 | +0.010 | P1:1/3 P2:2/3
```

- `P1:3/3 P2:3/3` 表示双方存活棋子数/总数
- `reason=eliminated` = 格子断连导致棋子被移除
- `reason=no valid moves` = 棋子无合法移动被移除

### 11.4 调试信息位置总览

#### 前端调试面板（浏览器）

在 URL 后加 `?debug=1` 可显示 Hex 数据面板（否则默认隐藏）：

| 面板 | 触发条件 | 内容 | 位置 |
|------|---------|------|------|
| **Hex 数据面板** | `?debug=1` | hex 总数/状态分布/前5个hex结构/第一个hex完整数据 | App.tsx `{debugMode && ...}` 区块 |
| **计分板** | 始终可见 | P1/P2 分数、回合中指示 | App.tsx 分数板区块 |
| **调试信息面板** | 始终可见 | 阶段、合法动作数、回合、活跃格子、存活棋子、分数、可移动棋子 | App.tsx 调试信息区块 |
| **棋子状态表** | 始终可见 | 所有 6 个棋子的 ID/归属/状态/坐标/hex索引/格子值 | App.tsx 棋子状态面板区块 |
| **状态提示** | 始终可见 | 当前操作提示（放置/移动/游戏结束） | App.tsx 操作提示区块 |
| **错误提示** | 出错时 | 红色错误信息条 | App.tsx 错误提示区块 |
| **Error Boundary** | 渲染崩溃时 | 错误信息 + 重新加载按钮 | App.tsx `<ErrorBoundary>` 包裹 |

#### 前端 Console 日志

| 日志 | 级别 | 触发 | 位置 |
|------|------|------|------|
| `[API] POST /api/game/xxx/action 200 5.2ms` | debug | 每次 POST 请求 | `api.ts request()` |
| `[API] GET /api/boards 200 2.0ms` | debug | 每次 GET 请求 | `api.ts request()` |
| `[API] ... 4xx 10.0ms` | warn | HTTP 错误响应 | `api.ts request()` |
| `[DEBUG] Game state from backend` | log | 每次游戏状态更新 | `App.tsx` useEffect |
| `[ErrorBoundary]` | error | React 渲染崩溃 | `App.tsx ErrorBoundary` |

#### 后端日志（stdout）

| 日志 | 级别 | 触发 | 位置 |
|------|------|------|------|
| `#001 | P1 placed | hex=...` | INFO | `server/game.py` → `step()` |
| `#007 | P1 moved | piece=...` | INFO | `server/game.py` → `step()` |
| `P1 piece DEAD | piece=4` | WARNING | `server/game.py` → `step()` |
| `P1 piece=6 @ (x,y,z) | moves=N` | INFO | `server/game.py` → 移动阶段每回合 |
| `MOVEMENT (seed=...)` | INFO | `server/game.py` → 阶段切换 |
| `Eliminated N hexes` | INFO | `server/game.py` → 消除后 |
| `GAME OVER | P1 WINS` | INFO | `server/game.py` → `step()` |
| `P1 INVALID action=N` | WARNING | `server/game.py` → 非法动作 |
| `POST /api/game 200 5.0ms` | INFO | `server/app.py` → request timing |
| `GET /api/game/xxx 404 1.0ms` | WARNING | `server/app.py` → 会话不存在 |
| `Unhandled exception: ...` | ERROR | `server/app.py` → `@app.errorhandler` |
| `Created builtin board: ...` | INFO | `server/boards.py` → 初始化 |
│   ├── RULES.md                   # 权威游戏规则
│   ├── ARCHITECTURE.md            # 架构设计与演进路线
│   └── BOARD_EDITOR.md            # 棋盘编辑器设计
│
├── examples/                        # 训练示例
│   └── random_ai.py               # 随机 AI 基准（已完成）
│   └── train_ppo.py              # ★ PPO 训练（待实现）
│   └── train_sac.py              # ★ SAC 训练（待实现）
│   └── selfplay.py               # ★ Self-play 训练（待实现）
│   └── evaluate.py               # ★ 评估工具（待实现）
│
└── game_engine/                    # ★ Rust 游戏核心（待实现）
    ├── board.rs
    ├── piece.rs
    ├── rules.rs
    └── state.rs
```

---

## 三、游戏规则（完整参考）

> **权威规则文档**: `docs/RULES.md` 是游戏的 source of truth。所有代码必须精确遵循该文档。
> 本章节仅作概览和开发参考，细节以 RULES.md 为准。

### 3.1 棋盘

- 60 个六边形格子，平行四边形排布
- 立方体坐标 (q, r, s)，约束 q + r + s = 0
- q ∈ [-4, 3]，r 随行变化
- 每个格子值 ∈ {1, 2, 3}，固定 10 个 3 分、20 个 2 分、30 个 1 分，总和 = 100
- value > 0 → 活跃；value ≤ 0 → 已消除

### 3.2 放置阶段

| 步骤 | 说明 |
|------|------|
| 顺序 | Player 1 先手，双方**严格交替**各放 3 个棋子（6 步） |
| 棋子 ID | Player 1: {4, 6, 8}，Player 2: {5, 7, 9} |
| 计分 | 放置时立即获得该格子分值 |
| 结束条件 | 双方都放满 3 个棋子后（步骤 6 完成后）→ 进入移动阶段 |
| 切换逻辑 | 每步检查当前玩家是否已放满：未满则切换到对方，已满则不切换 |

### 3.3 移动阶段

| 步骤 | 说明 |
|------|------|
| 顺序 | Player 1 先手，双方轮流移动 |
| 方向 | 棋子可沿 q轴、r轴、s轴 之一移动 |
| 约束 | 目标格子必须与当前位置在相同轴上坐标相同 |
| 阻挡 | 其他棋子（任意方）、已消除格子、中间格子不空 |
| 计分 | 移动到新格子后获得该格子分值 |
| 消除 | 每次移动后，不与任何棋子连通（6方向邻接）的格子被消除 |
| 销毁 | 无合法移动的棋子被移除 |

### 3.4 胜负条件

**胜负由最终分数决定**，而非棋子存活数量。分数更高者获胜，分数相同则平局。

游戏结束的触发条件（满足任一即结束）：

| 触发条件 | 处理 |
|----------|------|
| 一方棋子全部被移除 | 幸存者获得所有剩余格子分值 |
| 所有活跃格子均被消除 | 平局（双方都有棋子但无路可走） |
| 双方棋子同时被移除 | 平局 |

游戏结束时，比较双方最终得分（放置阶段得分 + 移动阶段得分 + 剩余格子分），分数高者获胜。

---

## 四、Gymnasium 环境规范

### 4.1 环境注册名

```python
gymnasium.make("PenguinChess-v0")
```

### 4.2 观测空间（Observation Space）

```python
# Dict 空间，包含以下键
{
    "board": Box(shape=(60, 3), low=0, high=99, dtype=np.float32),
        # shape=(60, 3): [q, r, value] 每行一个格子
        # value = 0 表示已消除格子
        # value = -1 表示无棋子，1/2/3 表示 P1/P2/无棋子但有值
        # 最终设计待实现时确定

    "pieces": Box(shape=(6, 4), low=-10, high=10, dtype=np.float32),
        # shape=(6, 4): [piece_id, q, r, s] 每行一个棋子
        # piece_id = -1 表示该棋子已被移除
        # 前3行 P1 棋子，后3行 P2 棋子

    "current_player": Box(shape=(1,), low=0, high=1, dtype=np.int8),
        # 0 = Player 1, 1 = Player 2

    "phase": Box(shape=(1,), low=0, high=1, dtype=np.int8),
        # 0 = 放置阶段, 1 = 移动阶段

    "scores": Box(shape=(2,), low=0, high=100, dtype=np.float32),
        # [player1_score, player2_score]
}
```

> **设计备注**: 实际观测空间设计需在 `env.py` 实现阶段根据游戏平衡性调优。可以先用简单的扁平化表示（board 展平成 60×3，pieces 展平成 24），稳定后再优化为 Dict 结构以提升 RL 效率。

### 4.3 动作空间（Action Space）

```python
# MultiDiscrete 或 Discrete，取决于设计选择

# 方案 A: Discrete（简单，60 个格子）
#   - 放置阶段: 60 个格子 ID（无效位置自动过滤）
#   - 移动阶段: 60 个格子 ID（无效移动自动过滤）

# 方案 B: MultiDiscrete（表达更丰富）
#   - axis: 选择移动轴（0=q, 1=r, 2=s）
#   - steps: 沿该轴移动的步数（+1 / -1）

# 推荐先实现方案 A，验证训练可行后再升级
Discrete(60)  # 或 MultiDiscrete([3, 2])
```

### 4.4 Step 返回值

```python
observation,    # 见上文观测空间
reward,         # float，本步回报
terminated,     # bool，游戏结束（胜负已定）
truncated,      # bool，是否被截断（如超过最大步数）
info            # dict，额外信息（见下文）
```

### 4.5 info 字段

```python
{
    "valid_actions": List[int],
        # 当前所有合法动作 ID，供智能体过滤用

    "current_player": int,
        # 当前玩家 0/1

    "phase": str,
        # "placement" | "movement"

    "scores": List[float],
        # [player1_score, player2_score]

    "pieces_remaining": List[int],
        # [player1_pieces, player2_pieces]

    "winner": int | None,
        # 游戏结束时: 0=Player1胜, 1=Player2胜, 2=平局
}
```

### 4.6 Reward 设计

| 阶段 | 事件 | Reward |
|------|------|--------|
| 任意 | 获得格子分值 | +分值/100（归一化） |
| 任意 | 对手棋子被移除 | +20 |
| 任意 | 自己的棋子被移除 | -20 |
| 任意 | 格子被消除（己方连通） | +0.5 |
| 移动阶段 | 移动到高价值格子 | +额外奖励 |
| 移动阶段 | 阻断对手连通 | +额外奖励 |
| 任意 | 胜利 | +100 |
| 任意 | 失败 | -100 |
| 任意 | 平局 | 0 |

> **设计备注**: Reward 设计对 RL 训练效果影响极大。初期用简单的稀疏 reward（胜负 ±1）也能收敛，但密集 reward 能显著加速训练。建议先实现稀疏版本，建立 baseline，再逐步加入 shaping reward。

### 4.7 最大步数

```python
max_episode_steps = 500  # 足够覆盖最长可能游戏
```

### 4.8 种子（Seed）

严格遵循 Gymnasium 规范：

```python
env.reset(seed=42)
env.action_space.seed(42)   # 动作空间也要设置种子
```

---

## 五、训练架构

### 5.1 推荐算法

| 算法 | 适用场景 | 推荐库 |
|------|----------|--------|
| **PPO** |首选，样本效率高，稳定性好 | Stable-Baselines3 / tianshou |
| **SAC** | 连续动作（若改用连续移动距离） | Stable-Baselines3 |
| **AlphaZero** | 完美信息博弈，自对弈 | 自主实现 |
| **DQN** | 简单基线，快速验证环境 | Stable-Baselines3 |

**推荐从 PPO 开始**，因为：
- 离散动作空间天然适配 PPO
- 社区成熟，文档丰富
- 超参数鲁棒

**GPU 加速**: 本机配备 NVIDIA GPU，PyTorch 自动使用 CUDA 加速训练（`device="auto"`）。可通过 `nvidia-smi` 确认 GPU 状态。

### 5.2 PPO 训练流程

#### 训练方式：自对弈

PPO 在 Gymnasium 环境中与自己下棋，每一回合环境自动切换玩家（P1↔P2），模型学会同时应对黑白两方的局面。

```
训练:  PPO ←step()→ Gymnasium(switch player) ←step()→ PPO
评估:  Gen_N → 随机 AI / Gen_{N-1} / Gen_{N-2} ...
```

#### 世代追踪（Generation）

每次训练产出新一代模型，自动与所有前代交叉对比：

```bash
# 训练 100k 步，4 个并行环境（推荐）
uv run python examples/train_ppo.py --timesteps 100000 --num-envs 4

# 续训已有模型
uv run python examples/train_ppo.py --resume models/ppo_penguinchess_gen_4.zip
```

训练结束后自动：
1. 保存到 `models/ppo_penguinchess_gen_N.zip`
2. vs 随机 AI — 测试基础强度
3. vs 所有前代 — 交叉对比，计算 ELO 评分

```
--- Gen 4 对战评估 ---
vs 随机 AI:  胜 56.7%  负 43.3%
vs gen_1:    胜 50.0%  负 23.3%  平 26.7%
vs gen_2:    胜 53.3%  负 20.0%  平 26.7%
vs gen_3:    胜 70.0%  负 23.3%  平 6.7%   ELO=1216
[OK] 超越前代平均值!
```

#### 速度优化

| 配置 | steps/s | 说明 |
|------|---------|------|
| GPU 单环境 | ~70 | MLP 策略不适用 GPU |
| CPU 单环境 | ~200 | CPU 对小模型更快 |
| CPU + 4 并行 | ~700+ | `--num-envs 4` 大幅加速 |

#### 超参数

```bash
uv run python examples/train_ppo.py \
    --timesteps 100000 \      # 训练步数
    --num-envs 4 \             # 并行环境数
    --lr 3e-4 \                # 学习率
    --batch-size 64 \          # 批次大小
    --n-steps 2048 \           # 每次更新步数
    --ent-coef 0.01 \          # 熵正则（鼓励探索）
    --force-gpu                # 强制 GPU（默认 CPU）
```

#### 评估已有模型

```bash
uv run python examples/train_ppo.py --evaluate-only
```

自动加载最新 `gen_N`，对比随机 AI 和所有前代。

### 5.3 前端 AI 对战

训练完成后，启动服务器即可在浏览器中选择 👤 vs 🤖 模式对战：

```bash
# 1. 训练（产出一代模型）
uv run python examples/train_ppo.py --timesteps 100000 --num-envs 4

# 2. 启动后端
uv run python server/app.py

# 3. 启动前端
cd frontend && bun run dev
# 浏览器 → 点击 👤 vs 🤖 → 新游戏 → 对战
```

前端 AI 自动加载最强一代模型（最新 `gen_N`）。

### 5.2 Self-Play 训练

```python
# examples/selfplay.py 核心循环

while training:
    # 当前策略与历史策略对战
    opponent = select_opponent(population, method="ELO_weighted")

    # 收集 self-play 对局数据
    for _ in range(num_selfplay_games):
        env = PenguinChessEnv()
        trajectory = collect_trajectory(env, current_policy, opponent)

    # 更新策略
    current_policy = ppo_update(trajectory, current_policy)

    # 评估 & 加入种群
    if should_eval(current_policy, population):
        elo_rating = evaluate(current_policy, population)
        population.add(current_policy, elo_rating)
```

### 5.3 评估指标

| 指标 | 说明 |
|------|------|
| 对随机策略胜率 | 基线 > 95% 才算环境可用 |
| 对简单贪心胜率 | 贪心 = 每步选最高分值移动 |
| ELO 评分 | Self-play 过程中跟踪 |
| 对抗特定对手胜率 | 测试泛化能力 |

---

## 六、架构演进与开发路线

> ⚠️ **重要**: Python 游戏核心（`core.py`）为**临时原型**，后续将迁移至 Rust 以获得极致性能。Python AI 训练代码（`env.py`、`reward.py` 等）**永久保留**，通过进程通信与 Rust 后端交互。
>
> 详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

### Phase 1: Gymnasium 环境核心（✅ 已完成）

**目标**: 实现 `penguinchess/env.py`，通过 Gymnasium API 完整覆盖游戏逻辑。

**交付物**:
- `penguinchess/core.py` — 游戏核心（棋盘/棋子/规则）
- `penguinchess/spaces.py` — 观测/动作空间定义
- `penguinchess/env.py` — Gymnasium 环境类
- `penguinchess/reward.py` — reward shaping 函数

**验收标准**: 100 局随机对战验证，P1/P2 胜率接近 50%（无先手优势）。

---

### Phase 2: 前后端分离 Web 架构（✅ 已完成）

**目标**: React + Flask 前后端分离，Python 游戏逻辑通过 HTTP API 暴露。

**交付物**:
- `server/app.py` — Flask HTTP 服务
- `server/game.py` — Game 会话封装
- `frontend/` — React + Vite + TypeScript 前端

---

### Phase 3: 游戏核心迁移到 Rust（🚧 待开始）

**目标**: 用 Rust 重写游戏核心逻辑，极致性能。

**动机**: 随着训练规模扩大，Python 游戏核心的性能瓶颈会越来越明显。Rust 版本可将单步推理延迟从 ~10-100μs 降至 ~0.1-1μs（100 倍提升），大幅缩短训练时间。

**交付物**:
- `game_engine/` — Rust 游戏核心 crate
  - `board.rs` — 六边形棋盘数据结构
  - `piece.rs` — 棋子状态与移动逻辑
  - `rules.rs` — 游戏规则引擎
  - `state.rs` — 游戏状态快照

**API 兼容性**: Rust 后端保持与 Python 后端完全相同的 HTTP API 接口，前端无感知。

---

### Phase 4: Python AI 进程通信接口（🚧 待开始）

**目标**: Rust 后端通过子进程调用 Python AI，保持 AI 研究灵活性。

**交付物**:
- `ai/python_ai.rs` — Rust 进程通信模块
- `penguinchess/ai/` — Python AI 研究目录（永久保留）
  - `policy_gradient.py` — PPO / A2C 策略梯度
  - `mcts.py` — 蒙特卡洛树搜索
  - `self_play.py` — 自我对弈训练循环

---

### Phase 5: RL 训练 pipeline（🚧 待开始）

**目标**: 完整的强化学习训练系统。

**交付物**:
- `examples/train_ppo.py` — PPO 训练脚本
- `examples/train_alphazero.py` — AlphaZero 风格训练

**验收标准**: 训练 100k 步后，对随机策略胜率 > 80%。

---

### Phase 6: Self-Play 训练框架（🚧 待开始）

**目标**: 自我对弈训练循环，持续提升 AI 水平。

**交付物**:
- ELO 评分系统
- 最佳模型保存与早停
- 训练曲线可视化

---

### Phase 7: Rust AI 推理集成（🚧 待开始）

**目标**: 将训练好的策略网络导出为 Rust 原生推理。

**交付物**:
- `ai/onnx_export.py` — 模型导出脚本
- `ai/rust_inference.rs` — Rust ONNX Runtime 推理

---

## 七、Web 对战、RL 环境与 Rust 后端的代码共享

游戏核心规则应在**一处统一实现**，Web 前端和 RL 环境均调用它。当前为 Python 原型，未来迁移到 Rust。

**当前架构（Python 原型）**:

```
penguinchess/core.py  ← 游戏核心（所有规则）
       ↓
server/game.py       ← Flask API 封装
       ↓
frontend/            ← React Web UI（纯渲染）
       ↓
penguinchess/env.py  ← Gymnasium RL 环境
```

**未来架构（Rust 后端）**:

```
game_engine/ (Rust)  ← 游戏核心（所有规则）
       ↓
api/ (Rust)          ← HTTP API（与 Python 版完全兼容）
       ↓
frontend/            ← React Web UI（无感知变化）
       ↓
penguinchess/env.py  ← Gymnasium RL 环境（永久保留）
       ↓
ai/python_ai.py      ← Python AI 训练（永久保留）
       ↓
game_engine/ai/python_ai.rs  ← Rust 进程通信调用 Python AI
```

**关键原则**:
- Python `env.py` **永不改变** — Gymnasium 接口是 RL 训练的稳定契约
- Rust 后端通过子进程调用 Python AI 模块，实现高性能推理
- Web 前端仅依赖 HTTP API，与实现语言无关

---

## 八、文件依赖关系

```
server/app.py          # Flask HTTP 服务，导入 server/game.py
server/game.py         # Game 会话，导入 penguinchess/core.py
penguinchess/core.py   # 游戏核心（规则）
penguinchess/env.py    # Gymnasium 环境，导入 core.py
penguinchess/reward.py # 奖励函数，被 env.py 调用
penguinchess/spaces.py # Space 定义，被 env.py 调用
frontend/src/          # React 前端，通过 HTTP 调用 server/app.py
```

---

## 八-2、棋盘可视化系统

### 架构设计

棋盘系统采用**策略模式**，将布局（Layout）与渲染逻辑分离，支持多种棋盘形状和主题。

```
frontend/src/board/
├── types.ts              # 核心类型定义
├── BoardContainer.tsx    # 主容器组件
├── HexCell.tsx           # 六边形单元组件
├── Piece.tsx             # 棋子组件
├── Legend.tsx            # 图例组件
├── layouts/
│   ├── index.ts          # 布局注册表
│   ├── parallelogram.ts  # 平行四边形布局（60格）
│   └── hexagon.ts        # 正六角形布局（61格，半径4）
└── themes/
    ├── index.ts          # 主题注册表
    ├── default.ts        # 默认主题（亮色）
    └── dark.ts           # 暗色主题
```

### 核心接口

```typescript
// 棋盘布局接口
interface BoardLayout {
  id: string;
  name: string;
  generateHexes(): HexCoord[];           // 生成六边形坐标
  cubeToPixel(q, r, config): PixelCoord; // 坐标转像素
  getBounds(hexes, config): Bounds;       // 获取边界
}

// 主题配置
interface BoardTheme {
  id: string;
  colors: ThemeColors;   // 11个颜色配置
  sizes: ThemeSizes;     // 尺寸配置
  effects: ThemeEffects; // 效果开关
}
```

### 添加新布局

1. 创建 `frontend/src/board/layouts/xxx.ts`
2. 实现 `BoardLayout` 接口
3. 在 `index.ts` 中注册: `register(xxxLayout)`

### 添加新主题

1. 创建 `frontend/src/board/themes/xxx.ts`
2. 实现 `BoardTheme` 接口
3. 在 `index.ts` 中注册: `register(xxxTheme)`

### 使用方式

```tsx
<BoardContainer
  state={gameState}
  layout={getLayout("parallelogram")}  // 或 "hexagon"
  theme={getTheme("default")}          // 或 "dark"
  selectedPieceId={selectedPieceId}
  targetIndices={targetIndices}
  onHexClick={handleHexClick}
/>
```

---

## 九、已知问题（已修复）

以下问题已在实现过程中修复：

| 问题 | 修复方案 |
|------|----------|
| `gameovercheck()` 棋子归属判断 | 改为 `piece.id % 2`，P1=[4,6,8]，P2=[5,7,9] |
| 坐标系统不一致（q/r/s） | 添加 `_q_raw` 字段存储原始 q 值 |
| flood_fill 起点包含被占据格子 | 改为 `value < 0` |
| 游戏结束检查时机错误 | `_check_game_over()` 移至消除之后 |
| 放置阶段规则 | 改为严格交替，P1 先手 |

---

## 十、环境设置与运行指南

### 10.1 系统要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.11+ | 游戏核心与后端 |
| Node.js | 18+ | 前端开发与构建 |
| bun | ≥ 1.0 | 前端包管理器与运行器（也可用 npm） |
| uv | ≥ 0.1.0 | Python 包管理器（推荐）|

### 10.2 安装依赖

```bash
# 1. 克隆仓库
git clone https://github.com/Milk0217/penguinchess.git
cd penguinchess

# 2. 安装 Python 依赖（使用 uv）
uv sync
# 或使用 pip
pip install -e ".[dev]"

# 3. 安装前端依赖（bun 或 npm）
cd frontend
bun install
# 或 npm install
cd ..
```

### 10.3 启动方式

#### 方式 A：纯后端（人类 vs 人类，Flask 托管静态文件）

```bash
# Flask 会同时托管编译后的前端静态文件
uv run python server/app.py
# 访问 http://localhost:8080
```

#### 方式 B：前后端分离开发（推荐）

终端 1 - 后端 API 服务器：
```bash
uv run python server/app.py
```

终端 2 - 前端 Vite 开发服务器（热重载）：
```bash
cd frontend
bun run dev  # 或 npm run dev
# 访问 http://localhost:5173
```

#### 方式 C：一键启动（后端 + 前端开发服务器）

```bash
uv run python start_all.py
# 同时启动 Flask:8080 和 Vite:5173
```

### 10.4 棋盘编辑器

1. 启动后端和前端（方式 B 或 C）
2. 访问前端页面（http://localhost:5173 或 http://localhost:8080）
3. 点击 **"Board Editor"** 按钮进入编辑器
4. 在画布上点击格子选择/取消选择，构建自定义棋盘形状
5. 至少选择 60 个格子后，点击 **"Save Board"** 保存
6. 保存后可在游戏中选择该棋盘

**注意**：编辑器生成的棋盘坐标存储在 `backend_data/boards/` 目录中，以 JSON 格式保存。

### 10.5 运行测试

```bash
# 运行所有 Python 测试
pytest tests/ -q

# 运行指定测试文件
pytest tests/test_core.py -v
pytest tests/test_env.py -v
pytest tests/test_spaces.py -v

# 运行前端测试（vitest）
cd frontend
bun run test  # 或 npx vitest run

# 运行 E2E 测试（需先启动后端）
cd frontend
bunx playwright test  # 或 npx playwright test
```

### 10.6 Gymnasium 环境验证

```bash
# 环境自检
uv run python -c "
import gymnasium as gym
from penguinchess.env import PenguinChessEnv

env = gym.make('PenguinChess-v0')
obs, info = env.reset()
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)
print('OK: env test passed')
env.close()
"

# 随机 AI 基准测试
uv run python examples/random_ai.py

# Python 环境直接使用
uv run python -c "
from penguinchess import PenguinChessCore

core = PenguinChessCore()
core.reset(seed=42)
print('Initial phase:', core.phase)
print('Player:', core.current_player)
for step in range(6):  # 放置阶段 6 步
    legal = core.get_legal_actions()
    core.step(legal[0])
print('Phase after placement:', core.phase)
print('Scores:', core.players_scores)
"
```

### 10.7 Web 对战流程

1. 启动服务器（方式 A 或 B）
2. 浏览器打开
3. 点击 **"New Game"** 创建游戏
4. 两名玩家轮流点击棋盘格子进行放置（各 3 个棋子）
5. 放置阶段结束后自动进入移动阶段
6. 点击棋子选中，再点击高亮目标格子移动
7. 游戏结束时显示胜负和分数

### 10.8 项目结构速览

```
penguinchess/
├── penguinchess/              # Python 游戏核心
│   ├── core.py               # 游戏规则引擎（~900 行）
│   ├── env.py                # Gymnasium 环境
│   ├── spaces.py             # 观测/动作空间
│   └── reward.py             # 奖励函数
├── server/                    # Flask HTTP 后端
│   ├── app.py                # 路由与 API
│   ├── game.py               # 游戏会话
│   └── boards.py             # 棋盘存储
├── frontend/                  # React + Vite 前端
│   ├── src/App.tsx           # 主应用
│   ├── src/api.ts            # API 客户端
│   ├── src/board/            # 棋盘可视化
│   └── src/editor/           # 棋盘编辑器
├── tests/                     # 测试套件
├── backend_data/boards/      # 已保存的棋盘
└── docs/                      # 文档
