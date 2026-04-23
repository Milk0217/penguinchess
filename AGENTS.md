# PenguinChess — Agent 开发指南

## 一、项目目标

本项目有两个核心方向：

1. **Gymnasium 环境** — 构建严格遵循 Farama Gymnasium 接口的企鹅棋 RL 环境
2. **强 AI 训练** — 基于该环境训练高胜率 RL 智能体

---

## 二、架构概览

```
penguinchess/
├── main.py                          # Flask Web 服务器（人类对战）
├── templates/index.html             # Web UI
├── statics/                         # Web 前端 JS/CSS
│   ├── main.js                      # 游戏逻辑（Web 版本）
│   ├── board.js                     # Hex 棋盘类
│   ├── piece.js                     # 棋子类
│   ├── player.js                   # 玩家类
│   └── config.js                   # 配置常量
│
├── penguinchess/                    # RL 环境模块
│   ├── __init__.py
│   ├── env.py                      # ★ Gymnasium 环境核心（待实现）
│   ├── spaces.py                   # ★ 观测空间 / 动作空间定义（待实现）
│   ├── reward.py                   # ★ Reward shaping 函数（待实现）
│   └── wrapped.py                  # ★ Gymnasium 包装器（待实现）
│
└── examples/                        # 训练示例
    ├── train_ppo.py                # ★ PPO 训练（待实现）
    ├── train_sac.py                # ★ SAC 训练（待实现）
    ├── selfplay.py                 # ★ Self-play 训练（待实现）
    └── evaluate.py                 # ★ 评估工具（待实现）
```

---

## 三、游戏规则（完整参考）

> **权威规则文档**: `docs/RULES.md` 是游戏的 source of truth。所有代码必须精确遵循该文档。
> 本章节仅作概览和开发参考，细节以 RULES.md 为准。

### 3.1 棋盘

- 60 个六边形格子，平行四边形排布
- 立方体坐标 (q, r, s)，约束 q + r + s = 0
- q ∈ [-4, 3]，r 随行变化
- 每个格子值 ∈ {1, 2, 3}，总和 = 99
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

    "scores": Box(shape=(2,), low=0, high=99, dtype=np.float32),
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
| 任意 | 获得格子分值 | +分值/99（归一化） |
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

## 十、运行测试

```bash
# Web 对战（人类 vs 人类）
cd /mnt/e/programming/penguinchess
source .venv/bin/activate
python server/app.py
# 访问 http://localhost:8080

# Gymnasium 环境自检
source .venv/bin/activate
python -c "
import gymnasium as gym
from penguinchess.env import PenguinChessEnv
env = gym.make('PenguinChess-v0')
obs, info = env.reset()
print('OK: env.reset() works')
env.close()
"

# 100 局随机对战基准测试
source .venv/bin/activate
python penguinchess/random_ai.py
```
