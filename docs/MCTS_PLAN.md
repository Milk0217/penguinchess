# PenguinChess AI 优化方案：MCTS + 神经网络

> 日期: 2026-04-26

## 一、纯 PPO 的问题

当前 PPO 训练使用的策略是 `MlpPolicy`（多层感知机），直接从棋盘状态映射到动作概率。这种方法的根本局限：

| 问题 | 说明 | 影响 |
|------|------|------|
| **短视** | 只看当前观测，不做前瞻搜索 | 看不到 2 步后的局面变化 |
| **贪小利** | 偏向立即获得高分值格子 | 为了一两分破坏长期布局 |
| **不会布局** | 不理解"弃子争先"等策略概念 | 放置阶段表现差（Gen 对随机仅 ~50-60%） |
| **无法应对多步组合** | 无法计算"我走这步，对手会怎么应" | 面对稍强对手就崩盘 |

## 二、进化方案：MCTS + 神经网络

### 2.1 架构对比

```
纯 PPO（当前）:
  棋盘 → [NN] → 动作

PPO + MCTS（目标）:
  棋盘 → [NN 评估] → MCTS 搜索（模拟 N 步） → [NN 评估汇总] → 动作
```

MCTS（蒙特卡洛树搜索）在推理时做 800~1600 次模拟，每次模拟：
1. **选择**（Selection）：用 UCB 公式从根节点向下选择最有潜力的分支
2. **扩展**（Expansion）：到达叶节点时创建子节点
3. **模拟**（Simulation）：用神经网络评估节点价值（或用快速走子策略）
4. **回溯**（Backpropagation）：将结果更新到路径上所有节点

### 2.2 为什么 PPO + MCTS 效果好

| 能力 | 纯 PPO | PPO + MCTS |
|------|--------|------------|
| 前瞻深度 | 0 步（只看当前） | 可看 5-15 步 |
| 分支评估 | 只选网络认为好的 | 模拟大量分支后选最优 |
| 对手应对 | 不考虑 | 假设对手也会最优应对 |
| 布局能力 | 无 | 通过搜索找到局部最优序列 |
| 抗干扰 | 弱 | 搜索树天然过滤噪声 |

### 2.3 AlphaZero 式架构（推荐）

```
         ┌─────────────────────────┐
         │      棋盘状态 (204维)      │
         └──────────┬──────────────┘
                    ▼
         ┌─────────────────────────┐
         │    残差网络 (ResNet)      │
         │   ┌─────┐   ┌─────┐     │
         │   │Conv1│ → │Conv2│ ...  │
         │   └─────┘   └─────┘     │
         └──────────┬──────────────┘
                    ▼
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐     ┌────────────────┐
│  策略头 (Policy)│     │  价值头 (Value)  │
│  Discrete(60)  │     │     [-1, 1]     │
│   动作概率分布   │     │  局面评分（谁赢）  │
└───────────────┘     └────────────────┘
        ▼                       ▼
┌───────────────┐     ┌────────────────┐
│   MCTS 搜索    │────▶│   最终决策动作    │
│  800~1600次模拟 │     │   (访问次数最多)   │
└───────────────┘     └────────────────┘
```

### 2.4 与纯 PPO 的训练差异

| 维度 | 纯 PPO | AlphaZero 式 MCTS |
|------|--------|-------------------|
| 网络输出 | 策略 (60维) | 策略 (60维) + 价值 (1维) |
| 训练数据 | 自对弈轨迹 | MCTS 搜索后的增强轨迹 |
| 策略标签 | 实际走的动作 | MCTS 访问次数的分布 |
| 价值标签 | 稀疏奖励 (±1) | 最终胜负结果 |
| 损失函数 | Policy Gradient | Policy Loss + Value Loss + L2 |

## 三、实施路线图

### 第一阶段：基础设施（优先级 P0）

```python
# penguinchess/ai/mcts.py  — MCTS 核心
class MCTSNode:
    """MCTS 树节点。"""
    def __init__(self, state, parent=None, action=None):
        self.state = state      # 游戏状态快照
        self.parent = parent
        self.action = action    # 到达本节点的动作
        self.children = {}      # {action: MCTSNode}
        self.visits = 0
        self.total_value = 0.0
        self.prior = 0.0        # 神经网络给出的先验概率
    
    def ucb_score(self, c_puct=1.4):
        """UCB 公式"""
        if self.visits == 0:
            return float('inf')
        return self.total_value / self.visits + c_puct * self.prior * sqrt(parent.visits) / (1 + self.visits)
```

**依赖**: 需要 `PenguinChessCore` 支持**深拷贝**或**状态快照**，以便 MCTS 在不影响真实游戏状态的情况下模拟。

### 第二阶段：神经网络升级（优先级 P1）

将 `MlpPolicy` 替换为带价值头的网络：

```python
class AlphaZeroNet(nn.Module):
    def __init__(self, obs_dim=204, action_dim=60):
        # 共享特征提取层
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        # 策略头
        self.policy_head = nn.Linear(256, action_dim)
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # 输出 [-1, 1]
        )
```

### 第三阶段：AlphaZero 自对弈训练（优先级 P2）

```
每轮训练:
  1. 当前网络 vs 自己（MCTS 搜索）
  2. 保存 (状态, MCTS策略分布, 胜负) 到经验池
  3. 从经验池采样训练网络
  4. 定期与历史最佳对战，若胜率 > 55% 则更新最佳
```

### 第四阶段：整合到前端（优先级 P3）

MCTS 推理需要 GPU 或优化后的 C++ 实现，前端建议用异步 API：

```
前端 → POST /api/game/xxx/ai_move → 后端 MCTS 搜索 → 返回动作
```

## 四、预期效果

| 指标 | 当前（纯 PPO Gen9） | 目标（PPO + MCTS） |
|------|-------------------|-------------------|
| 对随机胜率 | ~62% | > 99% |
| 对前代胜率 | ~60% | > 95% |
| 棋力水平 | 业余初级 | 业余高级 |
| 布局能力 | 无 | 有初步布局概念 |
| 单步推理时间 | < 1ms | ~100-500ms（800 次模拟） |

## 五、风险与注意事项

1. **推理速度**：MCTS 800 次模拟约需 100-500ms（Python 实现），需要异步处理或优化
2. **状态拷贝**：`PenguinChessCore` 需要支持快速深拷贝，当前 `reset()` 不支持部分重置
3. **训练时间**：AlphaZero 自对弈训练需要更多数据，训练时间约纯 PPO 的 2-3 倍
4. **超参数敏感**：UCB 的 `c_puct`、模拟次数、温度参数需要调参
5. **替代方案**：如果 Python MCTS 太慢，可改用 **PPO + Beam Search**（更轻量）或 **PPO + 滚动地平线**（Rollout）
