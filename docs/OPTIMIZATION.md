# PenguinChess Core 性能优化分析

> 日期: 2026-04-25

## 一、概述

本文档分析 `penguinchess/core.py` 的性能问题，并记录实施的优化方案。

**优化目标**: 提升 RL 训练场景下的核心游戏逻辑性能，特别是 `get_legal_actions()` 和 `step()` 方法的调用效率。

---

## 二、性能分析

### 2.1 现有优化 ✅

| 优化点 | 实现 | 效果 |
|--------|------|------|
| `_hex_map` | `(q,r,s) → Hex` 字典 | O(1) 坐标查找 |
| `_occupied_set` | Hex 对象集合 | O(1) 占据状态查询 |
| `__hash__`/`__eq__` | Hex 对象可哈希 | 支持集合操作 |
| 迭代 flood_fill | 栈而非递归 | 避免栈溢出 |

### 2.2 关键问题 ❌

#### 问题 1: `_hex_map` 用法错误 (严重)

**位置**: `core.py` line 389-391

```python
idx = self._hex_map.get((target.q, target.r, target.s))
if idx is not None:
    ids.add(self.hexes.index(target))  # ❌ O(n) 线性查找！
```

**问题**: `_hex_map` 存储的是 Hex 对象，但 `hexes.index(target)` 是 O(n) 线性搜索。整个 `_hex_map` 形同虚设。

**影响**: `get_legal_actions()` 移动阶段 O(n²) 复杂度。

#### 问题 2: 大量 O(n) 遍历

| 位置 | 操作 | 复杂度 | 调用频率 |
|------|------|--------|----------|
| line 375-378 | 放置阶段遍历 hex | O(n) | 每步 |
| line 647-668 | `_get_piece_moves` 遍历 hex | O(n) | 每棋子每步 |
| line 753-761 | 消除阶段遍历 hex | O(n) | 每移动步 |
| line 778, 792 | `any()` 检查存活 | O(n) | 每步 |
| line 821-824 | `_hex_has_player_piece` | O(pieces) | 频繁 |

#### 问题 3: 无 `__slots__`

**位置**: `Hex` 和 `Piece` dataclass

```python
@dataclass  # ← 没有 slots=True
class Hex:
    q, r, s, value, _q_raw  # 每个实例都有 __dict__
```

**影响**: 每个 Hex 实例 ~56 bytes 额外开销，60 hexes × 6 pieces ≈ 22KB 冗余。

---

## 三、优化方案

### 3.1 修复 `_hex_map` (P0 - 关键) ✅ 已完成

**方案**: 存储 `index` 而非 `Hex` 对象

```python
# 修改后（已实施）
self._hex_map[(h.q, h.r, h.s)] = index  # 存储索引
```

**收益**: `get_legal_actions()` 从 O(n²) → O(n)

### 3.2 添加 `__slots__` (P1)

**方案**: 使用 frozen dataclass 或手动 `__slots__`

```python
@dataclass(slots=True)
class Hex:
    q: int
    r: int
    s: int
    value: int
    _q_raw: int
```

**注意**: Python 3.10+ 支持 dataclass `slots`

**收益**: 内存减少 ~30%，访问速度提升

### 3.3 预计算邻居关系 (P1) ✅ 已完成

**方案**: 在 `create_board` 时计算每个 Hex 的邻居索引列表

```python
self._neighbors: List[List[int]]  # hex_idx → [neighbor_idx, ...]
```

**收益**: 减少 `_path_clear` 和 flood_fill 的重复遍历

### 3.4 观测编码向量化 (P2)

**方案**: 使用 NumPy 替代 Python list

```python
def _encode_board(self) -> np.ndarray:
    # 直接返回 numpy array
```

**收益**: RL 环境观测创建更快

---

## 四、优化优先级

| 优先级 | 优化项 | 预期收益 | 实施状态 |
|--------|--------|----------|----------|
| P0 | 修复 `_hex_map` 存 index | O(n²) → O(n) | ✅ 已完成 |
| P1 | 预计算邻居关系 | 减少重复遍历 | ✅ 已完成 |
| P1 | 添加 `__slots__` | 内存 -30% | ⬜ 待实施 |
| P2 | NumPy 观测编码 | 向量化加速 | ⬜ 未来 |

---

## 五、性能测试

### 5.1 测试用例

```python
# 性能测试
import time
from penguinchess.env import PenguinChessEnv

env = PenguinChessEnv()
iterations = 10000

# 测试 reset
start = time.perf_counter()
for _ in range(iterations):
    env.reset()
print(f"reset: {time.perf_counter() - start:.3f}s / {iterations} iters")

# 测试随机 step
env.reset()
start = time.perf_counter()
steps = 0
for _ in range(iterations):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
    if terminated or truncated:
        env.reset()
print(f"step: {time.perf_counter() - start:.3f}s / {steps} steps")

# 测试 get_legal_actions
env.reset()
start = time.perf_counter()
for _ in range(iterations * 10):
    legal = env._game.get_legal_actions()
print(f"get_legal_actions: {time.perf_counter() - start:.3f}s")
```

### 5.2 验收标准

| 指标 | 优化前 | 优化后目标 |
|------|--------|------------|
| `get_legal_actions()` | ~0.5ms | < 0.1ms |
| 单步 `step()` | ~1ms | < 0.5ms |
| 10000 步训练 | ~10s | < 5s |

---

## 六、已知风险

### 6.1 `__slots__` 兼容性

- Python 3.10+ 支持 dataclass `slots`
- 如果使用更低版本，需要手动实现 `__slots__`

### 6.2 邻居预计算的内存占用

- 60 hexes × 6 neighbors × 4 bytes ≈ 1.4KB
- 可接受

---

## 七、GPU 显存优化

### 7.1 问题

`train_alphazero.py` 在 RTX 4060（8GB）上训练旧版 `AlphaZeroResNet`（550K 参数）时，GPU 显存仅占用 **29MB**，远端未充分利用。

### 7.2 方案

重构网络架构为可配置 ResNet：
1. `AlphaZeroResNet`：550K 参数，~25MB 显存（兼容旧模型）
2. `AlphaZeroResNetLarge`：3M 参数，~74MB 显存（默认训练选项）
3. `AlphaZeroResNetXL`：581M 参数，~3GB 推理 / ~8GB 训练显存（最大）
4. `AlphaZeroResNetConfigurable`：自定义 `hidden_dim` 和 `num_blocks`

### 7.3 收益

| 模型 | 推理显存 | 参数量 | 迭代耗时 |
|------|---------|--------|---------|
| `AlphaZeroResNet` | ~25MB | 550K | ~48s |
| `AlphaZeroResNetLarge` | ~74MB | 3M | ~51s |
| `AlphaZeroResNetXL` | ~3GB | 581M | ~4.5h |

Large 模型仅增加 ~6% 迭代时间，显存翻倍。XL 模型虽达到 ~3GB 显存但训练耗时长，适合长时间的离线训练。

### 7.4 数据精度

训练使用 `torch.amp.autocast`（fp16 mixed precision），参数以 fp16 存储，梯度以 fp32 累积。激活值在 CUDA 上以 fp16 计算，减少显存占用和带宽。

## 八、变更记录

| 日期 | 变更内容 | 状态 |
|------|----------|------|
| 2026-04-25 | 创建优化分析文档 | ✅ |
| 2026-04-25 | 修复 `_hex_map` 存 index | ✅ 已完成 |
| 2026-04-25 | 预计算邻居关系 | ✅ 已完成 |
| 2026-04-25 | 添加 `__slots__` | ⬜ 待实施 |
| 2026-04-27 | 文档同步更新 | ✅ |
| 2026-04-28 | GPU 显存优化：可配置 ResNet 网络 | ✅ |