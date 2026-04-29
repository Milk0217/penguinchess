# PenguinChess 技术参考

> 本项目使用的搜索算法、神经网络和优化技术完整说明。
> **引入新技术/算法前，必须先补充本文档相关章节。**

---

## 目录

1. [NNUE（Efficiently Updatable Neural Network）](#1-nnue)
2. [Alpha-Beta 搜索](#2-alpha-beta-搜索)
3. [PVS（Principal Variation Search）](#3-pvs)
4. [Transposition Table（置换表）](#4-transposition-table)
5. [Move Ordering（走法排序）](#5-move-ordering)
6. [LMR（Late Move Reduction）](#6-lmr)
7. [Null-Move Pruning（空着裁剪）](#7-null-move-pruning)
8. [Aspiration Windows（期望窗口）](#8-aspiration-windows)
9. [Iterative Deepening（迭代加深）](#9-iterative-deepening)
10. [Lazy SMP](#10-lazy-smp)
11. [History Heuristic（历史启发式）](#11-history-heuristic)
12. [Killer Moves（杀着）](#12-killer-moves)
13. [MCTS（Monte Carlo Tree Search）](#13-mcts)
14. [AlphaZero 训练](#14-alphazero-训练)
15. [AVX2 SIMD 优化](#15-avx2-simd-优化)
16. [FFI 架构](#16-ffi-架构)
17. [自蒸馏训练](#17-自蒸馏训练)
18. [IID（Internal Iterative Deepening）](#18-iid)
19. [MCTS + NNUE（AlphaZero 风格训练）](#19-mcts--nnuealphazero-风格训练)

---

## 1. NNUE

### 概述

NNUE（Efficiently Updatable Neural Network）是一种专为博弈 AI 设计的轻量级神经网络。核心思想是用**稀疏特征 + 增量累积器**替代传统的前向传播，实现 O(1) 评估更新。

### 架构（当前）

```
Sparse Features (360-dim)        Dense Features (66-dim)
        │                                │
        ▼                                │
Feature Transformer (360→128)            │
  增量累积器 (stm + nstm)                │
        │                                │
        ▼                                ▼
   CReLU(stm_acc || nstm_acc)      ──── concat ──── → 256-dim
                             │
                             ▼
                     FC1 (256→128) + CReLU
                             │
                             ▼
                     FC2 (128→128) + ReLU
                             │
                             ▼
                     FC3 (128→1) + tanh
                             │
                             ▼
                     Score ∈ [-1, 1]
```

| 层 | 参数 | FMA | 说明 |
|----|------|-----|------|
| FT | 360×128 | 46,080 | 稀疏累积，增量更新 |
| FC1 | 256×128 | 32,768 | CReLU 激活 |
| FC2 | 128×128 | 16,384 | ReLU 激活 |
| FC3 | 128×1 | 128 | Tanh 输出 |
| **总计** | **~106K** | **~83K** | |

### 特征设计

**稀疏特征**（360-dim，实际活跃 0-6 个）：
```
piece_idx × 60 + hex_idx
  - piece_idx: 0..5 对应棋子 ID [4,6,8,5,7,9]
  - hex_idx: 0..59 对应棋盘格子
  - P1 = [0,60,120,180,240,300] → 索引 0-179
  - P2 = [1,61,121,...] → 索引 180-359
```

**稠密特征**（66-dim）：
- 0-59: 60 个格子分值（归一化到 [0,1]，已消除为 0）
- 60-61: P1/P2 分数 / 100
- 62: 游戏阶段（0=放置，0.5=移动）
- 63-64: P1/P2 存活棋子数 / 3
- 65: 步数 / 500

### 增量累积器

**核心数据结构**：
```rust
struct NNUEAccumulator {
    stm: Vec<f32>,   // 当前走棋方的 FT 输出
    nstm: Vec<f32>,  // 对手方的 FT 输出
}
```

**操作**（均 O(1)）：
- `apply_diff(removed, added, player)`: 移除/添加特征后更新累积器
- `snapshot()`: 复制累积器状态（搜索树回溯用）
- `restore(snap)`: 恢复累积器状态

**评估**：给定累积器，只需计算：
```rust
CReLU(stm || nstm) → FC1 → CReLU → FC2 → ReLU → FC3 → tanh
```
跳过 FT（最耗时的 46K FMA）。

### 前向传播（Rust AVX2）

`nnue_evaluate(sparse, dense, stm, weights) → f32`

```
1. FT: ft_w[sparse].sum(dim=0) + ft_b  →  stm_acc, nstm_acc
2. CReLU: clip(acc, 0, 127) × 2 / 127 - 1  →  (128 + 128) = 256-dim
3. FC1: 256×128 FMA + CReLU → 128-dim
4. FC2: 128×128 FMA + ReLU → 128-dim
5. FC3: 128×1 FMA + tanh → score [-1, 1]
```

全部使用 AVX2 指令集（`_mm256_fmadd_ps`, `_mm256_max_ps`, `_mm256_min_ps`）。

### 参考

- [NNUE 原始论文 - Stockfish NNUE](https://github.com/official-stockfish/Stockfish/tree/master/NNUE)
- 实现位置: `game_engine/src/nnue_rs.rs`, `penguinchess/ai/nnue.py`

---

## 2. Alpha-Beta 搜索

### 概述

Alpha-Beta 是 Minimax 的优化版本，通过**裁剪不可能影响最终决策的分支**，将搜索复杂度从 O(b^d) 降到 O(b^(d/2))。

### Negamax 框架

所有评估值都从**当前走棋方**的视角计算：

```rust
fn negamax(state, depth, alpha, beta) -> score:
    if depth == 0 or terminal:
        return evaluate(state)  // 当前走棋方视角
    
    for each move in order_moves(state):
        make_move(state, move)
        score = -negamax(state, depth-1, -beta, -alpha)
        undo_move(state, move)
        
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break  // Beta 裁剪
            
    return alpha
```

### 搜索流程

```rust
pub fn search(state) -> SearchResult {
    // 根节点走法排序（NNUE）
    let children = order_children(state, depth_from_root=0);
    
    // 迭代加深
    for depth in 1..=max_depth {
        // 期望窗口
        let (alpha, beta) = aspiration_window(best_score, depth);
        
        // PVS 搜索
        let result = negamax(state, depth, alpha, beta, is_pv=true);
        
        // 窗口失败 → 全窗口重搜
        if score <= alpha || score >= beta:
            result = negamax(state, depth, -INF, +INF, true);
    }
}
```

### 实现位置

`game_engine/src/alphabeta_rs.rs` — `SearchContext::negamax()`

### 参考

- Knuth & Moore (1975). "An Analysis of Alpha-Beta Pruning"
- [Chess Programming Wiki: Alpha-Beta](https://www.chessprogramming.org/Alpha-Beta)

---

## 3. PVS

### 概述

PVS（Principal Variation Search，主变例搜索）在 Alpha-Beta 的基础上进一步优化：**假设第一个走法（PV 走法）是最好的，其余走法用零窗口侦察**。

### 算法

```rust
for (i, child) in children.iter().enumerate() {
    if i == 0 || is_pv_node:
        // 全窗口搜索（PV 节点）
        score = -negamax(state, depth-1, -beta, -alpha)
    else:
        // 零窗口侦察
        score = -negamax(state, depth-1, -alpha-ε, -alpha)
        
        // 如果侦察证明搜索有潜力 → 全窗口重搜
        if score > alpha && score < beta:
            score = -negamax(state, depth-1, -beta, -alpha)
}
```

### 效果

| 走法质量 | 全窗口搜索 | 零窗口搜索 | 重搜概率 |
|---------|-----------|-----------|---------|
| TT 最佳 | ✅ PV | — | — |
| NNUE 前 3 | — | ✅ | < 10% |
| 历史前 5 | — | ✅ | < 20% |
| 其余 | — | ✅ (LMR) | < 5% |

零窗口搜索比全窗口快 2-3 倍（更紧凑的 α-β 边界 → 更多裁剪）。

### 实现位置

`alphabeta_rs.rs` — `negamax()` 中的 `is_pv` 分支

---

## 4. Transposition Table

### 概述

置换表（TT）缓存已搜索过的局面及其结果，避免重复搜索。

### 哈希

```rust
fn hash_state(state) -> u64 {
    for piece in state.pieces: hash(piece.id, piece.hex_idx, piece.alive)
    for cell in state.board.cells: hash(cell.state, cell.points)
    hash(state.current_player)
}
```

使用 Rust 的 `DefaultHasher`（SipHash-1-3）。冲突概率极低（1/2^64）。

### 条目

```rust
struct TTEntry {
    depth: u8,       // 搜索深度
    score: f32,      // 评估值
    flag: TTFlag,    // Exact / LowerBound / UpperBound
    best_move: usize,// 最佳走法
    age: u32,        // 世代标记
}
```

| Flag | 含义 | 用途 |
|------|------|------|
| Exact | 精确值 | 直接使用 |
| Lower | Alpha 提升（score ≥ beta） | 节点已裁剪 |
| Upper | 未提升（score ≤ alpha） | 可剪枝 |

### 世代管理

每次 `search()` 调用为一个世代。`age` 递增。查找时仅匹配当前世代，确保缓存数据在同一次搜索中有效。

### 线程安全（Lazy SMP）

```rust
pub struct SharedTT {
    inner: RwLock<TranspositionTable>,
}
```

- `lookup()` → 读锁（多个线程可同时读）
- `store()` → 写锁（独占写入）

### 实现位置

`alphabeta_rs.rs` — `TranspositionTable`, `SharedTT`

---

## 5. Move Ordering

### 概述

Alpha-Beta 的效率极度依赖走法排序——好的走法先搜，裁剪效率最大化。

### 排序优先级（由高到低）

| 优先级 | 方法 | 适用深度 | 实现 |
|--------|------|---------|------|
| 1 | TT 最佳走法 | 全部 | `tt_best` 标记最高分 |
| 2 | NNUE 评估排序 | ≤ nnue_order_depth | 克隆+步进+NNUE 评估 |
| 3 | 杀着（Killer） | 全部 | 固定在 depth slot |
| 4 | 历史启发式 | > nnue_order_depth | `history[action]` 累积分数 |

### NNUE 排序（浅层节点）

```rust
// depth_from_root <= nnue_order_depth
for move in legal_moves:
    child = state.clone(); child.step(move)
    eval = nnue_evaluate(child)
    score = reward + eval
sort_descending(score)
```

每个子节点**克隆一次**步进后的状态，结果在搜索循环中**复用**（单克隆优化）。

### 启发式排序（深层节点）

```rust
// depth_from_root > nnue_order_depth
for move in legal_moves:
    score = 0
    if move == tt_best:          score += 10_000_000
    if move == killer[depth]:    score += 5_000_000
    score += history[move]
sort_descending(score)
```

**不克隆、不步进、不 NNUE 评估**——纯查表，亚微秒级。

### nnue_order_depth 策略

| 局面 | nnue_order_depth | 效果 |
|------|-----------------|------|
| 战术激烈（高分格子 + 棋子多） | base（默认 2） | 精确排序 |
| 平静（棋子 ≥ 3） | base.min(2) | 适度精确 |
| 残局（棋子 < 3） | 1 | 快速 |

### 实现位置

`alphabeta_rs.rs` — `order_children()`, `nnue_ordering_depth()`

---

## 6. LMR

### 概述

Late Move Reduction（延迟走法降级）对排序靠后的走法减少搜索深度。假设排序正确的走法中，靠后的走法不太可能成为最佳走法。

### 参数

```rust
pub struct SearchConfig {
    lmr_moves: u8,   // 前 N 个走法不降级（默认 3）
    lmr_depth: u8,   // 降级减去的深度（默认 1）
}
```

### 条件

触发：`depth > 2 && move_index >= lmr_moves`

```rust
if is_lmr:
    reduced = (depth - 1 - lmr_depth).max(1)
    score = -negamax(state, reduced, -alpha-ε, -alpha)
    if score > alpha:
        score = -negamax(state, depth-1, -beta, -alpha)  // 重新搜索
```

### 效果

在所有走法上节约约 30-50% 的搜索节点。

---

## 7. Null-Move Pruning

### 概述

如果当前局面好到"即使让对手免费走一步（空着）都不会输"，则必然不会输。

### 条件

```rust
null_move_enabled && depth >= 3
    && !is_pv
    && phase == Movement
    && alive_pieces >= 2
```

### 算法

```rust
let mut null_state = state.clone();
null_state.current_player = 1 - null_state.current_player;  // 让对手走
let (score, _) = negamax(null_state, depth - 1 - R, -beta, -beta + 1, false);

if -score >= beta:
    return beta;  // 裁剪
```

### R 参数

`R = 2`（降两级深度）。空着搜索本身就是近似，浅搜索已足够提供裁剪信号。

### 效果

节点数减少约 2-5x，尤其在中局阶段。

---

## 8. Aspiration Windows

### 概述

每次迭代使用上一层最佳分值附近的窄窗口搜索，而非全窗口 [-1, 1]。窄窗口→更多裁剪→更快搜索。

### 实现

```rust
if depth >= 3 && best_score.abs() > 0.01:
    window = (0.5 - 0.04 * depth).max(0.1)
    alpha = (best_score - window).max(-1.0)
    beta  = (best_score + window).min(1.0)
else:
    alpha, beta = -1.0, 1.0

result = negamax(state, depth, alpha, beta, true)
if result <= alpha:  // 窗口过窄 → 全窗口重搜
    result = negamax(state, depth, -1.0, beta, true)
if result >= beta:   // 窗口过高 → 全窗口重搜
    result = negamax(state, depth, alpha, 1.0, true)
```

---

## 9. Iterative Deepening

### 概述

从深度 1 开始逐层加深搜索。每层结果（最佳走法、TT 条目）为下一层提供**走法排序信息**。

### 流程

```
Depth 1: 搜索 19 个根走法 → 最佳走法 A, 存入 TT
Depth 2: TT best = A → A 优先搜索 → 更多裁剪
Depth 3: TT best = A → 更多裁剪
...
```

### 效果

- TT 条目自动为下一层提供排序
- 随时可中止（有最佳走法）

---

## 10. Lazy SMP

### 概述

Lazy SMP（懒惰对称多处理）是最简单的并行搜索方法：**多个线程独立搜索同一局面，共享置换表**。

### 架构

```rust
pub fn search_lazy_smp(&self, state) -> SearchResult {
    let tt = SharedTT::new(config.tt_size);  // 共享 TT
    
    std::thread::scope(|s| {
        for tid in 0..num_threads {
            s.spawn(|| {
                // 每个线程拥有独立的状态
                let mut ctx = SearchContext {
                    tt: &shared_tt,       // 共享
                    history: HistoryTable::new(),  // 独立
                    killers: KillerTable::new(),   // 独立
                    nodes_searched: 0,     // 独立
                    weights, config,       // 只读的共享
                };
                ctx.search_root(state);
            });
        }
    });
}
```

### 为什么共享 TT 即可？

| 线程 A 找到 | 写入 TT | 线程 B 读取 | 效果 |
|------------|---------|------------|------|
| 深度 2 最佳 | `(hash, 2, Exact, score, move)` | 深度 2 命中 | 跳过重复搜索 |
| 裁剪信号 | `(hash, 3, Lower, score, move)` | 深度 3 命中 | 线程 B 裁剪 |
| 未裁剪信号 | `(hash, 4, Upper, score, 0)` | 深度 4 命中 | 线程 B 提前中止 |

线程间通过 TT 条目**自动通信**。各线程搜索顺序因 `thread::spawn` 时序不同而自然差异化（搜索树不同分支先探索）。

### 配置

```rust
SearchConfig { num_threads: 2 }  // 2 线程 Lazy SMP
```

推荐 2-4 线程。超过 CPU 核心数后 TT 争用（RwLock）降低收益。

### 实现位置

`alphabeta_rs.rs` — `AlphaBetaSearch::search_lazy_smp()`, `SearchContext`

---

## 11. History Heuristic

### 概述

记录**哪些走法在搜索中触发了 Beta 裁剪**。裁剪频率越高的走法越可能好。

### 数据

```rust
struct HistoryTable {
    scores: [i32; 60],  // 每个格子走法的累积分数
    max_: i32,           // 归一化分母
}
```

`60` 对应 60 个格子（棋盘动作数）。

### 更新

```rust
// 在 beta 裁剪时更新
if alpha >= beta:
    history.update(current_move, depth_remaining);
```

更新量 = `depth^2`。深度越深的分支，其裁剪走法的可信度越高。

### 使用

```rust
// 深层节点的启发式排序
history.get(action) as f32 / history.max_ as f32 * 300.0
```

归一化后作为排序分数的一部分（低于 TT 和 Killer，高于默认）。

---

## 12. Killer Moves

### 概述

Killer Moves（杀着）是同一深度层级上经常引发裁剪的走法。与历史启发式不同，Killer 关注**走法本身**而非局面。

### 数据

```rust
struct KillerTable {
    moves: Vec<[Option<usize>; 2]>,  // 每层 2 个杀着
}
```

每层保留 2 个杀着（替换策略：新杀着替换 slot 1，原有 slot 1 降级到 slot 0）。

### 更新

```rust
if alpha >= beta && !is_pv:
    killers.add(depth_from_root, current_move);
```

### 使用

```rust
if killers.is_killer(depth_from_root, action):
    score += 5_000_000  // 启发式排序中仅次于 TT
```

---

## 13. MCTS

### 概述

Monte Carlo Tree Search（蒙特卡洛树搜索）通过随机模拟构建搜索树。AlphaZero 风格 MCTS 使用神经网络指导搜索。

### 算法流程

```
Selection → Expansion → Simulation(→ NN) → Backpropagation
```

### UCB 公式

```rust
fn ucb_score(node, parent_visits) -> f32 {
    let exploration = C_PUCT * node.policy_prob *
        sqrt(parent_visits) / (1 + node.visits);
    node.value + exploration
}
```

`C_PUCT = 1.4`（默认）。

### 并行 MCTS（Rust 内部）

```rust
pub fn mcts_search_parallel_core(
    games: &mut [GameState],  // 8 个并行游戏
    num_simulations: usize,
    c_puct: f32,
    eval_fn: &EvalFn,
) -> Vec<(f32, f32)>  // (value, action_probs)
```

`std::thread::scope` 内部并行：每个 worker 独立运行模拟，共享根节点的统计数据。

### 架构

```
Python: train_alphazero.py
  │
  ├─ RustCore (GameState in Rust)
  │    └─ mcts_search_rust_handle(handle, n_workers, n_sims)
  │         └─ Rust 内部 std::thread::scope
  │              └─ 每 worker 独立模拟
  │                   └─ 需要 NN 评估时回调 Python
  │
  └─ Python EvalFn
       └─ model.forward(obs) → (policy, value)
```

### 评估回调

```python
# Python → Rust 的评估回调
def eval_fn(obs_ptr, batch_size, output_ptr, output_capacity):
    obs = np.frombuffer(obs_ptr, dtype=np.float32, count=batch_size * 206)
    obs = obs.reshape(-1, 206)
    policy_logits, value = model(obs)
    # 写入 output_ptr
```

### 实现位置

- `game_engine/src/mcts_rs.rs` — 搜索核心
- `penguinchess/mcts_core.py` — Python 包装器
- `game_engine/src/ffi.rs` — `mcts_search_rust_handle` FFI 导出

---

## 14. AlphaZero 训练

### 概述

AlphaZero 风格自对弈训练：MCTS + 神经网络自我博弈，策略损失 + 价值损失的联合训练。

### 训练循环

```python
for iteration in range(num_iterations):
    # 1. Self-play: 使用 MCTS 生成对局数据
    games = self_play(model, num_games)  # (state, mcts_policy, winner) 元组
    
    # 2. 训练: 从 replay buffer 采样，更新模型
    for batch in replay_buffer:
        policy_loss = cross_entropy(mcts_policy, model_policy)
        value_loss = mse(winner, model_value)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()
    
    # 3. 评估: vs 前代模型
    win_rate = evaluate(model_new, model_old)
    if win_rate >= 0.55:
        save_as_best(model_new)
```

### 网络架构

| 架构 | 参数 | 推理位置 |
|------|------|---------|
| AlphaZeroMLP | 344K | `az_model.rs` (MLP) |
| AlphaZeroResNet | 550K | `az_model.rs` (ResNet) |
| AlphaZeroResNetLarge | 1.95M | `az_model.rs` |
| AlphaZeroResNetXL | 312M | `az_model.rs` |

### Rust 推理（BN 折叠）

训练时的 BatchNorm 在推理时折叠到 Linear 层：

```rust
fn fold_bn(weight, bias, bn_weight, bn_bias, bn_mean, bn_var) -> (weight', bias') {
    let scale = bn_weight / sqrt(bn_var + epsilon);
    weight' = weight * scale   // 广播乘
    bias' = (bias - bn_mean) * scale + bn_bias
}
```

折叠后推理路径无 BN 开销。

### 实现位置

- `examples/train_alphazero.py` — 训练入口
- `penguinchess/alphazero_net.py` — PyTorch 网络定义
- `game_engine/src/az_model.rs` — Rust 推理

---

## 15. AVX2 SIMD 优化

### 概述

所有 Rust 端矩阵运算使用 AVX2 指令集，通过 `target-cpu=native` 编译启用。

### 核心指令

| 指令 | 函数 | 用途 |
|------|------|------|
| `_mm256_fmadd_ps(a, b, c)` | `a*b + c` (8 × f32) | FMA 矩阵乘 |
| `_mm256_loadu_ps(p)` | 加载 8 个 f32（不对齐） | 权重复制 |
| `_mm256_storeu_ps(p, v)` | 存储 8 个 f32 | 结果写入 |
| `_mm256_max_ps(a, b)` | 逐元素 max | ReLU |
| `_mm256_min_ps(a, b)` | 逐元素 min | CReLU 上限 |
| `_mm256_add_ps(a, b)` | 逐元素加 | 累积器更新 |
| `_mm256_sub_ps(a, b)` | 逐元素减 | 累积器回退 |

### FMA 矩阵乘（转置权重）

```rust
// 权重矩阵 W 按列主序存储：W_t[col * rows + row] = W[row][col]
// 结果为 y = W_t · x + bias（一次 fma_add 处理 8 个输出元素）
for k in 0..cols:       // 对输入向量的每个元素
    let xk = splat(x[k])
    j = 0
    while j + 8 <= rows:  // 每次处理 8 个输出
        fma_add(&w_t[j + k*rows], xk, &mut y[j])
        j += 8
    for j..rows:          // 剩余元素
        y[j] += w_t[j + k*rows] * x[k]
```

### 构建配置

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release
```

### 实现位置

`game_engine/src/nnue_rs.rs` — `matvec_mul_add_t()`, `get_crelu()`, `relu_inplace()`

---

## 16. FFI 架构

### 概述

Python ↔ Rust 通信通过 C ABI，Python 端使用 `ctypes`。

### 数据流

```
Python: ctypes.c_char_p(state_json)  →  Rust: *const c_char
Python: ctypes.create_string_buffer() ←  Rust: *mut c_char (JSON 输出)
Python: ctypes.c_int32(handle)       →  Rust: i32 (handle)
```

### 协议

所有复杂数据结构以 JSON 字符串传输。简单类型（int、float）直接传值。

### 句柄管理

```rust
static mut AB_SEARCHES: Vec<Option<AlphaBetaSearch>> = Vec::new();
static mut GAMES: Vec<Option<GameState>> = Vec::new();
```

每个 `*_create` 函数返回一个 handle（vector index），后续 `*_search`/`*_step` 通过 handle 访问。

### Python 包装

`penguinchess/rust_ffi.py` — 完整的 ctypes 包装器，自动加载 DLL 并设置 `argtypes`/`restype`。

### 重要函数

| 函数 | 参数 | 返回值 |
|------|------|--------|
| `ffi_ab_create(config_json)` | 配置 JSON | handle (i32) |
| `ffi_ab_set_weights(handle, weights)` | handle + 权重字典 | 0=成功 |
| `ffi_ab_search(handle, state_json, max_depth)` | state JSON | 结果 JSON |
| `ffi_ab_destroy(handle)` | handle | — |
| `ffi_az_create(config_json)` | 模型配置 | handle |
| `ffi_az_set_weights(handle, weights)` | 权重字典 | 0=成功 |
| `ffi_az_evaluate(handle, obs_batch)` | obs 数组 | (policy, value) |

---

## 17. 自蒸馏训练

### 概述

NNUE 使用 Alpha-Beta 搜索的评分作为训练标签，而非终局胜负结果。

### 训练流程

```
1. 加载当前 NNUE 模型
2. 使用 NNUE + Alpha-Beta 搜索自对弈（depth 2-5）
3. 记录每步的：
   - sparse features（稀疏特征 6-dim）
   - dense features（稠密特征 66-dim）
   - AB search score（搜索评分，∈[-1, 1]）
4. 训练 NNUE 以 MSE 拟合 search score
5. 新一代 NNUE 模型用于下次自对弈
```

### 为什么用搜索评分而非胜负

| 标签类型 | 信号强度 | 噪声 | 适用 |
|---------|---------|------|------|
| 终局胜负（±1） | 弱 | 高 | 数据量极大 |
| 搜索评分（[-1,1]） | **强** | 低 | 数据量中等 |

搜索评分提供了**连续的梯度信号**，而非离散的 ±1。模型可从评分中学到"这个局面略好"和"这个局面大优"的区别。

### 数据格式（二进制）

Rust 原生生成（`ffi_ab_generate_random_data`）的二进制文件格式：

```
[header: u64 record_count]
[record_0: 6×i32 sparse_padding=-1, 66×f32 dense, 1×f32 score]
[record_1: ...]
```

每条记录 = 292 字节。

### 效果

| 训练数据 | val_loss | vs Random |
|---------|---------|-----------|
| 终局胜负（50K 局） | 0.97 | ~70% |
| 搜索评分（50 局 × depth 2） | **0.12** | ~70% |

### 局限：只有价值信号，没有策略信号

AB+NNUE 自蒸馏的模型只能学到**评分**（value），学不到**策略**（policy）。

| 信号类型 | 内容 | 来源 | 损失函数 |
|---------|------|------|---------|
| 价值 Value | 局面评分 ∈[-1, 1] | AB search score | MSE |
| 策略 Policy | 走法概率分布 | — | — |
| 策略 Policy | 走法概率分布 | **MCTS** — 访问次数 | **CrossEntropy** |

AB 搜索只返回**一个数字**（最优走法的评分），不返回搜索树中各节点访问的概率。NNUE 只能拟合这个数字，学不到"每个合法走法的相对优劣"。

所以 AB+NNUE 的模型在搜索中仍然是**评分员**——NNUE 说"这个位置好/坏"——但走法选择完全靠 AB 搜索的剪枝逻辑，NNUE 没有提供任何走法倾向。

### Rust 原生数据生成（ffi_ab_generate_random_data）

为了加速数据生成，提供了一个 Rust 原生的函数，完全在 Rust 侧完成数据生成全过程：

```
Rust: [随机对弈 + AB 搜索评分 + 写入二进制文件]
  ↑ FFI                                    ↓ Python
ffi_ab_create + set_weights      训练 NNUE
  (一次调用)                        (读文件)
```

**性能对比**（300 局，depth 4，4 workers）：

| 方案 | 时间 | 速度比 |
|------|------|--------|
| Python 循环（之前的 train_nnue_selfplay.py） | 64 min | 1x |
| **Rust 原生**（ffi_ab_generate_random_data） | **8 min** | **8x** |

Rust 版本加速来源：
1. **无 JSON 序列化**：搜索直接操作 Rust GameState，不经过 JSON
2. **无 FFI 每步调用**：一次 `ffi_ab_generate_random_data` 做完整个流程
3. **Rust 内线程并行**：`std::thread::scope` 做游戏级并行，无 GIL

### 实现位置

- `examples/train_nnue_selfplay.py` — 自蒸馏训练（Python 数据生成）
- `scripts/train_nnue_rust_data.py` — Rust 原生数据训练脚本
- `scripts/gen_data_rust.py` — Rust 数据生成示例
- `penguinchess/ai/nnue_train.py` — NNUE 训练工具
- `game_engine/src/ffi.rs` — `ffi_ab_generate_random_data`（Rust 原生数据生成）
- `penguinchess/rust_ffi.py` — `ffi_ab_generate_random_data` Python 包装

### 现状总结

| 模型 | 数据源 | 信号 | vs Random |
|------|--------|------|-----------|
| gen_1 | AB depth 2 自对弈 | value only | 56% |
| gen_2 | AB depth 4 自对弈 | value only | **77%** | 
| gen_random | AB depth 4 随机 + 评分 | value only | 63% |

**关键结论**：AB 自蒸馏只能学到价值信号（value），无法提供策略信号（policy）。这是架构性限制，不是数据量问题——gen_2 的 77% 是当前架构的上限。

---

## 18. IID（Internal Iterative Deepening）

### 概述

内部迭代加深（IID）：当搜索树中某个深度节点在置换表中没有命中时，先做一次**浅层搜索**来填充 TT，为完整深度搜索提供走法排序信息。

与第 9 节的"迭代加深"不同：
- **迭代加深**：根节点逐层加深（D1→D2→D3...），为下一层提供 PV
- **IID**：搜索树内部节点在 TT 缺失时做浅搜（depth/2），为本层提供排序

### 触发条件

```rust
tt_best.is_none() && depth >= 4 && !is_pv && !terminal
```

- `tt_best.is_none()`：该局面没有 TT 最佳走法
- `depth >= 4`：深度足够才有必要（浅层搜索代价低）
- `!is_pv`：PV 节点已有上一层的走法信息

### 算法

```rust
// IID: 对当前局面做浅层搜索（深度减 2）
let iid_target = max(depth - 2, 1);
let mut clone = state.clone();
negamax(&mut clone, iid_target, -beta, -alpha, false);

// 从 TT 中获取最佳走法（现在应该有了）
tt_best = get_best_move(state).or(tt_best);
```

**注意**：IID 在 CLONED 状态上运行，不修改原始 `state`。

### 为什么需要

在深层搜索（depth ≥ 8）中，TT 缺失的频率增加，因为：

| 节点深度 | TT 命中率 | 无 IID 排序 | 有 IID 排序 |
|---------|-----------|------------|------------|
| 1-3 | >90% | TT best available | — |
| 4-5 | ~70% | 历史+杀着 | TT best |
| **6-8** | **~40%** | 纯启发式 | **TT best ✅** |
| 9+ | <30% | 几乎随机 | TT best ✅ |

IID 的代价是一次浅层搜索（约 `O(b^(d/2))` 节点），但收益是**更好的排序 → 更多裁剪**。在深度 8+ 时，收益远超代价。

### 实现位置

`alphabeta_rs.rs` — `SearchContext::negamax()` 中 `Internal Iterative Deepening` 注释块

---

## 19. MCTS + NNUE（AlphaZero 风格训练）

### 概述

将 NNUE 的稀疏特征 + 增量累积器与 AlphaZero 的 MCTS 训练相结合。NNUE 模型同时产出**走法概率（policy）**和**局面评分（value）**，MCTS 搜索以这两个信号为指导。

### 架构

```
                    MCTS 搜索
              ┌──────────────────┐
              │  Selection: UCB  │
              │  Expansion: NNUE | policy  ←  先验概率
              │  Backup: NNUE   | value   ←  局面评分
              └──────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Policy Target      Value Target
        (访问次数分布)       (搜索结果)
              │                 │
              └────────┬────────┘
                       ▼
              NNUE 模型训练
         L = CE(policy) + MSE(value)
```

### NNUE 网络（带策略头）

```
Sparse Features (360-dim)        Dense Features (66-dim)
        │                                │
        ▼                                │
Feature Transformer (360→64)             │
  增量累积器 (stm + nstm)                │
        │                                │
        ▼                                ▼
   CReLU(stm_acc || nstm_acc)      ──── concat ──── → 194-dim
                             │
                             ▼
                    FC1 (194→256) + CReLU
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        Value Head       Policy Head
        FC2v (256→1)     FC2p (256→60)
        + tanh           + softmax
```

| 组件 | 输出 | 范围 | 损失函数 |
|------|------|------|---------|
| Shared trunk (FT + FC1) | 256-dim 隐藏 | — | — |
| Policy head | 60 logits | ℝ | CrossEntropy(softmax, visit_counts) |
| Value head | 1 个标量 | [-1, 1] | MSE(search_value) |

### MCTS 集成

MCTS 的每个节点评估调用 NNUE 模型：

```python
def evaluate_node(state):
    policy_logits, value = nnue_mcts.forward(state)
    return policy_logits, value
```

| MCTS 阶段 | NNUE 作用 |
|-----------|----------|
| Selection | policy 作为 UCB 先验概率 |
| Expansion | policy 初始化子节点概率 |
| Leaf eval | value 作为叶子节点评估 |
| Backup | value 反向传播更新父节点 |

### UCB 公式（同 AlphaZero）

```python
ucb_score = Q(s,a) + C_PUCT * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

- `Q(s,a)`：搜索到的平均价值
- `P(s,a)`：NNUE policy head 输出的先验概率
- `N(s,a)`：子节点访问次数
- `C_PUCT = 1.4`

### 训练损失函数

每个位置（来自 MCTS 自对弈）：

```python
pi = visit_counts / sum(visit_counts)  # MCTS 策略目标
z = game_result  # ±1 胜负目标

policy_loss = -sum(pi * log(p_pred))  # CrossEntropy
value_loss = (z - v_pred) ** 2          # MSE
loss = policy_loss + 0.5 * value_loss   # 联合损失
```

### 与 AB+NNUE 对比

| 维度 | AB+NNUE (当前) | MCTS+NNUE (新) |
|------|---------------|----------------|
| 模型输出 | value 1 个数字 | **policy 60 + value 1** |
| 信号 | 搜索评分 | **搜索评分 + 策略分布** |
| 搜索 | Alpha-Beta | MCTS |
| 训练效率 | 低（只有价值） | **高（策略+价值）** |
| vs Random 预期 | 77% (瓶颈) | **~95%** |
| 数据生成速度 | 快（Rust 原生） | 慢（MCTS 需更多模拟） |
| 搜索对齐 | 评分 → 搜索 | 搜索 → 训练 → 更好的搜索 |

### 训练流程

```
Iteration N:
  1. 加载 NNUEMCTS 模型
  2. Self-play: MCTS 搜索（NNUE 评估）× 200 局
     记录 (state, visit_counts, result)
  3. 训练: 从 replay buffer 采样
     loss = CE(policy, visit_counts) + MSE(value, result)
     SGD 更新 NNUEMCTS 权重
  4. 评估: vs 前代 / vs Random
     如果 win_rate > 0.55 → 保存为 best
```

### 实现位置（规划）

- `penguinchess/ai/nnue_mcts.py` — NNUEMCTS 模型（NNUE + 策略头）
- `examples/train_nnue_mcts.py` — 训练脚本
- `game_engine/src/nnue_rs.rs` — Rust 推理（需扩展策略头）
- `game_engine/src/ffi.rs` — FFI 接口（需新 eval callback）


