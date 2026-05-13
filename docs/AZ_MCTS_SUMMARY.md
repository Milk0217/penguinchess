# AZ MCTS 训练总结

## 背景

本项目使用 AlphaZero 方法（MCTS + 神经网络）训练企鹅棋 AI。经过 6 轮系统性的优化尝试，确认了 AZ 自对弈在单 GPU 规模上的局限性。

## 训练管线

```
随机权重 → MCTS 自对弈 (800 sims) → (obs, policy_target, value_target) → 训练 30 epochs → 更新权重 → 循环
```

## 六轮训练全览

| 轮次 | 配置 | 最佳 vs Random | 问题 |
|------|------|:-------------:|------|
| 1 (206-dim) | 200 sims, Python ThreadPool | **70%** ✨ | UCB bug (INFINITY+PUCT=INFINITY) 导致伪成功 |
| 2 (272-dim) | UCB 修复, 200 sims | **58%** | 首次真实基线 |
| 3 (800 sims) | 更多模拟 | 46% | 更深搜索但不更好的数据 |
| 4 (随机开局10步) | 增加数据多样性 | 42% | 开局噪声淹没信号 |
| 5 (gen_2 蒸馏) | 监督学习 one-hot | 45% | one-hot 标签信息不足 |
| 6 (全部优化) | Rust batch, FIFO, t=1.0 | **61%** | 确认的天花板 |

## 核心模式

所有训练均呈现一致的行为：

```
iter 1-5:  快速上升到 55-61%
iter 10-25: 稳定在 50-58%
iter 25+:   缓慢下降到 40-48%
```

Loss 在 iter 5 达到最低（~0.15），然后逐步上升。Replay buffer 中的自对弈数据质量随时间下降——模型学到自己的噪声后，产生的训练数据污染了 buffer（自反饙问题）。

## 确认的 Bug

1. **UCB 公式**（mcts_rs.rs）：`INFINITY + 1e-12 = INFINITY`，所有未访问节点 UCB 相同，batch 模拟互相抵消。
2. **`terminal_value` 符号翻转**：`step()` 终止时不切换 `current_player`，backup 的交替取反与终端值视角不一致。
3. **Rust vs Python 观测编码**：dim 271（`episode_steps/500`）在 Rust 侧固定为 0。

## 已实现并测试的优化

| 优化 | 文件 | 效果 |
|------|------|------|
| Rust 批量自对弈 (ffi_az_selfplay_batch) | ffi.rs, mcts_rs.rs | 消除 Python 胶水开销 |
| Rust 观测编码 (ffi_encode_obs) | ffi.rs | 跳过 JSON 序列化 |
| FIFO replay buffer (500K) | train.py | 防止无限增长 |
| AlphaZero 温度策略 (t=1.0) | train.py | 策略分布更丰富 |
| 自适应值权重 | train.py | V loss 低时聚焦策略 |
| 树复用 vs Random 评估 | train.py | 1.7h → 15min |
| 树复用 ELO 评估 | train.py | 50min → 8min |

## 结论

**AZ 自对弈在单 GPU 上的上限约 60% vs Random。** 根本原因是自反饙循环：弱权重→弱 MCTS→弱数据→弱权重。突破此上限需要：

1. **分布式训练**（数百个自对弈 worker）
2. **更强的初始权重**（如蒸馏或离线数据）
3. **切换到 AB+NNUE**（已知达到 80% vs Random，持续改进中）

## 已保存模型

`models/alphazero/alphazero_resnet_2m_best.pth` — 最佳模型 (ELO 1231, 61% vs Random)
`models/alphazero/alphazero_resnet_2m_final.pth` — 50 代训练后最终模型
