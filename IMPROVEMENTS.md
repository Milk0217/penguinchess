# 企鹅棋 — 改进建议文档

## 一、问题修复（Bug Fix）

### ✅ 1. gameovercheck() 棋子归属判断错误 — 已修复

- **修复**: `piece.id % 2` → `Math.floor(piece.id / 2) % 2`
- **文件**: `statics/main.js`

### ✅ 2. gameovercheck() 死代码 — 已修复

- **修复**: 移除 `throw error` 后的 `return true`
- **文件**: `statics/main.js`

### ✅ 3. replayHistory() 全局状态未同步 — 已修复

- **修复**: 参数重命名 `history` → `historyRecord`，避免遮蔽全局变量
- **文件**: `statics/main.js`

### ✅ 4. aftergame() 空函数 — 已修复

- **修复**: 实现完整结算逻辑，显示胜者公告和得分差异
- **文件**: `statics/main.js`

---

## 二、代码质量改进

### 1. 未使用的 dead code

- `board.js` 第 69-70 行 `this.left` 和 `this.top` 在赋值后从未被读取
- `piece.js` 第 52-56 行 `cubeToPixel()` 方法未被调用

### 2. 魔法数字分散

多个关键数值散落在代码各处，难以调整：

| 数值 | 位置 | 含义 |
|------|------|------|
| `99` | main.js:10 | 总分 |
| `60` | main.js:607 | 棋盘格子数 |
| `4,6,8` | main.js:51-53 | 玩家1棋子ID |
| `5,7,9` | main.js:51-53 | 玩家2棋子ID |
| `8, 10` | main.js:610 | 3的个数范围 |
| `3` | main.js:613-615 | 数字3的分数 |
| `1000` | main.js:591 | 回放延迟(ms) |
| `40px` | piece.js:32-33 | 棋子直径 |
| `25px` | board.js:66-67 | hex偏移 |
| `-4, 3` | board.js:174 | q轴范围 |

**建议**: 集中到一个 `config.js` 或文件顶部的常量对象中：
```javascript
const CONFIG = {
  BOARD_RADIUS: 8,
  TOTAL_VALUE: 99,
  HEX_COUNT: 60,
  REPLAY_DELAY_MS: 1000,
  PIECE_DIAMETER: 40,
  PLAYER_1_PIECES: [4, 6, 8],
  PLAYER_2_PIECES: [5, 7, 9],
  COUNT_OF_THREE_MIN: 8,
  COUNT_OF_THREE_MAX: 10,
  THREE_VALUE: 3,
};
```

### 3. 错误处理缺失

- `createBoard()` 在 `board` 元素不存在时仅 `return;` 无提示
- 回放导入的 JSON 无 Schema 验证，格式错误会导致静默失败
- `generateSequence()` 递归深度无保护，极小概率可能爆栈

### 4. 命名不一致

| 当前名称 | 问题 |
|----------|------|
| `playerSymbol` | 实际是 player index（0 或 1），不是 symbol |
| `showCoords` | 局部变量在 board.js，toggle coords 功能分散在两个文件 |
| `hexes` / `pieces` / `players` | 全局变量无统一前缀，容易与局部变量混淆 |

### 5. JSDoc 不完整

仅 `placePieces` 和 `turn` 有 JSDoc，其余函数均缺失，不利于维护。

---

## 三、功能增强

### 1. AI 对战（高优先级）

`player.js` 已预留 `isComputer` 属性但从未使用。可实现：

- 简单 AI：随机选择合法移动
- 策略 AI：优先选择高价值格、阻断对手连接

### 2. 局域网对战（中优先级）

当前仅支持本地双人。可以：

- 用 Flask-SocketIO 添加 WebSocket 支持
- 其中一方作为 host，另一方通过 IP:Port 连接
- 实现实时同步

### 3. 历史记录改进

- 存储每步操作的分数变化
- 支持评论/标注特定回合
- 导出格式改为包含 meta 信息（时长、玩家、日期）的结构

### 4. 响应式布局（高优先级）

当前棋盘固定 600x600，完全不支持移动端：

- 使用 CSS Grid 或 flexbox 替代绝对定位
- 根据视口缩放 hex 尺寸
- 控制面板改为垂直堆叠

### 5. 音效与动画

- 放置/移动棋子时的音效
- 棋子移动的平滑过渡动画（CSS transition）
- hex 消除时的淡出效果

### 6. 统计面板

- 本局步数历史
- 双方得分曲线图
- 高价值格子的占领统计

### 7. 多人扩展

支持 3-4 人对战（每人 2-3 个棋子），需要重新设计胜利条件和回合顺序。

---

## 四、工程化改进

### 1. 添加 ESLint / Prettier

前端无任何 linting/formatting，JS 代码风格不统一。

### 2. 添加 Playwright / Cypress E2E 测试

当前零测试覆盖。项目虽小但逻辑复杂（放置、移动、消除、回放），手动测试成本高。

### 3. 分离 HTML 模板

`index.html` 中手写大量按钮结构，可改为从 JS 动态生成，便于扩展控制按钮。

### 4. 构建工具

当前纯 ES module 无构建，但未来如需引入 npm 包（图标库、动画库）会需要打包。建议评估后引入 Vite 或 esbuild。

### 5. GitHub Actions CI

添加自动化：lint → test → 构建检查。

---

## 五、安全改进

### 1. 导入文件无验证

`importFile` 直接 `JSON.parse()`，恶意 JSON 可导致应用崩溃或无限循环（特别是 `replayHistory` 中的递归）。

**修复**: 添加 JSON Schema 验证或递归深度限制。

### 2. CORS 配置

Flask 无任何 CORS 配置，如开启局域网对战会有跨域问题。

---

## 六、优先级排序建议

| 优先级 | 改进项 |
|--------|--------|
| **P0（必做）** | Bug 修复（gameovercheck, aftergame） |
| **P1（高价值）** | 响应式布局、Config 集中化、ESLint |
| **P2（中价值）** | AI 对战、回放功能增强、统计面板 |
| **P3（长期）** | 局域网对战、多人扩展、构建工具化 |
