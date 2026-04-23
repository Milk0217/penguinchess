# 棋盘可视化系统重构计划

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 构建灵活、可扩展的棋盘可视化系统，支持多种棋盘布局和自定义主题

**Architecture:** 
- 采用策略模式，将棋盘布局（Board Layout）与渲染逻辑分离
- 支持多种预设布局（平行四边形、六角形、自定义）
- 主题系统支持颜色、尺寸、间距等配置
- 预留接口支持未来自定义棋盘生成

**Tech Stack:** React + TypeScript, CSS Variables, 配置驱动

---

## 当前问题分析

1. **棋盘形状不灵活** - 硬编码的平行四边形布局
2. **配色方案单调** - 只按分值着色，缺乏层次感
3. **坐标系统不直观** - 立方体坐标对新手不友好
4. **缺乏主题切换** - 无法适应不同场景需求
5. **扩展性差** - 添加新布局需要修改核心代码

---

## 设计方案

### 1. 棋盘布局系统

```typescript
// 布局配置接口
interface BoardLayout {
  id: string;
  name: string;
  description: string;
  // 生成六边形坐标
  generateHexes(): HexCoord[];
  // 坐标转像素
  cubeToPixel(q: number, r: number, config: LayoutConfig): { x: number; y: number };
  // 获取布局边界
  getBounds(hexes: HexCoord[]): Bounds;
}

// 预设布局
const LAYOUTS: Record<string, BoardLayout> = {
  parallelogram: { /* 当前平行四边形 */ },
  hexagon: { /* 正六角形 */ },
  triangle: { /* 三角形 */ },
  custom: { /* 用户自定义 */ },
};
```

### 2. 主题系统

```typescript
interface BoardTheme {
  id: string;
  name: string;
  colors: {
    hexValue1: string;      // 1分格子
    hexValue2: string;      // 2分格子
    hexValue3: string;      // 3分格子
    hexOccupied: string;    // 被占据
    hexEliminated: string;  // 已消除
    hexTarget: string;      // 可移动目标
    pieceP1: string;        // P1棋子
    pieceP2: string;        // P2棋子
    background: string;     // 棋盘背景
    text: string;           // 文字颜色
  };
  sizes: {
    hexSize: number;        // 六边形尺寸
    pieceSize: number;      // 棋子尺寸
    padding: number;        // 内边距
    gap: number;            // 间距
  };
  effects: {
    showCoords: boolean;    // 显示坐标
    showValues: boolean;    // 显示分值
    showBorders: boolean;   // 显示边框
    animation: boolean;     // 启用动画
  };
}
```

### 3. 组件架构

```
Board/
├── BoardContainer.tsx      # 主容器，管理布局和主题
├── BoardRenderer.tsx       # 渲染引擎
├── HexCell.tsx            # 单个六边形组件
├── Piece.tsx              # 棋子组件
├── Legend.tsx              # 图例组件
├── CoordinateOverlay.tsx  # 坐标覆盖层
├── layouts/
│   ├── parallelogram.ts   # 平行四边形布局
│   ├── hexagon.ts         # 正六角形布局
│   └── index.ts           # 布局注册
├── themes/
│   ├── default.ts         # 默认主题
│   ├── dark.ts            # 暗色主题
│   ├── classic.ts         # 经典主题
│   └── index.ts           # 主题注册
└── types.ts               # 类型定义
```

---

## 实施任务

### Task 1: 定义核心类型接口

**Objective:** 建立棋盘系统的类型基础

**Files:**
- Create: `frontend/src/board/types.ts`

**Step 1: 创建类型定义文件**

```typescript
// frontend/src/board/types.ts

/** 六边形立方体坐标 */
export interface HexCoord {
  q: number;
  r: number;
  s: number;
}

/** 像素坐标 */
export interface PixelCoord {
  x: number;
  y: number;
}

/** 边界框 */
export interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  width: number;
  height: number;
}

/** 布局配置 */
export interface LayoutConfig {
  hexSize: number;
  padding: number;
}

/** 棋盘布局接口 */
export interface BoardLayout {
  id: string;
  name: string;
  description: string;
  /** 生成所有六边形坐标 */
  generateHexes(): HexCoord[];
  /** 立方体坐标转像素坐标 */
  cubeToPixel(q: number, r: number, config: LayoutConfig): PixelCoord;
  /** 获取布局边界 */
  getBounds(hexes: HexCoord[], config: LayoutConfig): Bounds;
}

/** 主题颜色配置 */
export interface ThemeColors {
  hexValue1: string;
  hexValue2: string;
  hexValue3: string;
  hexOccupied: string;
  hexEliminated: string;
  hexTarget: string;
  pieceP1: string;
  pieceP2: string;
  background: string;
  text: string;
  textOnDark: string;
}

/** 主题尺寸配置 */
export interface ThemeSizes {
  hexSize: number;
  pieceSize: number;
  padding: number;
  borderWidth: number;
}

/** 主题效果配置 */
export interface ThemeEffects {
  showCoords: boolean;
  showValues: boolean;
  showBorders: boolean;
  animation: boolean;
  shadow: boolean;
}

/** 完整主题配置 */
export interface BoardTheme {
  id: string;
  name: string;
  description: string;
  colors: ThemeColors;
  sizes: ThemeSizes;
  effects: ThemeEffects;
}

/** 棋盘状态（从后端获取） */
export interface HexData {
  index: number;
  q: number;
  r: number;
  s: number;
  value: number;
}

/** 棋子状态 */
export interface PieceData {
  id: number;
  owner: number;
  q: number | null;
  r: number | null;
  s: number | null;
  alive: boolean;
}

/** 游戏状态 */
export interface GameState {
  session_id: string;
  hexes: HexData[];
  pieces: PieceData[];
  current_player: number;
  phase: "placement" | "movement";
  scores: [number, number];
  legal_actions: number[];
  episode_steps: number;
  game_over: boolean;
  winner: number | null;
}
```

**Step 2: 验证类型文件**

```bash
cd frontend && npx tsc --noEmit src/board/types.ts
```

Expected: 无错误输出

**Step 3: Commit**

```bash
git add frontend/src/board/types.ts
git commit -m "feat(board): add core type definitions for board system"
```

---

### Task 2: 实现平行四边形布局

**Objective:** 将当前硬编码的布局逻辑提取为独立模块

**Files:**
- Create: `frontend/src/board/layouts/parallelogram.ts`
- Create: `frontend/src/board/layouts/index.ts`

**Step 1: 创建平行四边形布局**

```typescript
// frontend/src/board/layouts/parallelogram.ts

import type { BoardLayout, HexCoord, PixelCoord, Bounds, LayoutConfig } from "../types";

/** 平行四边形布局 - 当前使用的布局 */
export const parallelogramLayout: BoardLayout = {
  id: "parallelogram",
  name: "平行四边形",
  description: "标准平行四边形棋盘，60个六边形",

  generateHexes(): HexCoord[] {
    const hexes: HexCoord[] = [];
    
    // q 值范围：-4 到 3
    for (let q = -4; q <= 3; q++) {
      // 根据 q 的奇偶性确定 r 的范围
      const isEven = q % 2 === 0;
      const rStart = isEven ? -4 : -3;
      const rEnd = 3;
      
      for (let r = rStart; r <= rEnd; r++) {
        const s = -q - r;
        // 验证立方体坐标约束
        if (q + r + s === 0) {
          hexes.push({ q, r, s });
        }
      }
    }
    
    return hexes;
  },

  cubeToPixel(q: number, r: number, config: LayoutConfig): PixelCoord {
    const { hexSize } = config;
    const x = hexSize * (3 / 2) * q;
    const y = hexSize * (Math.sqrt(3) / 2 * q + Math.sqrt(3) * r);
    return { x, y };
  },

  getBounds(hexes: HexCoord[], config: LayoutConfig): Bounds {
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (const hex of hexes) {
      const { x, y } = this.cubeToPixel(hex.q, hex.r, config);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }

    return {
      minX,
      maxX,
      minY,
      maxY,
      width: maxX - minX,
      height: maxY - minY,
    };
  },
};
```

**Step 2: 创建布局注册表**

```typescript
// frontend/src/board/layouts/index.ts

import type { BoardLayout } from "../types";
import { parallelogramLayout } from "./parallelogram";

/** 所有可用布局 */
export const LAYOUTS: Record<string, BoardLayout> = {
  parallelogram: parallelogramLayout,
};

/** 获取布局 */
export function getLayout(id: string): BoardLayout {
  const layout = LAYOUTS[id];
  if (!layout) {
    throw new Error(`Unknown layout: ${id}`);
  }
  return layout;
}

/** 获取所有布局列表 */
export function getAllLayouts(): BoardLayout[] {
  return Object.values(LAYOUTS);
}
```

**Step 3: 验证布局生成**

```bash
cd frontend && npx ts-node -e "
import { parallelogramLayout } from './src/board/layouts/parallelogram';
const hexes = parallelogramLayout.generateHexes();
console.log('Hex count:', hexes.length);
console.log('First hex:', hexes[0]);
"
```

Expected: 
```
Hex count: 60
First hex: { q: -4, r: -4, s: 8 }
```

**Step 4: Commit**

```bash
git add frontend/src/board/layouts/
git commit -m "feat(board): extract parallelogram layout as independent module"
```

---

### Task 3: 实现主题系统

**Objective:** 创建可配置的主题系统

**Files:**
- Create: `frontend/src/board/themes/default.ts`
- Create: `frontend/src/board/themes/dark.ts`
- Create: `frontend/src/board/themes/index.ts`

**Step 1: 创建默认主题**

```typescript
// frontend/src/board/themes/default.ts

import type { BoardTheme } from "../types";

/** 默认主题 - 清晰现代风格 */
export const defaultTheme: BoardTheme = {
  id: "default",
  name: "默认",
  description: "清晰现代的默认主题",
  colors: {
    hexValue1: "#93c5fd",      // 浅蓝
    hexValue2: "#3b82f6",      // 中蓝
    hexValue3: "#f59e0b",      // 金色（高价值）
    hexOccupied: "#64748b",    // 深灰
    hexEliminated: "#cbd5e1",  // 浅灰
    hexTarget: "#fb923c",      // 橙色（可移动）
    pieceP1: "#3b82f6",        // 蓝色
    pieceP2: "#ef4444",        // 红色
    background: "#f8fafc",     // 浅灰背景
    text: "#1e293b",           // 深色文字
    textOnDark: "#ffffff",     // 浅色文字
  },
  sizes: {
    hexSize: 28,
    pieceSize: 44,
    padding: 40,
    borderWidth: 2,
  },
  effects: {
    showCoords: false,
    showValues: true,
    showBorders: true,
    animation: true,
    shadow: true,
  },
};
```

**Step 2: 创建暗色主题**

```typescript
// frontend/src/board/themes/dark.ts

import type { BoardTheme } from "../types";

/** 暗色主题 */
export const darkTheme: BoardTheme = {
  id: "dark",
  name: "暗色",
  description: "护眼暗色主题",
  colors: {
    hexValue1: "#1e40af",      // 深蓝
    hexValue2: "#1d4ed8",      // 更深蓝
    hexValue3: "#d97706",      // 暗金
    hexOccupied: "#374151",    // 深灰
    hexEliminated: "#1f2937",  // 更深灰
    hexTarget: "#ea580c",      // 暗橙
    pieceP1: "#60a5fa",        // 亮蓝
    pieceP2: "#f87171",        // 亮红
    background: "#0f172a",     // 深色背景
    text: "#f1f5f9",           // 浅色文字
    textOnDark: "#f1f5f9",     // 浅色文字
  },
  sizes: {
    hexSize: 28,
    pieceSize: 44,
    padding: 40,
    borderWidth: 1,
  },
  effects: {
    showCoords: false,
    showValues: true,
    showBorders: false,
    animation: true,
    shadow: true,
  },
};
```

**Step 3: 创建主题注册表**

```typescript
// frontend/src/board/themes/index.ts

import type { BoardTheme } from "../types";
import { defaultTheme } from "./default";
import { darkTheme } from "./dark";

/** 所有可用主题 */
export const THEMES: Record<string, BoardTheme> = {
  default: defaultTheme,
  dark: darkTheme,
};

/** 获取主题 */
export function getTheme(id: string): BoardTheme {
  const theme = THEMES[id];
  if (!theme) {
    throw new Error(`Unknown theme: ${id}`);
  }
  return theme;
}

/** 获取所有主题列表 */
export function getAllThemes(): BoardTheme[] {
  return Object.values(THEMES);
}
```

**Step 4: Commit**

```bash
git add frontend/src/board/themes/
git commit -m "feat(board): add theme system with default and dark themes"
```

---

### Task 4: 创建六边形单元组件

**Objective:** 实现可复用的六边形单元组件

**Files:**
- Create: `frontend/src/board/HexCell.tsx`

**Step 1: 创建 HexCell 组件**

```tsx
// frontend/src/board/HexCell.tsx

import type { HexData, ThemeColors, ThemeSizes, ThemeEffects } from "./types";

interface HexCellProps {
  hex: HexData;
  x: number;
  y: number;
  colors: ThemeColors;
  sizes: ThemeSizes;
  effects: ThemeEffects;
  isTarget: boolean;
  onClick?: (hex: HexData) => void;
}

export function HexCell({ 
  hex, 
  x, 
  y, 
  colors, 
  sizes, 
  effects, 
  isTarget, 
  onClick 
}: HexCellProps) {
  const { value, q, r } = hex;
  const isEliminated = value < 0;
  const isOccupied = value === 0;

  // 根据状态确定背景色
  let bg = colors.hexValue1;
  if (isTarget) bg = colors.hexTarget;
  else if (isEliminated) bg = colors.hexEliminated;
  else if (isOccupied) bg = colors.hexOccupied;
  else if (value === 2) bg = colors.hexValue2;
  else if (value === 3) bg = colors.hexValue3;

  // 文字颜色：深色背景用浅色文字
  const textColor = (value === 2 || isOccupied) ? colors.textOnDark : colors.text;

  const hexSize = sizes.hexSize * 2;

  return (
    <button
      style={{
        position: "absolute",
        left: x,
        top: y,
        width: hexSize,
        height: hexSize,
        background: bg,
        transform: "translate(-50%, -50%)",
        clipPath: "polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%)",
        border: "none",
        cursor: isEliminated ? "default" : "pointer",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "2px",
        transition: effects.animation ? "all 0.2s ease" : "none",
        zIndex: isTarget ? 15 : 5,
      }}
      onClick={() => !isEliminated && onClick?.(hex)}
      title={`(${q}, ${r}, ${hex.s}) = ${value}`}
    >
      {effects.showValues && value > 0 && (
        <span style={{ 
          fontSize: 14, 
          fontWeight: 800, 
          color: textColor, 
          pointerEvents: "none",
          lineHeight: 1 
        }}>
          {value}
        </span>
      )}
      {effects.showCoords && (
        <span style={{ 
          fontSize: 8, 
          fontWeight: 500, 
          color: textColor, 
          opacity: 0.8, 
          pointerEvents: "none",
          lineHeight: 1 
        }}>
          {q},{r}
        </span>
      )}
    </button>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/board/HexCell.tsx
git commit -m "feat(board): create reusable HexCell component"
```

---

### Task 5: 创建棋子组件

**Objective:** 实现可复用的棋子组件

**Files:**
- Create: `frontend/src/board/Piece.tsx`

**Step 1: 创建 Piece 组件**

```tsx
// frontend/src/board/Piece.tsx

import type { PieceData, ThemeColors, ThemeSizes } from "./types";

interface PieceProps {
  piece: PieceData;
  x: number;
  y: number;
  colors: ThemeColors;
  sizes: ThemeSizes;
  isSelected: boolean;
  isCurrentPlayer: boolean;
}

export function Piece({ 
  piece, 
  x, 
  y, 
  colors, 
  sizes, 
  isSelected, 
  isCurrentPlayer 
}: PieceProps) {
  const { id, owner } = piece;
  const isP1 = owner === 0;

  const bgColor = isP1 ? colors.pieceP1 : colors.pieceP2;
  const borderColor = isSelected 
    ? (isP1 ? "#1d4ed8" : "#b91c1c")
    : "rgba(255,255,255,0.3)";
  const ringColor = isSelected 
    ? (isP1 ? "#93c5fd" : "#fca5a5")
    : "transparent";

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        width: sizes.pieceSize,
        height: sizes.pieceSize,
        borderRadius: "50%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        transform: "translate(-50%, -50%)",
        border: `${sizes.borderWidth}px solid ${borderColor}`,
        boxShadow: isSelected
          ? `0 0 0 3px ${ringColor}, 0 4px 12px rgba(0,0,0,0.4)`
          : "0 2px 8px rgba(0,0,0,0.3)",
        zIndex: isCurrentPlayer ? 35 : 25,
        fontSize: 12,
        fontWeight: 800,
        color: "white",
        background: bgColor,
        pointerEvents: "none",
        userSelect: "none",
      }}
      title={`棋子 ${id} (${isP1 ? "P1" : "P2"})`}
    >
      {id}
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/board/Piece.tsx
git commit -m "feat(board): create reusable Piece component"
```

---

### Task 6: 创建图例组件

**Objective:** 实现可配置的图例组件

**Files:**
- Create: `frontend/src/board/Legend.tsx`

**Step 1: 创建 Legend 组件**

```tsx
// frontend/src/board/Legend.tsx

import type { ThemeColors } from "./types";

interface LegendProps {
  colors: ThemeColors;
  showCoords: boolean;
  onToggleCoords: () => void;
}

export function Legend({ colors, showCoords, onToggleCoords }: LegendProps) {
  const items = [
    { color: colors.hexValue1, label: "1分", border: colors.hexValue1 },
    { color: colors.hexValue2, label: "2分", border: colors.hexValue2 },
    { color: colors.hexValue3, label: "3分", border: colors.hexValue3 },
    { color: colors.hexOccupied, label: "被占据", border: colors.hexOccupied },
    { color: colors.hexEliminated, label: "已消除", border: colors.hexEliminated },
    { color: colors.hexTarget, label: "可移动", border: colors.hexTarget },
    { color: colors.pieceP1, label: "P1棋子", border: colors.pieceP1, round: true },
    { color: colors.pieceP2, label: "P2棋子", border: colors.pieceP2, round: true },
  ];

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: "8px",
      marginTop: "12px",
    }}>
      {/* 图例项 */}
      <div style={{
        display: "flex",
        gap: "16px",
        padding: "8px 16px",
        background: colors.background,
        borderRadius: "8px",
        border: `1px solid ${colors.hexEliminated}`,
        fontSize: "12px",
        color: colors.text,
        flexWrap: "wrap",
        justifyContent: "center",
      }}>
        {items.map((item) => (
          <div key={item.label} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <div style={{
              width: 16,
              height: 16,
              background: item.color,
              borderRadius: item.round ? "50%" : "3px",
              border: `1px solid ${item.border}`,
            }} />
            <span>{item.label}</span>
          </div>
        ))}
      </div>

      {/* 坐标切换按钮 */}
      <button
        onClick={onToggleCoords}
        style={{
          padding: "4px 12px",
          fontSize: "12px",
          background: showCoords ? colors.pieceP1 : colors.hexEliminated,
          color: showCoords ? "white" : colors.text,
          border: "none",
          borderRadius: "4px",
          cursor: "pointer",
        }}
      >
        {showCoords ? "隐藏坐标" : "显示坐标"}
      </button>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/board/Legend.tsx
git commit -m "feat(board): create configurable Legend component"
```

---

### Task 7: 创建主棋盘容器组件

**Objective:** 整合所有子组件，实现完整的棋盘渲染

**Files:**
- Create: `frontend/src/board/BoardContainer.tsx`

**Step 1: 创建 BoardContainer 组件**

```tsx
// frontend/src/board/BoardContainer.tsx

import { useState, useMemo } from "react";
import type { GameState, BoardLayout, BoardTheme } from "./types";
import { HexCell } from "./HexCell";
import { Piece } from "./Piece";
import { Legend } from "./Legend";

interface BoardContainerProps {
  state: GameState;
  layout: BoardLayout;
  theme: BoardTheme;
  selectedPieceId: number | null;
  targetIndices: Set<number>;
  onHexClick: (hex: GameState["hexes"][0]) => void;
}

export function BoardContainer({
  state,
  layout,
  theme,
  selectedPieceId,
  targetIndices,
  onHexClick,
}: BoardContainerProps) {
  const { hexes, pieces, current_player } = state;
  const { colors, sizes, effects } = theme;
  const [showCoords, setShowCoords] = useState(effects.showCoords);

  // 生成六边形坐标
  const layoutHexes = useMemo(() => layout.generateHexes(), [layout]);

  // 计算边界
  const bounds = useMemo(() => 
    layout.getBounds(layoutHexes, { hexSize: sizes.hexSize, padding: sizes.padding }),
    [layout, layoutHexes, sizes.hexSize, sizes.padding]
  );

  // 计算偏移量（居中）
  const offsetX = -bounds.minX + sizes.padding;
  const offsetY = -bounds.minY + sizes.padding;
  const containerWidth = Math.ceil(bounds.width + sizes.hexSize * 2 + sizes.padding * 2);
  const containerHeight = Math.ceil(bounds.height + sizes.hexSize * 2 + sizes.padding * 2);

  // 创建 hex index 到坐标的映射
  const hexCoordMap = useMemo(() => {
    const map = new Map<number, { x: number; y: number }>();
    hexes.forEach((hex, index) => {
      const { x, y } = layout.cubeToPixel(hex.q, hex.r, { hexSize: sizes.hexSize, padding: sizes.padding });
      map.set(index, { x: x + offsetX, y: y + offsetY });
    });
    return map;
  }, [hexes, layout, sizes.hexSize, offsetX, offsetY]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      {/* 棋盘 */}
      <div
        style={{
          position: "relative",
          width: Math.min(containerWidth, 720),
          height: Math.min(containerHeight, 720),
          background: colors.background,
          borderRadius: "12px",
          overflow: "hidden",
        }}
      >
        {/* 六边形层 */}
        {hexes.map((hex) => {
          const coord = hexCoordMap.get(hex.index);
          if (!coord) return null;

          return (
            <HexCell
              key={hex.index}
              hex={hex}
              x={coord.x}
              y={coord.y}
              colors={colors}
              sizes={sizes}
              effects={{ ...effects, showCoords: showCoords }}
              isTarget={targetIndices.has(hex.index)}
              onClick={onHexClick}
            />
          );
        })}

        {/* 棋子层 */}
        {pieces.map((piece) => {
          if (!piece.alive || piece.q === null || piece.r === null) return null;

          const { x, y } = layout.cubeToPixel(piece.q, piece.r, { hexSize: sizes.hexSize, padding: sizes.padding });
          const px = x + offsetX;
          const py = y + offsetY;

          return (
            <Piece
              key={piece.id}
              piece={piece}
              x={px}
              y={py}
              colors={colors}
              sizes={sizes}
              isSelected={selectedPieceId === piece.id}
              isCurrentPlayer={piece.owner === current_player}
            />
          );
        })}
      </div>

      {/* 图例 */}
      <Legend
        colors={colors}
        showCoords={showCoords}
        onToggleCoords={() => setShowCoords(!showCoords)}
      />
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/board/BoardContainer.tsx
git commit -m "feat(board): create main BoardContainer component"
```

---

### Task 8: 更新 App.tsx 使用新系统

**Objective:** 将现有 App 迁移到新的棋盘系统

**Files:**
- Modify: `frontend/src/App.tsx`

**Step 1: 更新 App.tsx 导入**

```typescript
// frontend/src/App.tsx (顶部添加)

import { BoardContainer } from "./board/BoardContainer";
import { getLayout } from "./board/layouts";
import { getTheme } from "./board/themes";
```

**Step 2: 更新 Board 组件调用**

```tsx
// 替换现有的 <Board /> 为 <BoardContainer />

{/* 棋盘 */}
<BoardContainer
  state={state}
  layout={getLayout("parallelogram")}
  theme={getTheme("default")}
  selectedPieceId={selectedPieceId}
  targetIndices={effectiveTargets}
  onHexClick={handleBoardHexClick}
/>
```

**Step 3: 删除旧的 Board 组件引用**

```typescript
// 删除这行
import Board from "./Board";
```

**Step 4: 验证构建**

```bash
cd frontend && npm run build
```

Expected: 构建成功

**Step 5: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "refactor(board): migrate App to new board system"
```

---

### Task 9: 添加主题切换功能

**Objective:** 在 UI 中添加主题切换下拉菜单

**Files:**
- Modify: `frontend/src/App.tsx`

**Step 1: 添加主题状态**

```typescript
// 在 App 组件中添加

const [currentThemeId, setCurrentThemeId] = useState("default");
const currentTheme = useMemo(() => getTheme(currentThemeId), [currentThemeId]);
```

**Step 2: 添加主题切换 UI**

```tsx
{/* 在控制按钮区域添加 */}
<select
  value={currentThemeId}
  onChange={(e) => setCurrentThemeId(e.target.value)}
  style={{
    padding: "4px 8px",
    fontSize: "12px",
    borderRadius: "4px",
    border: "1px solid #e2e8f0",
  }}
>
  <option value="default">默认主题</option>
  <option value="dark">暗色主题</option>
</select>
```

**Step 3: 更新 BoardContainer 调用**

```tsx
<BoardContainer
  state={state}
  layout={getLayout("parallelogram")}
  theme={currentTheme}  // 使用动态主题
  selectedPieceId={selectedPieceId}
  targetIndices={effectiveTargets}
  onHexClick={handleBoardHexClick}
/>
```

**Step 4: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat(board): add theme switcher UI"
```

---

### Task 10: 添加正六角形布局（示例）

**Objective:** 创建第二个布局作为扩展示例

**Files:**
- Create: `frontend/src/board/layouts/hexagon.ts`
- Modify: `frontend/src/board/layouts/index.ts`

**Step 1: 创建正六角形布局**

```typescript
// frontend/src/board/layouts/hexagon.ts

import type { BoardLayout, HexCoord, PixelCoord, Bounds, LayoutConfig } from "../types";

/** 正六角形布局 - 用于不同尺寸的棋盘 */
export const hexagonLayout: BoardLayout = {
  id: "hexagon",
  name: "正六角形",
  description: "对称六角形棋盘，可配置半径",

  // 可配置半径
  radius: 4,

  generateHexes(): HexCoord[] {
    const hexes: HexCoord[] = [];
    const radius = this.radius;

    for (let q = -radius; q <= radius; q++) {
      for (let r = -radius; r <= radius; r++) {
        const s = -q - r;
        if (Math.abs(s) <= radius) {
          hexes.push({ q, r, s });
        }
      }
    }

    return hexes;
  },

  cubeToPixel(q: number, r: number, config: LayoutConfig): PixelCoord {
    const { hexSize } = config;
    const x = hexSize * (3 / 2) * q;
    const y = hexSize * (Math.sqrt(3) / 2 * q + Math.sqrt(3) * r);
    return { x, y };
  },

  getBounds(hexes: HexCoord[], config: LayoutConfig): Bounds {
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (const hex of hexes) {
      const { x, y } = this.cubeToPixel(hex.q, hex.r, config);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }

    return {
      minX,
      maxX,
      minY,
      maxY,
      width: maxX - minX,
      height: maxY - minY,
    };
  },
};
```

**Step 2: 注册新布局**

```typescript
// frontend/src/board/layouts/index.ts

import { parallelogramLayout } from "./parallelogram";
import { hexagonLayout } from "./hexagon";

export const LAYOUTS: Record<string, BoardLayout> = {
  parallelogram: parallelogramLayout,
  hexagon: hexagonLayout,
};
```

**Step 3: Commit**

```bash
git add frontend/src/board/layouts/
git commit -m "feat(board): add hexagon layout as alternative board shape"
```

---

### Task 11: 添加布局切换功能

**Objective:** 在 UI 中添加布局切换下拉菜单

**Files:**
- Modify: `frontend/src/App.tsx`

**Step 1: 添加布局状态**

```typescript
const [currentLayoutId, setCurrentLayoutId] = useState("parallelogram");
const currentLayout = useMemo(() => getLayout(currentLayoutId), [currentLayoutId]);
```

**Step 2: 添加布局切换 UI**

```tsx
<select
  value={currentLayoutId}
  onChange={(e) => setCurrentLayoutId(e.target.value)}
  style={{
    padding: "4px 8px",
    fontSize: "12px",
    borderRadius: "4px",
    border: "1px solid #e2e8f0",
  }}
>
  <option value="parallelogram">平行四边形</option>
  <option value="hexagon">正六角形</option>
</select>
```

**Step 3: 更新 BoardContainer 调用**

```tsx
<BoardContainer
  state={state}
  layout={currentLayout}  // 使用动态布局
  theme={currentTheme}
  selectedPieceId={selectedPieceId}
  targetIndices={effectiveTargets}
  onHexClick={handleBoardHexClick}
/>
```

**Step 4: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat(board): add layout switcher UI"
```

---

### Task 12: 清理旧文件

**Objective:** 删除不再使用的旧 Board 组件

**Files:**
- Delete: `frontend/src/Board.tsx`

**Step 1: 删除旧文件**

```bash
rm frontend/src/Board.tsx
```

**Step 2: 验证构建**

```bash
cd frontend && npm run build
```

Expected: 构建成功

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor(board): remove old Board component"
```

---

## 验证清单

- [ ] 所有布局正确生成 60 个六边形
- [ ] 主题切换正常工作
- [ ] 布局切换正常工作
- [ ] 坐标显示/隐藏功能正常
- [ ] 棋子渲染正确
- [ ] 图例显示正确
- [ ] 响应式布局正常
- [ ] 无 TypeScript 错误
- [ ] 构建成功

---

## 未来扩展方向

1. **自定义布局编辑器** - 允许用户创建自己的棋盘形状
2. **更多主题** - 高对比度、色盲友好等
3. **动画效果** - 棋子移动、消除动画
4. **音效集成** - 配合主题的音效
5. **导出/导入** - 保存和分享自定义布局/主题
6. **性能优化** - 大棋盘的虚拟滚动

---

## 文件结构总览

```
frontend/src/board/
├── types.ts              # 类型定义
├── BoardContainer.tsx    # 主容器组件
├── HexCell.tsx           # 六边形单元
├── Piece.tsx             # 棋子组件
├── Legend.tsx            # 图例组件
├── layouts/
│   ├── index.ts          # 布局注册表
│   ├── parallelogram.ts  # 平行四边形
│   └── hexagon.ts        # 正六角形
└── themes/
    ├── index.ts          # 主题注册表
    ├── default.ts        # 默认主题
    └── dark.ts           # 暗色主题
```

---

## 实施建议

1. **按顺序实施** - 每个 Task 依赖前一个
2. **频繁测试** - 每完成一个 Task 就测试
3. **保持兼容** - 确保旧功能不受影响
4. **文档更新** - 在 AGENTS.md 中记录新架构
5. **代码审查** - 每个 PR 都进行审查

这个计划提供了灵活、可扩展的棋盘系统，支持未来的自定义需求。需要我开始实施吗？
