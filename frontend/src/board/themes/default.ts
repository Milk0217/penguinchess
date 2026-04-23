import type { BoardTheme } from "../types";

/** 默认主题 — 清晰现代的默认配色 */
export const defaultTheme: BoardTheme = {
  id: "default",
  name: "默认",
  description: "清晰现代的默认主题",
  colors: {
    hexValue1: "#93c5fd",      // 浅蓝 - 1分格子
    hexValue2: "#3b82f6",      // 中蓝 - 2分格子
    hexValue3: "#f59e0b",      // 金色 - 3分格子（高价值）
    hexOccupied: "#64748b",    // 深灰 - 被占据
    hexEliminated: "#cbd5e1",  // 浅灰 - 已消除
    hexTarget: "#fb923c",      // 橙色 - 可移动目标
    pieceP1: "#3b82f6",        // 蓝色 - P1棋子
    pieceP2: "#ef4444",        // 红色 - P2棋子
    background: "#f8fafc",     // 浅灰背景
    text: "#1e293b",           // 深色文字
    textOnDark: "#ffffff",     // 浅色文字
  },
  sizes: {
    hexSize: 28,
    pieceSize: 44,
    padding: 80,
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
