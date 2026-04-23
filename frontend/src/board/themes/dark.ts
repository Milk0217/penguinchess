import type { BoardTheme } from "../types";

/** 暗色主题 — 护眼暗色配色 */
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
