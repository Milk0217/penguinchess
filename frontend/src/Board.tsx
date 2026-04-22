/**
 * 六边形棋盘渲染组件（纯展示）。
 *
 * 职责：
 * - 根据 GameState 渲染所有格子和棋子
 * - 高亮显示 targetIndices 传入的目标格子
 * - 触发 onHexClick 回调，将点击事件传给 App 处理
 *
 * 所有游戏逻辑（路径检查、合法性判断）都在 App.tsx 层执行。
 */

import type { HexData, PieceData } from "./api";

// 六边形网格尺寸
const HEX_SIZE = 22;
const HEX_DIAMETER = 44;
const BOARD_PADDING = 30;

// 立方体坐标 → 像素坐标（扁平化六边形网格）
function cubeToPixel(q: number, r: number): { x: number; y: number } {
  const x = HEX_SIZE * (3 / 2) * q;
  const y = HEX_SIZE * (Math.sqrt(3) / 2 * q + Math.sqrt(3) * r);
  return { x, y };
}

// 根据格子状态计算背景色
function hexBg(q: number, value: number): string {
  if (value < 0) return "#e2e8f0";   // 已消除
  if (value === 0) return "#94a3b8";  // 被占据
  const hue = ((q + 4) / 7) * 60 + 160; // 160~220 蓝绿色系
  return `hsl(${hue}, 55%, 72%)`;
}

interface BoardProps {
  /** 完整游戏状态 */
  state: {
    hexes: HexData[];
    pieces: PieceData[];
    current_player: number;
    legal_actions: number[];
  };
  /** 移动阶段：当前选中的己方棋子 ID */
  selectedPieceId: number | null;
  /** 要高亮的目标格子 index 集合（App 层计算后传入） */
  targetIndices: Set<number>;
  /** 格子点击回调 */
  onHexClick: (hex: HexData) => void;
}

export default function Board({ state, selectedPieceId, targetIndices, onHexClick }: BoardProps) {
  const { hexes, pieces, current_player } = state;

  // 棋盘边界（一次性计算）
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const h of hexes) {
    const { x, y } = cubeToPixel(h.q, h.r);
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const offsetX = -minX + BOARD_PADDING;
  const offsetY = -minY + BOARD_PADDING;
  const containerW = Math.ceil(maxX - minX + HEX_DIAMETER + BOARD_PADDING * 2);
  const containerH = Math.ceil(maxY - minY + HEX_DIAMETER + BOARD_PADDING * 2);

  return (
    <div
      className="board-container"
      style={{ width: Math.min(containerW, 680), height: Math.min(containerH, 680), margin: "0 auto" }}
    >
      {/* 格子层 */}
      {hexes.map((h) => {
        const { x, y } = cubeToPixel(h.q, h.r);
        const px = x + offsetX;
        const py = y + offsetY;
        const isEliminated = h.value < 0;
        const isOccupied = h.value === 0;
        const isTarget = targetIndices.has(h.index);

        let bg = hexBg(h.q, h.value);
        if (isTarget) bg = "#fb923c";
        else if (isEliminated) bg = "#e2e8f0";
        else if (isOccupied) bg = "#94a3b8";

        return (
          <button
            key={h.index}
            style={{
              position: "absolute",
              left: px,
              top: py,
              width: HEX_DIAMETER,
              height: HEX_DIAMETER,
              background: bg,
              transform: "translate(-50%, -50%)",
              clipPath: "polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%)",
              zIndex: isTarget ? 15 : 5,
              border: "none",
              cursor: isEliminated ? "default" : "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
            onClick={() => !isEliminated && onHexClick(h)}
            title={`(${h.q}, ${h.r}, ${h.s}) = ${h.value}`}
          >
            {h.value > 0 && (
              <span style={{ fontSize: 12, fontWeight: 700, color: "#1e293b", pointerEvents: "none" }}>
                {h.value}
              </span>
            )}
          </button>
        );
      })}

      {/* 棋子层（与格子平级，确保正确层叠） */}
      {pieces.map((piece) => {
        if (!piece.alive || piece.q === null) return null;
        const { x, y } = cubeToPixel(piece.q, piece.r);
        const px = x + offsetX;
        const py = y + offsetY;
        const isSelected = selectedPieceId === piece.id;
        const isCurrent = piece.owner === current_player;
        const borderColor = isSelected
          ? piece.owner === 0 ? "#1d4ed8" : "#b91c1c"
          : "rgba(255,255,255,0.3)";
        const ringColor = isSelected
          ? piece.owner === 0 ? "#93c5fd" : "#fca5a5"
          : "transparent";

        return (
          <div
            key={piece.id}
            style={{
              position: "absolute",
              left: px,
              top: py,
              width: HEX_DIAMETER - 10,
              height: HEX_DIAMETER - 10,
              borderRadius: "50%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transform: "translate(-50%, -50%)",
              border: `2px solid ${borderColor}`,
              boxShadow: isSelected
                ? `0 0 0 3px ${ringColor}, 0 4px 12px rgba(0,0,0,0.4)`
                : "0 2px 8px rgba(0,0,0,0.3)",
              zIndex: isCurrent ? 35 : 25,
              fontSize: 11,
              fontWeight: 800,
              color: "white",
              background: piece.owner === 0 ? "#3b82f6" : "#ef4444",
              pointerEvents: "none",
              userSelect: "none",
            }}
            title={`棋子 ${piece.id} (${piece.owner === 0 ? "P1" : "P2"})`}
          >
            {piece.id}
          </div>
        );
      })}
    </div>
  );
}
