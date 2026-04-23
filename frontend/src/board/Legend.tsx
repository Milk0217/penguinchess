// ============================================================
// PenguinChess — Legend Component
// ============================================================

import React from "react";
import type { ThemeColors } from "./types";

interface LegendProps {
  colors: ThemeColors;
  showCoords: boolean;
  onToggleCoords: () => void;
}

interface LegendItem {
  label: string;
  color: string;
  round: boolean;
}

const Legend: React.FC<LegendProps> = ({ colors, showCoords, onToggleCoords }) => {
  const items: LegendItem[] = [
    { label: "1分", color: colors.hexValue1, round: false },
    { label: "2分", color: colors.hexValue2, round: false },
    { label: "3分", color: colors.hexValue3, round: false },
    { label: "被占据", color: colors.hexOccupied, round: false },
    { label: "已消除", color: colors.hexEliminated, round: false },
    { label: "可移动", color: colors.hexTarget, round: false },
    { label: "P1棋子", color: colors.pieceP1, round: true },
    { label: "P2棋子", color: colors.pieceP2, round: true },
  ];

  const containerStyle: React.CSSProperties = {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  };

  const legendListStyle: React.CSSProperties = {
    display: "flex",
    flexWrap: "wrap",
    gap: 16,
    padding: "8px 16px",
    background: colors.background,
    borderRadius: 8,
    border: `1px solid ${colors.hexEliminated}`,
    fontSize: 12,
    color: colors.text,
  };

  const itemStyle = (): React.CSSProperties => ({
    display: "flex",
    alignItems: "center",
    gap: 6,
  });

  const swatchStyle = (color: string, round: boolean): React.CSSProperties => ({
    width: 16,
    height: 16,
    background: color,
    borderRadius: round ? "50%" : 3,
    border: `1px solid ${color}`,
  });

  const buttonStyle: React.CSSProperties = {
    marginTop: 8,
    padding: "4px 12px",
    fontSize: 12,
    background: showCoords ? colors.pieceP1 : colors.hexEliminated,
    color: showCoords ? "white" : colors.text,
    border: "none",
    borderRadius: 4,
    cursor: "pointer",
  };

  return (
    <div style={containerStyle}>
      <div style={legendListStyle}>
        {items.map((item) => (
          <div key={item.label} style={itemStyle()}>
            <div style={swatchStyle(item.color, item.round)} />
            <span>{item.label}</span>
          </div>
        ))}
      </div>
      <button style={buttonStyle} onClick={onToggleCoords}>
        {showCoords ? "隐藏坐标" : "显示坐标"}
      </button>
    </div>
  );
};

export default Legend;
