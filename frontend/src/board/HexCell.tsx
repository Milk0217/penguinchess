import React from "react";
import type { HexData, ThemeColors, ThemeSizes, ThemeEffects } from "./types";

// 与 penguinchess/core.py Q_ADJUSTMENTS 保持一致的坐标调整表
// 用于将后端调整后坐标转换回原始显示坐标
const Q_ADJUSTMENTS: Record<string, number> = {
  "-4": 2, "-3": 1, "-2": 0, "-1": 0,
  "0": 0, "1": -1, "2": -2, "3": -2,
};

function toRawCoords(hex: HexData): { q: number; r: number } {
  // 后端存储: hex.q = adjusted_r = raw_r + adjustment
  //            hex.r = adjusted_s = -raw_q - adjusted_r
  //            hex.s = raw_q
  const raw_q = hex.s;
  const adj = Q_ADJUSTMENTS[String(raw_q)] ?? 0;
  const raw_r = hex.q - adj;
  return { q: raw_q, r: raw_r };
}

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

function getBackgroundColor(
  hex: HexData,
  isTarget: boolean,
  colors: ThemeColors,
): string {
  if (isTarget) return colors.hexTarget;
  if (hex.state === 'eliminated' || hex.state === 'used') return colors.hexEliminated;
  if (hex.state === 'occupied') return colors.hexOccupied;
  if (hex.points === 1) return colors.hexValue1;
  if (hex.points === 2) return colors.hexValue2;
  if (hex.points === 3) return colors.hexValue3;
  return colors.hexValue1;
}

function getTextColor(hex: HexData, colors: ThemeColors): string {
  if (hex.points === 2 || hex.state === 'occupied') return colors.textOnDark;
  return colors.text;
}

const HexCell: React.FC<HexCellProps> = ({
  hex,
  x,
  y,
  colors,
  sizes,
  effects,
  isTarget,
  onClick,
}) => {
  const isEliminated = hex.state === 'eliminated' || hex.state === 'used';

  const handleClick = () => {
    if (!isEliminated && onClick) {
      onClick(hex);
    }
  };

  const cellSize = sizes.hexSize * 2;

  const style: React.CSSProperties = {
    position: "absolute",
    left: `${x}px`,
    top: `${y}px`,
    width: `${cellSize}px`,
    height: `${cellSize}px`,
    transform: "translate(-50%, -50%)",
    clipPath: "polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%)",
    backgroundColor: getBackgroundColor(hex, isTarget, colors),
    color: getTextColor(hex, colors),
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    cursor: isEliminated ? "default" : "pointer",
    userSelect: "none",
    zIndex: isTarget ? 15 : 5,
    transition: effects.animation ? "all 0.2s ease" : undefined,
    fontSize: `${sizes.hexSize * 0.6}px`,
  };

  const showValue = effects.showValues && hex.state === 'active';
  const showCoords = effects.showCoords;

  return (
    <div style={style} onClick={handleClick}>
      {showValue && <span>{hex.points}</span>}
      {showCoords && (
        <span style={{ fontSize: `${sizes.hexSize * 0.3}px`, opacity: 0.7 }}>
          ({toRawCoords(hex).q},{toRawCoords(hex).r})
        </span>
      )}
    </div>
  );
};

export default HexCell;
