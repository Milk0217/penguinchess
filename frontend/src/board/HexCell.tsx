import React from "react";
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

function getBackgroundColor(
  hex: HexData,
  isTarget: boolean,
  colors: ThemeColors,
): string {
  if (isTarget) return colors.hexTarget;
  if (hex.value < 0) return colors.hexEliminated;
  if (hex.value === 0) return colors.hexOccupied;
  if (hex.value === 1) return colors.hexValue1;
  if (hex.value === 2) return colors.hexValue2;
  if (hex.value === 3) return colors.hexValue3;
  return colors.hexValue1;
}

function getTextColor(hex: HexData, colors: ThemeColors): string {
  if (hex.value === 2 || hex.value === 0) return colors.textOnDark;
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
  const isEliminated = hex.value < 0;

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

  const showValue = effects.showValues && hex.value > 0;
  const showCoords = effects.showCoords;

  return (
    <div style={style} onClick={handleClick}>
      {showValue && <span>{hex.value}</span>}
      {showCoords && (
        <span style={{ fontSize: `${sizes.hexSize * 0.3}px`, opacity: 0.7 }}>
          ({hex.q},{hex.r})
        </span>
      )}
    </div>
  );
};

export default HexCell;
