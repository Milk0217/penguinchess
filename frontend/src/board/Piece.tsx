import React from "react";
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

const Piece: React.FC<PieceProps> = ({
  piece,
  x,
  y,
  colors,
  sizes,
  isSelected,
  isCurrentPlayer,
}) => {
  // Background color based on owner
  const bgColor = piece.owner === 0 ? colors.pieceP1 : colors.pieceP2;

  // Border color
  const borderColor = isSelected
    ? piece.owner === 0
      ? "#1d4ed8"
      : "#b91c1c"
    : "rgba(255,255,255,0.3)";

  // Ring color (outer glow when selected)
  const ringColor = isSelected
    ? piece.owner === 0
      ? "#93c5fd"
      : "#fca5a5"
    : "transparent";

  // Box shadow
  const boxShadow = isSelected
    ? `0 0 0 3px ${ringColor}, 0 4px 12px rgba(0,0,0,0.4)`
    : "0 2px 8px rgba(0,0,0,0.3)";

  return (
    <div
      title={`棋子 ${piece.id} (${piece.owner === 0 ? "P1" : "P2"})`}
      style={{
        position: "absolute",
        left: x,
        top: y,
        width: sizes.pieceSize,
        height: sizes.pieceSize,
        borderRadius: "50%",
        backgroundColor: bgColor,
        border: `${sizes.borderWidth}px solid ${borderColor}`,
        boxShadow,
        transform: "translate(-50%, -50%)",
        zIndex: isCurrentPlayer ? 35 : 25,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 12,
        fontWeight: 800,
        color: "white",
        pointerEvents: "none",
        userSelect: "none",
      }}
    >
      {piece.id}
    </div>
  );
};

export default Piece;
