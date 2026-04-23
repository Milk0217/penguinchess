// ============================================================
// PenguinChess — Board Container Component
// ============================================================

import React, { useState, useMemo } from "react";
import type {
  GameState,
  BoardLayout,
  BoardTheme,
} from "./types";
import HexCell from "./HexCell";
import Piece from "./Piece";
import Legend from "./Legend";

interface BoardContainerProps {
  state: GameState;
  layout: BoardLayout;
  theme: BoardTheme;
  selectedPieceId: number | null;
  targetIndices: Set<number>;
  onHexClick: (hex: GameState["hexes"][0]) => void;
}

const BoardContainer: React.FC<BoardContainerProps> = ({
  state,
  layout,
  theme,
  selectedPieceId,
  targetIndices,
  onHexClick,
}) => {
  const [showCoords, setShowCoords] = useState(theme.effects.showCoords);

  // Generate hex layout coordinates
  const layoutHexes = useMemo(() => layout.generateHexes(), [layout]);

  // Calculate bounds for positioning
  const sizes = useMemo(
    () => ({
      hexSize: theme.sizes.hexSize,
      padding: theme.sizes.padding,
    }),
    [theme.sizes.hexSize, theme.sizes.padding],
  );

  const bounds = useMemo(
    () => layout.getBounds(layoutHexes, sizes),
    [layout, layoutHexes, sizes],
  );

  const offsetX = -bounds.minX + sizes.padding + sizes.hexSize;
  const offsetY = -bounds.minY + sizes.padding + sizes.hexSize;

  const padding = sizes.padding;
  // 容器尺寸 = bounds（含完整六边形尺寸）+ padding*2 + hexSize（补偿 translate(-50%,-50%)）
  // bounds 已包含六边形宽度 (maxX-minX + 2*hexSize)，translate(-50%,-50%) 将内容向左上偏移 hexSize
  // 容器需要额外加 hexSize 才能完整容纳最右侧和最下方的六边形
  const containerWidth = Math.ceil(bounds.width + padding * 2 + sizes.hexSize);
  const containerHeight = Math.ceil(bounds.height + padding * 2 + sizes.hexSize);

  // Map hex index → pixel coordinates
  const hexCoordMap = useMemo(() => {
    const map = new Map<number, { x: number; y: number }>();
    for (const hex of state.hexes) {
      const pixel = layout.cubeToPixel(hex.q, hex.r, sizes);
      map.set(hex.index, {
        x: pixel.x + offsetX,
        y: pixel.y + offsetY,
      });
    }
    return map;
  }, [state.hexes, layout, sizes, offsetX, offsetY]);

  // Board container dimensions (use actual size needed)
  const boardWidth = containerWidth;
  const boardHeight = containerHeight;

  // Filter alive pieces for rendering
  const alivePieces = useMemo(
    () =>
      state.pieces.filter(
        (p) => p.alive && p.q !== null && p.r !== null,
      ),
    [state.pieces],
  );

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      {/* Board */}
      <div
        style={{
          position: "relative",
          width: boardWidth,
          height: boardHeight,
          background: theme.colors.background,
          borderRadius: "12px",
          overflow: "hidden",
        }}
      >
        {/* Hex cells layer */}
        {state.hexes.map((hex) => {
          const coord = hexCoordMap.get(hex.index);
          if (!coord) return null;
          return (
            <HexCell
              key={hex.index}
              hex={hex}
              x={coord.x}
              y={coord.y}
              colors={theme.colors}
              sizes={theme.sizes}
              effects={{
                ...theme.effects,
                showCoords,
              }}
              isTarget={targetIndices.has(hex.index)}
              onClick={onHexClick}
            />
          );
        })}

        {/* Pieces layer */}
        {alivePieces.map((piece) => {
          const pixel = layout.cubeToPixel(piece.q!, piece.r!, sizes);
          const px = pixel.x + offsetX;
          const py = pixel.y + offsetY;
          return (
            <Piece
              key={piece.id}
              piece={piece}
              x={px}
              y={py}
              colors={theme.colors}
              sizes={theme.sizes}
              isSelected={piece.id === selectedPieceId}
              isCurrentPlayer={piece.owner === state.current_player}
            />
          );
        })}
      </div>

      {/* Legend */}
      <Legend
        colors={theme.colors}
        showCoords={showCoords}
        onToggleCoords={() => setShowCoords((prev) => !prev)}
      />
    </div>
  );
};

export default BoardContainer;
