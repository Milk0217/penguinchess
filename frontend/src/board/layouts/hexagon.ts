// ============================================================
// PenguinChess — Hexagonal Board Layout
// ============================================================
//
// A symmetric hexagonal board defined by a configurable radius.
//
//   Radius 3:           37 hexes
//   Radius 4 (default): 61 hexes
//
//   Coordinates satisfy: q + r + s = 0  (cube constraint)
//   Valid range: |q|, |r|, |s| <= radius
//
// ============================================================

import type {
  BoardLayout,
  HexCoord,
  LayoutConfig,
  PixelCoord,
  Bounds,
} from "../types";

const DEFAULT_RADIUS = 4;

function generateHexes(radius: number = DEFAULT_RADIUS): HexCoord[] {
  const hexes: HexCoord[] = [];
  for (let q = -radius; q <= radius; q++) {
    for (let r = -radius; r <= radius; r++) {
      const s = -q - r;
      if (Math.abs(s) <= radius) {
        hexes.push({ q, r, s });
      }
    }
  }
  return hexes;
}

function cubeToPixel(q: number, r: number, config: LayoutConfig): PixelCoord {
  const { hexSize } = config;
  return {
    x: hexSize * 1.5 * q,
    y: hexSize * (Math.sqrt(3) * 0.5 * q + Math.sqrt(3) * r),
  };
}

function getBounds(hexes: HexCoord[], config: LayoutConfig): Bounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (const hex of hexes) {
    const { x, y } = cubeToPixel(hex.q, hex.r, config);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  // 扩展边界以包含完整的六边形尺寸
  const hexSize = config.hexSize;
  minX -= hexSize;
  maxX += hexSize;
  minY -= hexSize;
  maxY += hexSize;

  return {
    minX,
    maxX,
    minY,
    maxY,
    width: maxX - minX,
    height: maxY - minY,
  };
}

export const hexagonLayout: BoardLayout = {
  id: "hexagon",
  name: "Hexagon",
  description:
    "Symmetric hexagonal board (radius 4 = 37 hexes, radius 5 = 61 hexes).",
  generateHexes,
  cubeToPixel,
  getBounds,
};
