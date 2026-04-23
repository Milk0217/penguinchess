// ============================================================
// PenguinChess — Parallelogram Board Layout
// ============================================================
//
// The standard PenguinChess board: 60 hexes in a parallelogram.
//
//   q →  -4  -3  -2  -1   0  +1  +2  +3
// r = -4:          ░░  ░░  ░░  ░░  ░░
// r = -3:       ░░  ░░  ░░  ░░  ░░  ░░
// r = -2:    ░░  ░░  ░░  ░░  ░░  ░░  ░░
// r = -1:       ░░  ░░  ░░  ░░  ░░  ░░
// r =  0:    ░░  ░░  ░░  ░░  ░░  ░░  ░░
// r = +1:       ░░  ░░  ░░  ░░  ░░  ░░
// r = +2:    ░░  ░░  ░░  ░░  ░░  ░░  ░░
//
// Even q columns (-4, -2, 0, 2):  r ∈ [-4, 3]  (8 hexes)
// Odd  q columns (-3, -1, 1, 3):  r ∈ [-3, 3]  (7 hexes)
// Total: 4×8 + 4×7 = 60
// ============================================================

import type {
  BoardLayout,
  HexCoord,
  LayoutConfig,
  PixelCoord,
  Bounds,
} from "../types";

const Q_MIN = -4;
const Q_MAX = 3;

function generateHexes(): HexCoord[] {
  const hexes: HexCoord[] = [];
  for (let q = Q_MIN; q <= Q_MAX; q++) {
    const rMin = q % 2 === 0 ? -4 : -3;
    const rMax = 3;
    for (let r = rMin; r <= rMax; r++) {
      hexes.push({ q, r, s: -q - r });
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

export const parallelogramLayout: BoardLayout = {
  id: "parallelogram",
  name: "Parallelogram",
  description:
    "Standard 60-hex parallelogram board (8 columns × 7-8 rows, staggered).",
  generateHexes,
  cubeToPixel,
  getBounds,
};
