// ============================================================
// PenguinChess — Board System Core Types
// ============================================================

/** 六边形立方体坐标 (cube coordinates) */
export interface HexCoord {
  q: number;
  r: number;
  s: number;
}

/** 像素坐标 */
export interface PixelCoord {
  x: number;
  y: number;
}

/** 边界框 */
export interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  width: number;
  height: number;
}

/** 布局配置 */
export interface LayoutConfig {
  hexSize: number;
  padding: number;
}

/** 棋盘布局接口 — 不同棋盘形状实现此接口 */
export interface BoardLayout {
  id: string;
  name: string;
  description: string;
  generateHexes(): HexCoord[];
  cubeToPixel(q: number, r: number, config: LayoutConfig): PixelCoord;
  getBounds(hexes: HexCoord[], config: LayoutConfig): Bounds;
}

/** 主题颜色配置 */
export interface ThemeColors {
  hexValue1: string;
  hexValue2: string;
  hexValue3: string;
  hexOccupied: string;
  hexEliminated: string;
  hexTarget: string;
  pieceP1: string;
  pieceP2: string;
  background: string;
  text: string;
  textOnDark: string;
}

/** 主题尺寸配置 */
export interface ThemeSizes {
  hexSize: number;
  pieceSize: number;
  padding: number;
  borderWidth: number;
}

/** 主题效果配置 */
export interface ThemeEffects {
  showCoords: boolean;
  showValues: boolean;
  showBorders: boolean;
  animation: boolean;
  shadow: boolean;
}

/** 完整主题配置 */
export interface BoardTheme {
  id: string;
  name: string;
  description: string;
  colors: ThemeColors;
  sizes: ThemeSizes;
  effects: ThemeEffects;
}

/** 棋盘格子状态（从后端获取） */
export interface HexData {
  index: number;
  q: number;
  r: number;
  s: number;
  value: number;
}

/** 棋子状态 */
export interface PieceData {
  id: number;
  owner: number;
  q: number | null;
  r: number | null;
  s: number | null;
  alive: boolean;
}

/** 游戏状态 */
export interface GameState {
  session_id: string;
  hexes: HexData[];
  pieces: PieceData[];
  current_player: number;
  phase: "placement" | "movement";
  scores: [number, number];
  legal_actions: number[];
  episode_steps: number;
  game_over: boolean;
  winner: number | null;
}
