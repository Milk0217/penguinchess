/**
 * 棋盘编辑器预设模板
 * 每个模板定义了一组初始选中的格子
 */

import type { HexCoord } from "../board/types";

interface Template {
  id: string;
  name: string;
  description: string;
  hexes: HexCoord[];
}

// 生成平行四边形 60 格
function generateParallelogram(): HexCoord[] {
  const hexes: HexCoord[] = [];
  for (let q = -4; q <= 3; q++) {
    const rMin = q % 2 === 0 ? -4 : -3;
    const rMax = 3;
    for (let r = rMin; r <= rMax; r++) {
      hexes.push({ q, r, s: -q - r });
    }
  }
  return hexes;
}

// 生成矩形排列 60 格（10×6）
function generateRectangle(): HexCoord[] {
  const hexes: HexCoord[] = [];
  // 矩形排列使用 offset 坐标，这里转换为立方体坐标
  // 10列，交替排列
  for (let col = 0; col < 10; col++) {
    const q = col - 4;  // q 从 -4 到 +5
    const rMin = col % 2 === 0 ? -2 : -3;
    const rMax = rMin + 5;  // 每列6个
    for (let r = rMin; r <= rMax; r++) {
      hexes.push({ q, r, s: -q - r });
    }
  }
  return hexes;
}

// 生成钻石形 60 格
function generateDiamond(): HexCoord[] {
  // 以中心为 (0,0,0)，向外扩展
  // 钻石形状：|q| + |r| + |s| <= 5，但只取其中60个
  const allHexes: HexCoord[] = [];
  for (let q = -5; q <= 5; q++) {
    for (let r = -5; r <= 5; r++) {
      const s = -q - r;
      if (Math.abs(s) <= 5) {
        allHexes.push({ q, r, s });
      }
    }
  }
  // 按到中心距离排序，选取最近的60个
  allHexes.sort((a, b) => {
    const distA = Math.max(Math.abs(a.q), Math.abs(a.r), Math.abs(a.s));
    const distB = Math.max(Math.abs(b.q), Math.abs(b.r), Math.abs(b.s));
    return distA - distB;
  });
  return allHexes.slice(0, 60);
}

// 生成正六边形 61 格（参考布局）
function generateHexagon(): HexCoord[] {
  const hexes: HexCoord[] = [];
  const radius = 4;
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

// 空模板
function generateEmpty(): HexCoord[] {
  return [];
}

// 预设模板列表
export const templates: Template[] = [
  {
    id: "parallelogram",
    name: "平行四边形",
    description: "标准企鹅棋布局 (60 格)",
    hexes: generateParallelogram(),
  },
  {
    id: "rectangle",
    name: "长方形",
    description: "10×6 矩形排列 (60 格)",
    hexes: generateRectangle(),
  },
  {
    id: "diamond",
    name: "钻石形",
    description: "菱形对称排列 (60 格)",
    hexes: generateDiamond(),
  },
  {
    id: "hexagon",
    name: "正六边形",
    description: "中心对称六边形 (61 格，需取消1个)",
    hexes: generateHexagon(),
  },
  {
    id: "empty",
    name: "空模板",
    description: "从零开始自由绘制",
    hexes: generateEmpty(),
  },
];

/**
 * 根据 ID 获取模板
 */
export function getTemplateById(id: string): Template | undefined {
  return templates.find((t) => t.id === id);
}

/**
 * 将 HexCoord 集合转换为 key 字符串
 */
export function hexToKey(h: HexCoord): string {
  return `${h.q},${h.r},${h.s}`;
}

/**
 * 将 key 字符串转换回 HexCoord
 */
export function keyToHex(key: string): HexCoord {
  const [q, r, s] = key.split(",").map(Number);
  return { q, r, s };
}
