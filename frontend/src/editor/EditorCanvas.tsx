/**
 * 棋盘编辑器画布
 * 显示可点击的六边形网格，支持选择/取消选择
 */

import React, { useCallback, useMemo } from "react";
import type { HexCoord, PixelCoord } from "../board/types";

interface EditorCanvasProps {
  selected: Set<string>;      // 已选择的格子 key="q,r,s"
  onToggle: (key: string) => void;
  hexSize?: number;
  zoom?: number;             // 缩放级别
}

// 画布使用的格子生成器（基于 radius=10，覆盖所有可能位置）
// 支持创建各种形状的 60 格棋盘
function generateCanvasHexes(): HexCoord[] {
  const hexes: HexCoord[] = [];
  const radius = 10;
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

// 立方体坐标转像素坐标
function cubeToPixel(q: number, r: number, hexSize: number): PixelCoord {
  return {
    x: hexSize * 1.5 * q,
    y: hexSize * (Math.sqrt(3) * 0.5 * q + Math.sqrt(3) * r),
  };
}

// 计算画布边界
function computeBounds(hexes: HexCoord[], hexSize: number) {
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;

  for (const h of hexes) {
    const { x, y } = cubeToPixel(h.q, h.r, hexSize);
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }

  // 扩展边界
  const pad = hexSize * 2;
  return {
    minX: minX - pad,
    maxX: maxX + pad,
    minY: minY - pad,
    maxY: maxY + pad,
    width: (maxX - minX) + pad * 2,
    height: (maxY - minY) + pad * 2,
  };
}

// 六边形路径
function hexPath(size: number): string {
  const angles = [0, 60, 120, 180, 240, 300];
  const points = angles.map((angle) => {
    const rad = (Math.PI / 180) * angle;
    return `${size * Math.cos(rad)},${size * Math.sin(rad)}`;
  });
  return `M ${points.join(" L ")} Z`;
}

const EditorCanvas: React.FC<EditorCanvasProps> = ({
  selected,
  onToggle,
  hexSize = 40,
  zoom = 1,
}) => {
  // 生成画布上所有可能的格子
  const canvasHexes = useMemo(() => generateCanvasHexes(), []);

  // 计算画布尺寸
  const bounds = useMemo(() => computeBounds(canvasHexes, hexSize), [canvasHexes, hexSize]);

  // 中心偏移，使棋盘居中
  const offsetX = -bounds.minX;
  const offsetY = -bounds.minY;

  const handleHexClick = useCallback(
    (key: string) => {
      onToggle(key);
    },
    [onToggle]
  );

  // 计算每个格子的像素位置（用于排序和渲染）
  const positionedHexes = useMemo(() => {
    return canvasHexes.map((h) => {
      const { x, y } = cubeToPixel(h.q, h.r, hexSize);
      return {
        ...h,
        key: `${h.q},${h.r},${h.s}`,
        px: x + offsetX,
        py: y + offsetY,
        selected: selected.has(`${h.q},${h.r},${h.s}`),
      };
    });
  }, [canvasHexes, hexSize, offsetX, offsetY, selected]);

  // 按行分组，用于 z-index 排序
  const groupedHexes = useMemo(() => {
    const groups: Map<number, typeof positionedHexes> = new Map();
    for (const h of positionedHexes) {
      const row = h.r;
      if (!groups.has(row)) groups.set(row, []);
      groups.get(row)!.push(h);
    }
    return groups;
  }, [positionedHexes]);

  const hexPathD = hexPath(hexSize);

  // SVG 尺寸
  const svgWidth = bounds.width * zoom;
  const svgHeight = bounds.height * zoom;

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        overflow: "auto",
      }}
    >
      <svg
        width={Math.min(svgWidth, 900)}
        height={svgHeight}
        viewBox={`0 0 ${bounds.width} ${bounds.height}`}
        style={{
          background: "#1a1a2e",
          borderRadius: "8px",
          display: "block",
          maxWidth: "100%",
          height: "auto",
        }}
      >
      {/* 渲染所有格子 */}
      {Array.from(groupedHexes.entries())
        .sort(([a], [b]) => a - b)
        .map(([row, hexes]) => (
          <g key={row}>
            {hexes
              .sort((a, b) => a.q - b.q)
              .map((h) => (
                <g
                  key={h.key}
                  transform={`translate(${h.px}, ${h.py})`}
                  onClick={() => handleHexClick(h.key)}
                  style={{ cursor: "pointer" }}
                >
                  {/* 六边形背景 */}
                  <path
                    d={hexPathD}
                    fill={h.selected ? "#4ade80" : "#334155"}
                    fillOpacity={h.selected ? 1 : 0.3}
                    stroke={h.selected ? "#22c55e" : "#475569"}
                    strokeWidth={h.selected ? 2 : 1}
                    style={{
                      transition: "all 0.15s ease",
                    }}
                  />
                  {/* 悬停效果 */}
                  <path
                    d={hexPathD}
                    fill="transparent"
                    stroke="transparent"
                    strokeWidth={3}
                    onMouseEnter={(e) => {
                      if (!h.selected) {
                        e.currentTarget.style.stroke = "#94a3b8";
                      }
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.stroke = "transparent";
                    }}
                  />
                </g>
              ))}
          </g>
        ))}
      </svg>
    </div>
  );
};

export default EditorCanvas;
