/**
 * 企鹅棋 · 棋盘编辑器
 *
 * 可视化编辑 60 格正六边形棋盘布局
 * 支持保存到后端，直接在游戏中选择
 */

import React, { useCallback, useMemo, useState, useEffect } from "react";
import EditorCanvas from "./EditorCanvas";
import Sidebar from "./Sidebar";
import { getTemplateById, hexToKey } from "./templates";
import type { HexCoord } from "../board/types";
import { api, type BoardInfo } from "../api";

interface BoardEditorProps {
  onBack: () => void;
}

const TARGET_COUNT = 60;

// 生成导出代码
function generateExportCode(selectedHexes: HexCoord[], boardName: string): string {
  const id = `custom-${Date.now().toString(36)}`;

  // 排序输出（按 q,r 排序）
  const sorted = [...selectedHexes].sort((a, b) => {
    if (a.q !== b.q) return a.q - b.q;
    return a.r - b.r;
  });

  const hexList = sorted
    .map((h) => `    { q: ${h.q}, r: ${h.r}, s: ${h.s} }`)
    .join(",\n");

  return `// frontend/src/board/layouts/${id}.ts
// 棋盘编辑器导出 - ${boardName}

import type { BoardLayout, HexCoord, LayoutConfig, PixelCoord, Bounds } from "../types";

function generateHexes(): HexCoord[] {
  return [
${hexList}
  ];
}

function cubeToPixel(q: number, r: number, config: LayoutConfig): PixelCoord {
  const { hexSize } = config;
  return {
    x: hexSize * 1.5 * q,
    y: hexSize * (Math.sqrt(3) * 0.5 * q + Math.sqrt(3) * r),
  };
}

function getBounds(hexes: HexCoord[], config: LayoutConfig): Bounds {
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;

  for (const hex of hexes) {
    const { x, y } = cubeToPixel(hex.q, hex.r, config);
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }

  const hexSize = config.hexSize;
  minX -= hexSize;
  maxX += hexSize;
  minY -= hexSize;
  maxY += hexSize;

  return { minX, maxX, minY, maxY, width: maxX - minX, height: maxY - minY };
}

export const ${id.replace(/-/g, "_")}: BoardLayout = {
  id: "${id}",
  name: "${boardName}",
  description: "User-designed 60-hex board",
  generateHexes,
  cubeToPixel,
  getBounds,
};
`;
}

const BoardEditor: React.FC<BoardEditorProps> = ({ onBack }) => {
  // 状态
  const [selected, setSelected] = useState<Set<string>>(() => {
    const template = getTemplateById("parallelogram");
    if (template) {
      return new Set(template.hexes.map(hexToKey));
    }
    return new Set();
  });
  const [boardName, setBoardName] = useState("我的棋盘");
  const [exportCode, setExportCode] = useState<string | null>(null);

  // 保存状态
  const [savedBoards, setSavedBoards] = useState<BoardInfo[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // 缩放状态
  const [zoom, setZoom] = useState(1);

  // 加载已保存棋盘列表
  const loadBoards = useCallback(async () => {
    try {
      const boards = await api.getBoards();
      setSavedBoards(boards);
    } catch (e) {
      console.error("Failed to load boards:", e);
    }
  }, []);

  useEffect(() => {
    loadBoards();
  }, [loadBoards]);

  // 将 Set<string> 转换为 HexCoord[]
  const selectedHexes = useMemo(() => {
    return Array.from(selected).map((key) => {
      const [q, r, s] = key.split(",").map(Number);
      return { q, r, s } as HexCoord;
    });
  }, [selected]);

  // 切换格子选择状态
  const handleToggle = useCallback((key: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
    setExportCode(null); // 清除导出代码
  }, []);

  // 清空选择
  const handleClear = useCallback(() => {
    setSelected(new Set());
    setExportCode(null);
    setBoardName("我的棋盘");
  }, []);

  // 随机生成
  const handleRandom = useCallback(() => {
    const allHexes: HexCoord[] = [];
    for (let q = -10; q <= 10; q++) {
      for (let r = -10; r <= 10; r++) {
        const s = -q - r;
        if (Math.abs(s) <= 10) {
          allHexes.push({ q, r, s });
        }
      }
    }

    const shuffled = [...allHexes].sort(() => Math.random() - 0.5);
    const selected60 = shuffled.slice(0, TARGET_COUNT);
    setSelected(new Set(selected60.map(hexToKey)));
    setExportCode(null);
  }, []);

  // 保存到后端
  const handleSave = useCallback(async () => {
    if (selected.size !== TARGET_COUNT) return;

    setIsSaving(true);
    try {
      const result = await api.saveBoard(boardName, selectedHexes);
      console.log("Board saved:", result);
      await loadBoards(); // 刷新列表
      setExportCode(generateExportCode(selectedHexes, boardName));
      alert(`棋盘 "${boardName}" 保存成功！可以在游戏中选择该棋盘了。`);
    } catch (e: any) {
      alert(`保存失败: ${e.message}`);
    } finally {
      setIsSaving(false);
    }
  }, [boardName, selectedHexes, selected.size, loadBoards]);

  // 加载已保存棋盘
  const handleLoad = useCallback(async (boardId: string) => {
    setIsLoading(true);
    try {
      const boards = await api.getBoards();
      const board = boards.find((b: BoardInfo) => b.id === boardId);
      if (board) {
        // 需要获取完整数据才能加载
        // 目前通过保存时的导出代码功能来获取
        alert(`请在游戏中选择棋盘 "${board.name}" 进行对战。\n\n如需编辑，请使用导出代码功能。`);
      }
    } catch (e: any) {
      alert(`加载失败: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // 删除棋盘
  const handleDelete = useCallback(async (boardId: string) => {
    if (!confirm(`确定要删除棋盘 "${boardId}" 吗？`)) return;

    setIsLoading(true);
    try {
      await api.deleteBoard(boardId);
      await loadBoards();
      alert("删除成功");
    } catch (e: any) {
      alert(`删除失败: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [loadBoards]);

  // 响应式断点样式
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#020617",
        color: "#f1f5f9",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* 顶部导航栏 */}
      <header
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.75rem",
          padding: "0.75rem 1rem",
          background: "#0f172a",
          borderBottom: "1px solid #1e293b",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <button
            onClick={onBack}
            style={{
              padding: "0.4rem 0.8rem",
              background: "#334155",
              border: "none",
              borderRadius: "6px",
              color: "#f1f5f9",
              fontSize: "0.85rem",
              cursor: "pointer",
              transition: "background 0.15s",
              whiteSpace: "nowrap",
            }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.background = "#475569")
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.background = "#334155")
            }
          >
            ← 返回
          </button>
          <h1
            style={{
              margin: 0,
              fontSize: "1.1rem",
              fontWeight: "bold",
              color: "#f1f5f9",
            }}
          >
            企鹅棋 · 棋盘编辑器
          </h1>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          {/* 缩放控制 */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <span style={{ color: "#94a3b8", fontSize: "0.8rem" }}>缩放:</span>
            <input
              type="range"
              min="0.2"
              max="2"
              step="0.1"
              value={zoom}
              onChange={(e) => setZoom(parseFloat(e.target.value))}
              style={{ width: "80px" }}
            />
            <span style={{ color: "#f1f5f9", fontSize: "0.8rem", minWidth: "36px" }}>
              {Math.round(zoom * 100)}%
            </span>
          </div>
          {!isMobile && (
            <div style={{ color: "#64748b", fontSize: "0.8rem" }}>
              设计你的专属 60 格正六边形棋盘
            </div>
          )}
        </div>
      </header>

      {/* 主内容区 */}
      <main
        style={{
          display: "flex",
          flexDirection: isMobile ? "column" : "row",
          gap: isMobile ? "1rem" : "1.5rem",
          padding: isMobile ? "0.75rem" : "1.5rem",
          flex: 1,
          overflow: "hidden",
        }}
      >
        {/* 画布区域 */}
        <div
          style={{
            flex: 1,
            display: "flex",
            justifyContent: "center",
            alignItems: "flex-start",
            overflow: "auto",
            padding: "0.75rem",
            background: "#0f172a",
            borderRadius: "12px",
            minHeight: isMobile ? "50vh" : "auto",
            maxHeight: isMobile ? "60vh" : "calc(100vh - 120px)",
          }}
        >
          <EditorCanvas
            selected={selected}
            onToggle={handleToggle}
            hexSize={isMobile ? 32 : 40}
            zoom={zoom}
          />
        </div>

        {/* 侧边栏 */}
        <Sidebar
          selectedCount={selected.size}
          targetCount={TARGET_COUNT}
          onClear={handleClear}
          onRandom={handleRandom}
          onSave={handleSave}
          onLoad={handleLoad}
          onDelete={handleDelete}
          exportCode={exportCode}
          boardName={boardName}
          onBoardNameChange={setBoardName}
          savedBoards={savedBoards}
          isSaving={isSaving}
          isLoading={isLoading}
          isMobile={isMobile}
        />
      </main>
    </div>
  );
};

export default BoardEditor;
