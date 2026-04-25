/**
 * 棋盘编辑器侧边栏
 * 显示计数、操作按钮、保存/加载功能
 */

import React, { useState } from "react";

interface BoardInfo {
  id: string;
  name: string;
  hex_count: number;
  created_at?: string;
}

interface SidebarProps {
  selectedCount: number;
  targetCount: number;
  onClear: () => void;
  onRandom: () => void;
  onSave: () => void;
  onLoad: (boardId: string) => void;
  onDelete: (boardId: string, boardName?: string) => void;
  exportCode: string | null;
  boardName: string;
  onBoardNameChange: (name: string) => void;
  savedBoards: BoardInfo[];
  isSaving: boolean;
  isLoading: boolean;
  isMobile?: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({
  selectedCount,
  targetCount,
  onClear,
  onRandom,
  onSave,
  onLoad,
  onDelete,
  exportCode,
  boardName,
  onBoardNameChange,
  savedBoards,
  isSaving,
  isLoading,
  isMobile = false,
}) => {
  const [copied, setCopied] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const [showSaved, setShowSaved] = useState(false);

  const isValid = selectedCount === targetCount;
  const progress = Math.min((selectedCount / targetCount) * 100, 100);

  const handleCopy = async () => {
    if (exportCode) {
      await navigator.clipboard.writeText(exportCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: isMobile ? "1rem" : "1.5rem",
        padding: isMobile ? "1rem" : "1.5rem",
        background: "#0f172a",
        borderRadius: "12px",
        minWidth: isMobile ? "100%" : "280px",
        maxWidth: isMobile ? "100%" : "320px",
        maxHeight: isMobile ? "none" : "calc(100vh - 120px)",
        overflowY: isMobile ? "visible" : "auto",
        flexShrink: 0,
      }}
    >
      {/* 棋盘名称 */}
      <div>
        <label
          style={{
            display: "block",
            color: "#94a3b8",
            fontSize: "0.85rem",
            marginBottom: "0.5rem",
          }}
        >
          棋盘名称
        </label>
        <input
          type="text"
          value={boardName}
          onChange={(e) => onBoardNameChange(e.target.value)}
          placeholder="输入棋盘名称"
          style={{
            width: "100%",
            padding: "0.6rem 0.8rem",
            background: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "6px",
            color: "#f1f5f9",
            fontSize: "0.9rem",
            boxSizing: "border-box",
          }}
        />
      </div>

      {/* 计数显示 */}
      <div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "0.5rem",
          }}
        >
          <span style={{ color: "#94a3b8", fontSize: "0.85rem" }}>
            已选择
          </span>
          <span
            style={{
              color: isValid ? "#4ade80" : "#f87171",
              fontSize: "1.2rem",
              fontWeight: "bold",
            }}
          >
            {selectedCount}/{targetCount}
          </span>
        </div>
        <div
          style={{
            width: "100%",
            height: "8px",
            background: "#1e293b",
            borderRadius: "4px",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${progress}%`,
              height: "100%",
              background: isValid
                ? "linear-gradient(90deg, #22c55e, #4ade80)"
                : "linear-gradient(90deg, #ef4444, #f87171)",
              transition: "width 0.2s ease",
            }}
          />
        </div>
        {!isValid && (
          <p
            style={{
              color: "#f87171",
              fontSize: "0.75rem",
              marginTop: "0.4rem",
            }}
          >
            {selectedCount < targetCount
              ? `还需选择 ${targetCount - selectedCount} 个格子`
              : `多选了 ${selectedCount - targetCount} 个格子`}
          </p>
        )}
        {isValid && (
          <p
            style={{
              color: "#4ade80",
              fontSize: "0.75rem",
              marginTop: "0.4rem",
            }}
          >
            ✓ 可以保存
          </p>
        )}
      </div>

      {/* 操作按钮 */}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        <button
          onClick={onClear}
          style={{
            padding: "0.6rem 1rem",
            background: "#334155",
            border: "none",
            borderRadius: "6px",
            color: "#f1f5f9",
            fontSize: "0.9rem",
            cursor: "pointer",
            transition: "background 0.15s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#475569")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "#334155")}
        >
          清空
        </button>
        <button
          onClick={onRandom}
          style={{
            padding: "0.6rem 1rem",
            background: "#334155",
            border: "none",
            borderRadius: "6px",
            color: "#f1f5f9",
            fontSize: "0.9rem",
            cursor: "pointer",
            transition: "background 0.15s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#475569")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "#334155")}
        >
          随机生成
        </button>
      </div>

      {/* 保存按钮 */}
      <button
        onClick={onSave}
        disabled={!isValid || isSaving}
        style={{
          padding: "0.8rem 1rem",
          background: isValid && !isSaving ? "#4ade80" : "#334155",
          border: "none",
          borderRadius: "6px",
          color: isValid && !isSaving ? "#052e16" : "#64748b",
          fontSize: "0.95rem",
          fontWeight: "bold",
          cursor: isValid && !isSaving ? "pointer" : "not-allowed",
          transition: "all 0.15s",
        }}
      >
        {isSaving ? "保存中..." : "💾 保存到后端"}
      </button>

      {/* 已保存棋盘列表 */}
      <div>
        <button
          onClick={() => setShowSaved(!showSaved)}
          style={{
            width: "100%",
            padding: "0.6rem 1rem",
            background: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "6px",
            color: "#f1f5f9",
            fontSize: "0.9rem",
            cursor: "pointer",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span>📁 已保存棋盘 ({savedBoards.length})</span>
          <span>{showSaved ? "▲" : "▼"}</span>
        </button>

        {showSaved && (
          <div
            style={{
              marginTop: "0.5rem",
              maxHeight: "200px",
              overflowY: "auto",
            }}
          >
            {savedBoards.length === 0 ? (
              <p style={{ color: "#64748b", fontSize: "0.8rem", padding: "0.5rem" }}>
                暂无已保存棋盘
              </p>
            ) : (
              savedBoards.map((board) => (
                <div
                  key={board.id}
                  style={{
                    padding: "0.5rem",
                    background: "#1e293b",
                    borderRadius: "4px",
                    marginTop: "0.25rem",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                  }}
                >
                  <div>
                    <div style={{ color: "#f1f5f9", fontSize: "0.85rem" }}>
                      {board.name}
                    </div>
                    <div style={{ color: "#64748b", fontSize: "0.7rem" }}>
                      {board.hex_count}格 · {board.id}
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: "0.25rem" }}>
                    <button
                      onClick={() => onLoad(board.id)}
                      disabled={isLoading}
                      style={{
                        padding: "0.25rem 0.5rem",
                        background: "#334155",
                        border: "none",
                        borderRadius: "4px",
                        color: "#f1f5f9",
                        fontSize: "0.7rem",
                        cursor: "pointer",
                      }}
                    >
                      加载
                    </button>
                    <button
                      onClick={() => onDelete(board.id, board.name)}
                      disabled={isLoading}
                      style={{
                        padding: "0.25rem 0.5rem",
                        background: "#dc2626",
                        border: "none",
                        borderRadius: "4px",
                        color: "#fff",
                        fontSize: "0.7rem",
                        cursor: "pointer",
                      }}
                    >
                      删除
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* 导出代码 */}
      <button
        onClick={() => {
          if (exportCode) setShowExport(!showExport);
        }}
        disabled={!exportCode}
        style={{
          padding: "0.6rem 1rem",
          background: exportCode ? "#334155" : "#1e293b",
          border: "none",
          borderRadius: "6px",
          color: exportCode ? "#f1f5f9" : "#475569",
          fontSize: "0.9rem",
          cursor: exportCode ? "pointer" : "not-allowed",
        }}
      >
        📋 导出代码
      </button>

      {/* 导出代码显示 */}
      {showExport && exportCode && (
        <div
          style={{
            marginTop: "1rem",
            padding: "1rem",
            background: "#0d1117",
            borderRadius: "8px",
            border: "1px solid #30363d",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "0.5rem",
            }}
          >
            <span
              style={{ color: "#58a6ff", fontSize: "0.85rem", fontWeight: "bold" }}
            >
              TypeScript 代码
            </span>
            <button
              onClick={handleCopy}
              style={{
                padding: "0.3rem 0.6rem",
                background: copied ? "#22c55e" : "#238636",
                border: "none",
                borderRadius: "4px",
                color: "#fff",
                fontSize: "0.75rem",
                cursor: "pointer",
              }}
            >
              {copied ? "已复制!" : "复制"}
            </button>
          </div>
          <pre
            style={{
              margin: 0,
              padding: "0.5rem",
              background: "#161b22",
              borderRadius: "4px",
              fontSize: "0.65rem",
              color: "#c9d1d9",
              overflow: "auto",
              maxHeight: "250px",
              fontFamily: "monospace",
              whiteSpace: "pre-wrap",
              wordBreak: "break-all",
            }}
          >
            {exportCode}
          </pre>
        </div>
      )}

      {/* 使用说明 */}
      <div
        style={{
          marginTop: "auto",
          padding: "1rem",
          background: "#1e293b",
          borderRadius: "8px",
        }}
      >
        <h4
          style={{
            color: "#f1f5f9",
            fontSize: "0.85rem",
            marginBottom: "0.5rem",
          }}
        >
          使用说明
        </h4>
        <ol
          style={{
            color: "#94a3b8",
            fontSize: "0.75rem",
            margin: 0,
            paddingLeft: "1.2rem",
            lineHeight: 1.6,
          }}
        >
          <li>选择模板或从空模板开始</li>
          <li>点击格子切换选择状态</li>
          <li>必须恰好选择 60 个格子</li>
          <li>输入名称并保存到后端</li>
          <li>在游戏中选择该棋盘进行对战</li>
        </ol>
      </div>
    </div>
  );
};

export default Sidebar;
