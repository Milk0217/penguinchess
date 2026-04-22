/**
 * 企鹅棋主应用。
 *
 * 架构原则：
 * - 所有游戏逻辑在后端（PenguinChessCore）执行
 * - 前端通过 API 与后端交互，只负责渲染和用户交互
 * - 每次玩家动作后，重新获取完整状态（state()），前端重渲染
 *
 * 人类对战流程（双方都在同一浏览器）：
 *   放置阶段: 点击合法空格 → 放置己方棋子 → 自动切换玩家
 *   移动阶段: 点击己方棋子 → 高亮可移动目标 → 点击目标格子 → 移动棋子
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import Board from "./Board";
import { api, type GameState, type HexData } from "./api";

const PLAYER_NAMES = ["Player 1 (P1)", "Player 2 (P2)"];

// -------------------------------------------------------------------------
// 路径检查工具（与 PenguinChessCore._path_clear / JS calculatePossibleMoves 对齐）
// -------------------------------------------------------------------------

/** 根据 hex index 建立 (q,r,s) → index 映射 */
function buildHexIndexMap(hexes: HexData[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const h of hexes) m.set(`${h.q},${h.r},${h.s}`, h.index);
  return m;
}

/** 检查 (q1,r1,s1) 到 (q2,r2,s2) 的直线路径是否畅通 */
function isPathClear(
  q1: number, r1: number, s1: number,
  q2: number, r2: number, s2: number,
  hexMap: Map<string, number>,
  hexes: HexData[],
  occupiedKeys: Set<string>,
): boolean {
  const dq = q2 - q1, dr = r2 - r1, ds = s2 - s1;
  const steps = Math.max(Math.abs(dq), Math.abs(dr), Math.abs(ds));
  if (steps <= 1) return true;

  const sign = (n: number) => (n > 0 ? 1 : n < 0 ? -1 : 0);
  const sq = sign(dq), sr = sign(dr), ss = sign(ds);

  for (let i = 1; i < steps; i++) {
    const mq = q1 + sq * i, mr = r1 + sr * i, ms = s1 + ss * i;
    const key = `${mq},${mr},${ms}`;
    if (!hexMap.has(key)) return false;           // 棋盘外
    const idx = hexMap.get(key)!;
    const h = hexes[idx];
    if (h.value <= 0 || occupiedKeys.has(key)) return false; // 被占据/消除
  }
  return true;
}

/** 计算某棋子的所有合法移动目标格子索引 */
function computeTargets(
  piece: HexData & { owner: number, alive: boolean },
  hexes: HexData[],
  hexMap: Map<string, number>,
  allPieces: { q: number | null; r: number | null; s: number | null; alive: boolean }[],
): Set<number> {
  if (piece.q === null) return new Set();
  const occupiedKeys = new Set<string>();
  for (const p of allPieces) {
    if (p.alive && p.q !== null) occupiedKeys.add(`${p.q},${p.r},${p.s}`);
  }
  const targets = new Set<number>();
  for (const h of hexes) {
    if (h.value <= 0) continue;
    if (occupiedKeys.has(`${h.q},${h.r},${h.s}`)) continue;
    if (h.q !== piece.q && h.r !== piece.r && h.s !== piece.s) continue;
    if (isPathClear(piece.q, piece.r, piece.s, h.q, h.r, h.s, hexMap, hexes, occupiedKeys)) {
      targets.add(h.index);
    }
  }
  return targets;
}

function getStatusText(state: GameState): string {
  if (state.game_over) {
    if (state.winner === 0) return `${PLAYER_NAMES[0]} 获胜！`;
    if (state.winner === 1) return `${PLAYER_NAMES[1]} 获胜！`;
    return "平局！";
  }
  const turn = state.episode_steps + 1;
  if (state.phase === "placement") {
    return `第 ${turn} 步 · ${PLAYER_NAMES[state.current_player]} 放置棋子`;
  }
  return `第 ${turn} 步 · ${PLAYER_NAMES[state.current_player]} 移动棋子`;
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 移动阶段：选中棋子的可移动目标格子 index 集合
  const [selectedPieceId, setSelectedPieceId] = useState<number | null>(null);
  const [targetIndices, setTargetIndices] = useState<Set<number>>(new Set());

  // 当选中棋子变化时，重新计算可移动目标
  const targetIndicesFromSelected = useMemo(() => {
    if (!state || state.phase !== "movement" || selectedPieceId === null) {
      return new Set<number>();
    }
    const hexMap = buildHexIndexMap(state.hexes);
    const piece = state.pieces.find((p) => p.id === selectedPieceId);
    if (!piece) return new Set();
    return computeTargets(
      { ...piece, q: piece.q!, r: piece.r!, s: piece.s!, value: 0 } as any,
      state.hexes,
      hexMap,
      state.pieces,
    );
  }, [state, selectedPieceId]);

  // 有效目标集合 = 选中棋子后计算出的目标（优先）或合法动作集合
  const effectiveTargets = useMemo(() => {
    if (targetIndices.size > 0) return targetIndices;
    return targetIndicesFromSelected;
  }, [targetIndices, targetIndicesFromSelected]);

  // -------------------------------------------------------------------------
  // 初始化 / 重置游戏
  // -------------------------------------------------------------------------
  const initGame = useCallback(async (seed?: number) => {
    setLoading(true);
    setError(null);
    setSelectedPieceId(null);
    setTargetIndices(new Set());
    try {
      const res = await api.createGame(seed);
      setSessionId(res.state.session_id);
      setState(res.state);
    } catch (e: any) {
      setError(e.message || "创建游戏失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    initGame();
  }, [initGame]);

  // -------------------------------------------------------------------------
  // 重开（保持 seed 不变）
  // -------------------------------------------------------------------------
  const handleReset = async () => {
    if (!sessionId) return;
    setLoading(true);
    setSelectedPieceId(null);
    setTargetIndices(new Set());
    try {
      const res = await api.reset(sessionId);
      setState(res.state);
    } catch (e: any) {
      setError(e.message || "重置失败");
    } finally {
      setLoading(false);
    }
  };

  // -------------------------------------------------------------------------
  // 提交动作到后端
  // -------------------------------------------------------------------------
  const submitAction = async (hexIndex: number) => {
    if (!sessionId || !state || state.game_over) return;
    setLoading(true);
    try {
      const res = await api.action(sessionId, hexIndex);
      setState(res.state);
      setSelectedPieceId(null);
      setTargetIndices(new Set());
    } catch (e: any) {
      setError(e.message || "动作执行失败");
      setLoading(false);
    }
  };

  // -------------------------------------------------------------------------
  // 棋盘格子点击处理（由 Board 组件触发）
  // -------------------------------------------------------------------------
  const handleBoardHexClick = (hex: HexData) => {
    if (!state || state.game_over) return;

    if (state.phase === "movement") {
      // 移动阶段：检查是否点击了己方棋子
      const pieceOnHex = state.pieces.find(
        (p) =>
          p.alive &&
          p.q === hex.q &&
          p.r === hex.r &&
          p.s === hex.s &&
          p.owner === state.current_player,
      );

      if (pieceOnHex) {
        // 选中 / 取消选中棋子
        if (selectedPieceId === pieceOnHex.id) {
          setSelectedPieceId(null);
          setTargetIndices(new Set());
        } else {
          setSelectedPieceId(pieceOnHex.id);
          setTargetIndices(new Set()); // Board 用 targetIndicesFromSelected 计算
        }
      } else if (selectedPieceId !== null) {
        // 有选中棋子 → 检查是否点的是合法目标
        if (targetIndicesFromSelected.has(hex.index)) {
          submitAction(hex.index);
        } else {
          // 点在非法格子 → 取消选择
          setSelectedPieceId(null);
          setTargetIndices(new Set());
        }
      }
    } else {
      // 放置阶段：直接提交
      submitAction(hex.index);
    }
  };

  if (loading && !state) {
    return (
      <div className="loading">
        <span>加载中...</span>
      </div>
    );
  }

  if (error && !state) {
    return (
      <div style={{ textAlign: "center", marginTop: "2rem", color: "#ef4444" }}>
        <p>{error}</p>
        <button className="btn btn-primary" onClick={() => initGame()}>
          重试
        </button>
      </div>
    );
  }

  if (!state) return null;

  const statusText = getStatusText(state);
  const isGameOver = state.game_over;

  return (
    <div style={{
      minHeight: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "1.5rem 1rem",
    }}>
      {/* 标题 */}
      <h1 style={{ fontSize: "1.6rem", fontWeight: 800, color: "#0f172a", margin: "0 0 0.5rem" }}>
        🐧 企鹅棋 PenguinChess
      </h1>
      <p style={{ color: "#64748b", fontSize: "0.85rem", margin: "0 0 1.5rem" }}>
        前后端分离架构 · 游戏逻辑在后端执行
      </p>

      {/* 分数板 */}
      <div className="scoreboard">
        <div>
          <span style={{ color: "#3b82f6", fontWeight: 800, fontSize: "1.1rem" }}>
            {PLAYER_NAMES[0]}
          </span>
          <div style={{ fontSize: "2rem", fontWeight: 900, color: "#1e40af" }}>
            {state.scores[0]}
          </div>
          {state.current_player === 0 && !isGameOver && (
            <span style={{ fontSize: "0.7rem", color: "#3b82f6", fontWeight: 700 }}>● 回合中</span>
          )}
        </div>
        <div className="vs" style={{ fontSize: "1.2rem", fontWeight: 700, paddingTop: "0.5rem" }}>vs</div>
        <div>
          <span style={{ color: "#ef4444", fontWeight: 800, fontSize: "1.1rem" }}>
            {PLAYER_NAMES[1]}
          </span>
          <div style={{ fontSize: "2rem", fontWeight: 900, color: "#b91c1c" }}>
            {state.scores[1]}
          </div>
          {state.current_player === 1 && !isGameOver && (
            <span style={{ fontSize: "0.7rem", color: "#ef4444", fontWeight: 700 }}>● 回合中</span>
          )}
        </div>
      </div>

      {/* 阶段标签 */}
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", margin: "0 0 0.5rem" }}>
        <span className={`phase-badge ${state.phase}`}>
          {state.phase === "placement" ? "放置阶段" : "移动阶段"}
        </span>
        {!isGameOver && (
          <span style={{ color: "#94a3b8", fontSize: "0.8rem" }}>
            合法动作: {state.legal_actions.length}
          </span>
        )}
      </div>

      {/* 状态提示 */}
      <div className={`status-bar ${isGameOver ? "game-over" : ""}`}>
        {loading ? "处理中..." : statusText}
      </div>

      {/* 棋盘 */}
      <Board
        state={state}
        selectedHexIndex={null}
        selectedPieceId={selectedPieceId}
        targetIndices={effectiveTargets}
        onHexClick={handleBoardHexClick}
      />

      {/* 操作提示 */}
      {!isGameOver && (
        <div style={{ color: "#64748b", fontSize: "0.8rem", marginTop: "0.75rem", textAlign: "center" }}>
          {state.phase === "placement"
            ? "点击空白格子放置棋子（双方轮流各放 3 个）"
            : selectedPieceId === null
              ? "点击己方棋子选中，然后点击目标格子移动"
              : "点击橙色高亮目标格子移动，或点击其他棋子切换"}
        </div>
      )}

      {/* 重开按钮 */}
      <div style={{ marginTop: "1.5rem", display: "flex", gap: "0.75rem" }}>
        <button className="btn btn-primary" onClick={handleReset} disabled={loading}>
          🔄 新游戏
        </button>
      </div>

      {/* 错误提示 */}
      {error && state && (
        <div style={{ marginTop: "0.75rem", color: "#ef4444", fontSize: "0.85rem" }}>
          ⚠️ {error}
        </div>
      )}
    </div>
  );
}
