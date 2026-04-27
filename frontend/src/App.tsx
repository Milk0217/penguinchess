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
import { api, type GameState, type HexData, type ModelInfo } from "./api";
import BoardContainer from "./board/BoardContainer";
import { getLayout } from "./board/layouts";
import { getTheme, getAllThemes } from "./board/themes";
import type { BoardLayout, HexCoord, LayoutConfig, PixelCoord, Bounds } from "./board/types";
import BoardEditor from "./editor/BoardEditor";
import TrainingDashboard from "./training/TrainingDashboard";

const PLAYER_NAMES = ["Player 1 (P1)", "Player 2 (P2)"];

// 应用模式
type AppMode = "game" | "editor";

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
    if (h.state !== 'active' || occupiedKeys.has(key)) return false; // 非活跃/已占据
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
    if (h.state !== 'active') continue;
    if (occupiedKeys.has(`${h.q},${h.r},${h.s}`)) continue;
    if (h.q !== piece.q && h.r !== piece.r && h.s !== piece.s) continue;
    if (isPathClear(piece.q, piece.r, piece.s, h.q, h.r, h.s, hexMap, hexes, occupiedKeys)) {
      targets.add(h.index);
    }
  }
  return targets;
}

/** 根据 hex 数据集动态创建 BoardLayout（用于自定义棋盘） */
function createLayoutFromHexes(hexes: HexCoord[], boardId: string, boardName: string): BoardLayout {
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

  return {
    id: boardId,
    name: boardName,
    description: `Custom board: ${boardName}`,
    generateHexes: () => [...hexes],
    cubeToPixel,
    getBounds,
  };
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

// =============================================================================
// Error Boundary — 捕获 React 渲染错误
// =============================================================================
import React from "react";

interface ErrorBoundaryProps { children: React.ReactNode }
interface ErrorBoundaryState { hasError: boolean; error: Error | null }
class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "#020617", color: "#f1f5f9", padding: "2rem" }}>
          <h2 style={{ color: "#f87171" }}>应用出错</h2>
          <pre style={{ color: "#94a3b8", fontSize: "0.8rem", maxWidth: "600px", overflow: "auto" }}>{this.state.error?.message}</pre>
          <button onClick={() => { this.setState({ hasError: false }); window.location.reload(); }}
            style={{ marginTop: "1rem", padding: "0.5rem 1rem", background: "#3b82f6", color: "white", border: "none", borderRadius: "6px", cursor: "pointer" }}>
            重新加载
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// =============================================================================
// Debug 模式开关（通过 URL 参数 ?debug=1 开启）
// =============================================================================
function useDebugMode(): boolean {
  const [debug] = React.useState(() => {
    if (typeof window === "undefined") return false;
    const params = new URLSearchParams(window.location.search);
    return params.get("debug") === "1";
  });
  return debug;
}

export default function App() {
// 全局主题
type GlobalTheme = "dark" | "light";

const GLOBAL_THEMES: Record<GlobalTheme, {
  bg: string;
  text: string;
  textMuted: string;
  cardBg: string;
  cardBorder: string;
  accent: string;
}> = {
  dark: {
    bg: "#020617",
    text: "#f1f5f9",
    textMuted: "#94a3b8",
    cardBg: "#0f172a",
    cardBorder: "#1e293b",
    accent: "#3b82f6",
  },
  light: {
    bg: "#f8fafc",
    text: "#1e293b",
    textMuted: "#64748b",
    cardBg: "#ffffff",
    cardBorder: "#e2e8f0",
    accent: "#2563eb",
  },
};
  const [mode, setMode] = useState<AppMode>("game");

  // 游戏相关状态
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [opponent, setOpponent] = useState<string>("human");

  // 主题与布局
  const [currentThemeId, setCurrentThemeId] = useState("dark");
  const [currentLayoutId, setCurrentLayoutId] = useState("default");
  const [globalTheme, setGlobalTheme] = useState<GlobalTheme>("dark");
  const pageTheme = GLOBAL_THEMES[globalTheme];
  const debugMode = useDebugMode();
  const currentTheme = useMemo(() => getTheme(currentThemeId), [currentThemeId]);

  // AI 模型信息
  const [bestModelInfo, setBestModelInfo] = useState<ModelInfo | null>(null);
  const [rankings, setRankings] = useState<ModelInfo[] | null>(null);
  const [showRankings, setShowRankings] = useState(false);
  const [showTraining, setShowTraining] = useState(false);

  // 自定义棋盘 layout 缓存（key: boardId）
  const [customLayouts, setCustomLayouts] = useState<Map<string, BoardLayout>>(new Map());

  // 计算当前 layout（优先从缓存获取，否则用内置的）
  const currentLayout = useMemo<BoardLayout | undefined>(() => {
    const cached = customLayouts.get(currentLayoutId);
    if (cached) return cached;
    return getLayout(currentLayoutId);
  }, [currentLayoutId, customLayouts]);

  // 棋盘选择
  const [availableBoards, setAvailableBoards] = useState<Array<{ id: string; name: string; hex_count: number }>>([]);
  const [selectedBoardId, setSelectedBoardId] = useState<string>("default");

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
    if (!piece || piece.q === null || piece.r === null || piece.s === null) return new Set<number>();
    return computeTargets(
      { ...piece, q: piece.q, r: piece.r, s: piece.s, state: 'occupied', points: 0, index: piece.index ?? 0 },
      state.hexes,
      hexMap,
      state.pieces,
    );
  }, [state, selectedPieceId]);

  // 有效目标集合 = 选中棋子后计算出的目标（优先）或合法动作集合
  const effectiveTargets = useMemo<Set<number>>(() => {
    if (targetIndices.size > 0) return targetIndices;
    return targetIndicesFromSelected;
  }, [targetIndices, targetIndicesFromSelected]);

  // -------------------------------------------------------------------------
  // 初始化 / 重置游戏
  // -------------------------------------------------------------------------
  const initGame = useCallback(async (seed?: number, boardId?: string) => {
    setLoading(true);
    setError(null);
    setSelectedPieceId(null);
    setTargetIndices(new Set());
    try {
      const res = await api.createGame({ seed, board_id: boardId ?? selectedBoardId, opponent });
      setSessionId(res.state.session_id);
      setState(res.state);

      // 如果是自定义棋盘且尚未注册，则动态创建 layout
      const actualBoardId = boardId ?? selectedBoardId;
      const builtIn = getLayout(actualBoardId);
      if (!builtIn && res.state.hexes) {
        // 这是自定义棋盘，需要创建 layout
        // 后端发送原始坐标，直接使用
        const hexCoords: HexCoord[] = res.state.hexes.map(h =>
          ({ q: h.q, r: h.r, s: h.s })
        );
        const boardInfo = availableBoards.find(b => b.id === actualBoardId);
        const boardName = boardInfo?.name ?? actualBoardId;
        const customLayout = createLayoutFromHexes(hexCoords, actualBoardId, boardName);
        setCustomLayouts(prev => {
          const next = new Map(prev);
          next.set(actualBoardId, customLayout);
          return next;
        });
      }
    } catch (e: any) {
      setError(e.message || "创建游戏失败");
    } finally {
      setLoading(false);
    }
  }, [selectedBoardId, availableBoards, opponent]);

  // 加载可用棋盘列表
  useEffect(() => {
    api.getBoards().then(setAvailableBoards).catch(console.error);
  }, []);

  // AI 模式：获取最优模型信息
  useEffect(() => {
    if (opponent === "ai") {
      api.getBestModel()
        .then(setBestModelInfo)
        .catch(() => setBestModelInfo(null));
    } else {
      setBestModelInfo(null);
    }
  }, [opponent]);

  // 每次创建新游戏时刷新模型信息（AI 模式）
  useEffect(() => {
    if (opponent === "ai" && sessionId) {
      api.getBestModel()
        .then(setBestModelInfo)
        .catch(() => setBestModelInfo(null));
    }
  }, [sessionId, opponent]);

  useEffect(() => {
    initGame();
  }, [initGame]);

  // DEBUG: 打印从后端获取的棋盘数据
  useEffect(() => {
    if (state) {
      console.log('[DEBUG] Game state from backend:', {
        session_id: state.session_id,
        phase: state.phase,
        current_player: state.current_player,
        hexes_count: state.hexes.length,
        hexes_sample: state.hexes.slice(0, 5).map(h => ({
          index: h.index,
          q: h.q,
          r: h.r,
          s: h.s,
          state: h.state,
          points: h.points
        })),
        pieces: state.pieces.map(p => ({
          id: p.id,
          owner: p.owner,
          alive: p.alive,
          q: p.q,
          r: p.r,
          s: p.s
        })),
        legal_actions: state.legal_actions.slice(0, 10),
        scores: state.scores
      });
    }
  }, [state]);

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
  const submitAction = async (hexIndex: number, pieceId?: number) => {
    if (!sessionId || !state || state.game_over) return;
    setLoading(true);
    try {
      const res = await api.action(sessionId, hexIndex, pieceId);
      setState(res.state);
      setSelectedPieceId(null);
      setTargetIndices(new Set());

      // AI 对手：如果轮到 AI 则自动触发 AI 移动
      if (!res.state.game_over && opponent === "ai") {
        await triggerAIMove(res.state);
      }
    } catch (e: any) {
      setError(e.message || "动作执行失败");
    } finally {
      setLoading(false);
    }
  };

  /** 触发 AI 移动（循环直到轮到人类或游戏结束） */
  const triggerAIMove = async (currentState: GameState) => {
    if (!sessionId) return;
    let stateRef = currentState;
    while (stateRef.current_player === 1 && !stateRef.game_over && stateRef.opponent_type === "ai") {
      try {
        const aiRes = await api.aiMove(sessionId);
        stateRef = aiRes.state;
        setState(aiRes.state);
      } catch {
        break;
      }
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
          submitAction(hex.index, selectedPieceId);
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

  if (mode === "editor") {
    return <BoardEditor onBack={() => setMode("game")} />;
  }

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

  // DEBUG: 显示后端返回的棋盘数据
  const hexStates = state.hexes.reduce((acc, h) => {
    acc[h.state] = (acc[h.state] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const statusText = getStatusText(state);
  const isGameOver = state.game_over;

  return (
    <ErrorBoundary>
    <div style={{
      minHeight: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "1rem 0.5rem",
      width: "100%",
      maxWidth: "100vw",
      overflow: "hidden",
      boxSizing: "border-box",
      background: pageTheme.bg,
      color: pageTheme.text,
      transition: "all 0.2s ease",
    }}>
      {/* 标题栏 + 主题切换 */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "0.75rem",
        marginBottom: "0.25rem",
        width: "100%",
        maxWidth: "560px",
      }}>
        <h1 style={{ fontSize: "1.3rem", fontWeight: 800, color: pageTheme.text, margin: 0 }}>
          🐧 企鹅棋 PenguinChess
        </h1>
        <button
          onClick={() => {
            const next = globalTheme === "dark" ? "light" : "dark";
            setGlobalTheme(next);
            setCurrentThemeId(next === "dark" ? "dark" : "default");
          }}
          title={globalTheme === "dark" ? "切换到白天模式" : "切换到暗色模式"}
          style={{
            background: "transparent",
            border: "1px solid " + pageTheme.cardBorder,
            borderRadius: "6px",
            padding: "0.3rem 0.6rem",
            cursor: "pointer",
            fontSize: "1.1rem",
            color: pageTheme.text,
            lineHeight: 1,
          }}
        >
          {globalTheme === "dark" ? "☀️" : "🌙"}
        </button>
        {/* 对手选择 */}
        <button
          onClick={() => setOpponent(opponent === "human" ? "ai" : "human")}
          title={opponent === "human" ? "切换到 AI 对战" : "切换到人类对战"}
          style={{
            background: opponent === "ai" ? (globalTheme==="dark" ? "#1e3a5f" : "#dbeafe") : "transparent",
            border: "1px solid " + pageTheme.cardBorder,
            borderRadius: "6px",
            padding: "0.3rem 0.6rem",
            cursor: "pointer",
            fontSize: "0.85rem",
            color: pageTheme.text,
            lineHeight: 1,
            fontWeight: 600,
          }}
        >
          {opponent === "human" ? "👤 vs 👤" : "👤 vs 🤖"}
        </button>
        {/* AI 模型信息标签 */}
        {opponent === "ai" && bestModelInfo && (
          <span style={{
            fontSize: "0.65rem",
            color: pageTheme.textMuted,
            background: pageTheme.cardBg,
            border: "1px solid " + pageTheme.cardBorder,
            borderRadius: "4px",
            padding: "0.2rem 0.4rem",
            whiteSpace: "nowrap",
          }}>
            🤖 {(() => {
              const m = bestModelInfo;
              if (m.type === "ppo") return `PPO gen_${m.generation ?? "?"}`;
              const archLabel =
                m.arch === "resnet" ? " ResNet" : m.arch === "mlp" ? " MLP" : "";
              return `AZ${archLabel} iter_${m.iteration ?? "?"}`;
            })()}
            {bestModelInfo.eval?.elo != null && ` · ELO ${bestModelInfo.eval.elo}`}
          </span>
        )}
      </div>
      <p style={{ color: pageTheme.textMuted, fontSize: "0.75rem", margin: "0 0 1rem", textAlign: "center" }}>
        前后端分离架构 · 游戏逻辑在后端执行
      </p>

      {debugMode && (<>
      {/* DEBUG: 后端返回的 hex 数据 */}
      <div style={{
        background: pageTheme.cardBg,
        border: "1px solid " + pageTheme.cardBorder,
        borderRadius: "8px",
        padding: "0.75rem",
        marginBottom: "1rem",
        maxWidth: "500px",
        fontSize: "0.7rem",
        fontFamily: "monospace",
        color: pageTheme.textMuted,
      }}>
        <strong style={{ color: pageTheme.accent }}>[DEBUG] 后端返回的 hex 数据:</strong>
        <div style={{ marginTop: "0.5rem" }}>
          <div>hex 总数: <strong>{state.hexes.length}</strong></div>
          <div>hex 状态分布: <strong>{JSON.stringify(hexStates)}</strong></div>
          <div>phase: <strong>{state.phase}</strong></div>
          <div>前 5 个 hex:
            <pre style={{ margin: "0.25rem 0 0", background: globalTheme==="dark" ? "#0f172a" : "#ffffff", padding: "0.25rem", borderRadius: "4px", overflow: "auto", color: pageTheme.text }}>
              {JSON.stringify(state.hexes.slice(0, 5).map(h => ({
                i: h.index, q: h.q, r: h.r, s: h.s, st: h.state, pt: h.points
              })), null, 0)}
            </pre>
          </div>
          <div style={{ marginTop: "0.5rem" }}>
            第一个 hex 对象完整结构:
            <pre style={{ margin: "0.25rem 0 0", background: globalTheme==="dark" ? "#0f172a" : "#ffffff", padding: "0.25rem", borderRadius: "4px", overflow: "auto", color: pageTheme.text }}>
              {JSON.stringify(state.hexes[0], null, 2)}
            </pre>
          </div>
        </div>
      </div>
      </>)}

      {/* 分数板 - 响应式 */}
      <div className="scoreboard" style={{ gap: "0.5rem", padding: "0.5rem 1rem", background: pageTheme.cardBg, border: "1px solid " + pageTheme.cardBorder, borderRadius: "8px", marginBottom: "0.5rem", width: "100%", maxWidth: "560px", boxSizing: "border-box" }}>
        <div style={{ display: "flex", justifyContent: "space-around", alignItems: "center" }}>
          <div style={{ textAlign: "center" }}>
            <span style={{ color: globalTheme==="dark" ? "#60a5fa" : "#3b82f6", fontWeight: 800, fontSize: "1rem" }}>
              {PLAYER_NAMES[0]}
            </span>
            <div style={{ fontSize: "1.6rem", fontWeight: 900, color: globalTheme==="dark" ? "#93c5fd" : "#1e40af" }}>
              {state.scores[0]}
            </div>
            {state.current_player === 0 && !isGameOver && (
              <span style={{ fontSize: "0.65rem", color: globalTheme==="dark" ? "#60a5fa" : "#3b82f6", fontWeight: 700 }}>● 回合中</span>
            )}
          </div>
          <div className="vs" style={{ fontSize: "1rem", fontWeight: 700, color: pageTheme.textMuted }}>vs</div>
          <div style={{ textAlign: "center" }}>
            <span style={{ color: globalTheme==="dark" ? "#f87171" : "#ef4444", fontWeight: 800, fontSize: "1rem" }}>
              {PLAYER_NAMES[1]}
            </span>
            <div style={{ fontSize: "1.6rem", fontWeight: 900, color: globalTheme==="dark" ? "#fca5a5" : "#b91c1c" }}>
              {state.scores[1]}
            </div>
            {state.current_player === 1 && !isGameOver && (
              <span style={{ fontSize: "0.65rem", color: globalTheme==="dark" ? "#f87171" : "#ef4444", fontWeight: 700 }}>● 回合中</span>
            )}
          </div>
        </div>
      </div>

      {/* 调试信息面板 */}
      <div style={{
        width: "100%",
        maxWidth: "560px",
        marginBottom: "0.5rem",
        padding: "0.5rem 0.75rem",
        background: pageTheme.cardBg,
        border: "1px solid " + pageTheme.cardBorder,
        borderRadius: "8px",
        fontFamily: "monospace",
        fontSize: "0.68rem",
        color: pageTheme.textMuted,
      }}>
        {/* 第一行：阶段 + 总合法动作数 */}
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", marginBottom: "0.25rem" }}>
          <span style={{ fontWeight: 700, color: state.phase === "placement" ? (globalTheme==="dark"?"#60a5fa":"#1e40af") : (globalTheme==="dark"?"#f0abfc":"#9d174d") }}>
            [{state.phase === "placement" ? "放置" : "移动"}]
          </span>
          <span>总合法动作: <strong style={{ color: pageTheme.text }}>{state.legal_actions.length}</strong></span>
          <span>回合: <strong style={{ color: pageTheme.text }}>{state.episode_steps + 1}</strong></span>
          <span>活跃格子: <strong style={{ color: pageTheme.text }}>{state.hexes.filter(h => h.state === 'active').length}</strong></span>
        </div>

        {/* 第二行：棋子存活情况 */}
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", marginBottom: "0.25rem" }}>
          <span style={{ color: globalTheme==="dark" ? "#60a5fa" : "#3b82f6" }}>P1 棋子: {state.pieces.filter(p => p.owner === 0 && p.alive).length}/3</span>
          <span style={{ color: globalTheme==="dark" ? "#f87171" : "#ef4444" }}>P2 棋子: {state.pieces.filter(p => p.owner === 1 && p.alive).length}/3</span>
        </div>

        {/* AI 模型信息（非调试模式也显示） */}
        {opponent === "ai" && bestModelInfo && (
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", marginBottom: "0.25rem" }}>
            <span>AI 模型: <strong style={{ color: pageTheme.accent }}>
              {(() => {
                const m = bestModelInfo;
                if (m.type === "ppo") return `PPO gen_${m.generation ?? "?"}`;
                const archLabel =
                  m.arch === "resnet" ? " ResNet" : m.arch === "mlp" ? " MLP" : "";
                return `AlphaZero${archLabel} iter_${m.iteration ?? "?"}`;
              })()}
            </strong></span>
            {bestModelInfo.eval?.elo != null && (
              <span>ELO: <strong style={{ color: pageTheme.text }}>{bestModelInfo.eval.elo}</strong></span>
            )}
            {bestModelInfo.eval?.vs_random && (
              <span>vs 随机: <strong style={{ color: pageTheme.text }}>
                {((bestModelInfo.eval.vs_random.win) * 100).toFixed(0)}%
              </strong></span>
            )}
          </div>
        )}

        {/* 第三行：当前玩家 + 分数 */}
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", marginBottom: state.phase === "movement" || selectedPieceId !== null ? "0.25rem" : "0" }}>
          <span>当前: <strong style={{ color: state.current_player === 0 ? pageTheme.accent : (globalTheme==="dark"?"#f87171":"#ef4444") }}>
            {state.current_player === 0 ? "P1" : "P2"}
          </strong></span>
          <span>P1 分数: <strong style={{ color: globalTheme==="dark" ? "#93c5fd" : "#1e40af" }}>{state.scores[0]}</strong></span>
          <span>P2 分数: <strong style={{ color: globalTheme==="dark" ? "#fca5a5" : "#b91c1c" }}>{state.scores[1]}</strong></span>
        </div>

        {/* 移动阶段额外信息 */}
        {state.phase === "movement" && (
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
            <span>
              可移动棋子: {(() => {
                const ownPieces = state.pieces.filter(p => p.owner === state.current_player && p.alive && p.q !== null && p.r !== null && p.s !== null) as Array<HexData & { owner: number; alive: boolean; id: number }>;
                const mobile = ownPieces.filter(p => {
                  const pieceHex = state.hexes.find(h => h.q === p.q && h.r === p.r && h.s === p.s);
                  if (!pieceHex) return false;
                  const targets = computeTargets(
                    { ...p, index: pieceHex.index, state: 'occupied', points: 0 },
                    state.hexes,
                    buildHexIndexMap(state.hexes),
                    state.pieces,
                  );
                  return targets.size > 0;
                });
                return `${mobile.length}/${ownPieces.length}`;
              })()}
            </span>
            {selectedPieceId !== null && (() => {
              const piece = state.pieces.find(p => p.id === selectedPieceId);
              return (
                <>
                  <span style={{ color: "#7c3aed" }}>
                    选中: ID={selectedPieceId}{piece && piece.q !== null ? ` @ (${piece.q},${piece.r},${piece.s})` : ""}
                  </span>
                  <span style={{ color: "#7c3aed" }}>
                    目标: {effectiveTargets.size}
                  </span>
                </>
              );
            })()}
          </div>
        )}
      </div>

      {/* ===== 棋子状态面板 ===== */}
      <div style={{
        width: "100%",
        maxWidth: "560px",
        marginBottom: "0.5rem",
        padding: "0.4rem 0.6rem",
        background: pageTheme.cardBg,
        border: "1px solid " + pageTheme.cardBorder,
        borderRadius: "8px",
        fontFamily: "monospace",
        fontSize: "0.65rem",
        color: pageTheme.textMuted,
      }}>
        <div style={{ fontWeight: 700, marginBottom: "0.3rem", color: pageTheme.text }}>棋子状态</div>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: globalTheme==="dark" ? "#1e293b" : "#e2e8f0" }}>
              <th style={{ padding: "2px 6px", textAlign: "center" }}>ID</th>
              <th style={{ padding: "2px 6px", textAlign: "center" }}>归属</th>
              <th style={{ padding: "2px 6px", textAlign: "center" }}>状态</th>
              <th style={{ padding: "2px 6px", textAlign: "center" }}>坐标 (q,r,s)</th>
              <th style={{ padding: "2px 6px", textAlign: "center" }}>hex index</th>
              <th style={{ padding: "2px 6px", textAlign: "center" }}>格子值</th>
            </tr>
          </thead>
          <tbody>
            {state.pieces.map(piece => {
              const pieceHex = piece.q !== null ? state.hexes.find(h => h.q === piece.q && h.r === piece.r && h.s === piece.s) : null;
              const hexValue = pieceHex ? (pieceHex.state === 'active' ? pieceHex.points : pieceHex.state) : null;
              const isSelected = piece.id === selectedPieceId;
              return (
                <tr
                  key={piece.id}
                  style={{
                    background: isSelected
                      ? piece.owner === 0 ? "#dbeafe" : "#fee2e2"
                      : piece.owner === 0 ? "#f8fafc" : "#fff5f5",
                    fontWeight: isSelected ? 700 : 400,
                  }}
                >
                  <td style={{ padding: "2px 6px", textAlign: "center" }}>{piece.id}</td>
                  <td style={{ padding: "2px 6px", textAlign: "center", color: piece.owner === 0 ? "#3b82f6" : "#ef4444" }}>
                    {piece.owner === 0 ? "P1" : "P2"}
                  </td>
                  <td style={{
                    padding: "2px 6px",
                    textAlign: "center",
                    color: !piece.alive ? "#dc2626" : piece.q === null ? "#94a3b8" : "#16a34a",
                  }}>
                    {!piece.alive ? "已消除" : piece.q === null ? "未放置" : "存活"}
                  </td>
                  <td style={{ padding: "2px 6px", textAlign: "center" }}>
                    {piece.q !== null ? `(${piece.q},${piece.r},${piece.s})` : "—"}
                  </td>
                  <td style={{ padding: "2px 6px", textAlign: "center" }}>
                    {piece.index !== undefined && piece.index !== null ? piece.index : "—"}
                  </td>
                  <td style={{ padding: "2px 6px", textAlign: "center" }}>
                    {hexValue !== null ? hexValue : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Training Dashboard */}
      {showTraining && <TrainingDashboard />}

      {!showTraining && (
      /* 状态提示 */
      <div className={`status-bar ${isGameOver ? "game-over" : ""}`}>
        {loading ? "处理中..." : statusText}
      </div>
      )}

      {!showTraining && currentLayout && (
        <BoardContainer
          state={state}
          layout={currentLayout}
          theme={currentTheme}
          selectedPieceId={selectedPieceId}
          targetIndices={effectiveTargets}
          onHexClick={handleBoardHexClick}
        />
      )}

      {/* 操作提示 */}
      {!showTraining && !isGameOver && (
        <div style={{ color: pageTheme.textMuted, fontSize: "0.8rem", marginTop: "0.75rem", textAlign: "center" }}>
          {state.phase === "placement"
            ? "点击空白格子放置棋子（双方轮流各放 3 个）"
            : selectedPieceId === null
              ? "点击己方棋子选中，然后点击目标格子移动"
              : "点击橙色高亮目标格子移动，或点击其他棋子切换"}
        </div>
      )}

      {/* 控制按钮 - 响应式布局 */}
      <div
        style={{
          marginTop: "1.5rem",
          display: "flex",
          flexWrap: "wrap",
          gap: "0.5rem",
          justifyContent: "center",
          maxWidth: "100%",
          padding: "0 0.5rem",
        }}
      >
        <button
          className="btn btn-secondary"
          onClick={() => setMode("editor")}
          style={{ background: "#8b5cf6", color: "white", border: "none", padding: "8px 16px", borderRadius: "6px", fontSize: "14px", cursor: "pointer", fontWeight: 600 }}
        >
          棋盘编辑器
        </button>
        <button
          className="btn btn-secondary"
          onClick={() => {
            api.getModels().then(setRankings).catch(() => setRankings([]));
            setShowRankings(true);
          }}
          style={{ background: "#f59e0b", color: "white", border: "none", padding: "8px 16px", borderRadius: "6px", fontSize: "14px", cursor: "pointer", fontWeight: 600 }}
        >
          🏆 排行榜
        </button>
        <button
          className="btn btn-secondary"
          onClick={() => setShowTraining((v) => !v)}
          style={{
            background: showTraining ? "#0ea5e9" : "#0891b2",
            color: "white",
            border: "none",
            padding: "8px 16px",
            borderRadius: "6px",
            fontSize: "14px",
            cursor: "pointer",
            fontWeight: 600,
            boxShadow: showTraining ? "0 0 12px rgba(14,165,233,0.4)" : "none",
          }}
        >
          🧠 Training
        </button>
        <button className="btn btn-primary" onClick={() => initGame()} disabled={loading}
          style={{ background: pageTheme.accent, color: "white", border: "none", padding: "8px 16px", borderRadius: "6px", fontSize: "14px", cursor: loading ? "not-allowed" : "pointer", fontWeight: 600, opacity: loading ? 0.6 : 1 }}>
          新游戏
        </button>
        <button className="btn btn-secondary" onClick={handleReset} disabled={loading}
          style={{ background: globalTheme==="dark" ? "#475569" : "#64748b", color: "white", border: "none", padding: "8px 16px", borderRadius: "6px", fontSize: "14px", cursor: loading ? "not-allowed" : "pointer", fontWeight: 600, opacity: loading ? 0.6 : 1 }}>
          重开
        </button>
        <select
          value={selectedBoardId}
          onChange={(e) => {
            const newId = e.target.value;
            setSelectedBoardId(newId);
            setCurrentLayoutId(newId);
            initGame(undefined, newId);
          }}
          style={{
            padding: "6px 10px",
            fontSize: "13px",
            borderRadius: "6px",
            border: "1px solid " + pageTheme.cardBorder,
            background: pageTheme.cardBg,
            color: pageTheme.text,
            cursor: "pointer",
            minWidth: "100px",
          }}
        >
          {availableBoards.map(board => (
            <option key={board.id} value={board.id}>
              {board.name} ({board.hex_count}格)
            </option>
          ))}
        </select>
        <select
          value={currentThemeId}
          onChange={(e) => setCurrentThemeId(e.target.value)}
          style={{
            padding: "6px 10px",
            fontSize: "13px",
            borderRadius: "6px",
            border: "1px solid " + pageTheme.cardBorder,
            background: pageTheme.cardBg,
            color: pageTheme.text,
            cursor: "pointer",
            minWidth: "80px",
          }}
        >
          {getAllThemes().map(theme => (
            <option key={theme.id} value={theme.id}>
              {theme.name}
            </option>
          ))}
        </select>
      </div>

      {/* 排行榜弹窗 */}
      {showRankings && (
        <div
          onClick={() => setShowRankings(false)}
          style={{
            position: "fixed", inset: 0, zIndex: 1000,
            display: "flex", alignItems: "center", justifyContent: "center",
            background: "rgba(0,0,0,0.6)",
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: pageTheme.cardBg,
              border: "1px solid " + pageTheme.cardBorder,
              borderRadius: "12px",
              padding: "1.5rem",
              maxWidth: "520px", width: "90%",
              maxHeight: "80vh", overflow: "auto",
              fontFamily: "monospace",
              fontSize: "0.8rem",
              color: pageTheme.text,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <h2 style={{ margin: 0, fontSize: "1.1rem" }}>🏆 AI 模型排行榜</h2>
              <button onClick={() => setShowRankings(false)}
                style={{ background: "transparent", border: "none", color: pageTheme.textMuted, fontSize: "1.3rem", cursor: "pointer", padding: "0 4px" }}>
                ✕
              </button>
            </div>
            {rankings === null ? (
              <div style={{ color: pageTheme.textMuted }}>加载中...</div>
            ) : rankings.length === 0 ? (
              <div style={{ color: pageTheme.textMuted }}>暂无模型数据</div>
            ) : (
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid " + pageTheme.cardBorder, color: pageTheme.textMuted }}>
                    <th style={{ padding: "6px 8px", textAlign: "center" }}>#</th>
                    <th style={{ padding: "6px 8px", textAlign: "left" }}>模型</th>
                    <th style={{ padding: "6px 8px", textAlign: "center" }}>类型</th>
                    <th style={{ padding: "6px 8px", textAlign: "center" }}>ELO</th>
                    <th style={{ padding: "6px 8px", textAlign: "center" }}>vs随机</th>
                  </tr>
                </thead>
                <tbody>
                    // 将后端 arch 字段映射为显示标签
                    const archToLabel = (a?: string) =>
                      a === "resnet" ? "ResNet" : a === "mlp" ? "MLP" : "";

                    {rankings.sort((a, b) => (b.eval?.elo ?? 0) - (a.eval?.elo ?? 0)).map((m, i) => {
                        const isBest = i === 0;
                        const az = m.type === "alphazero";
                        const vsr = m.eval?.vs_random;
                        const winRate = vsr ? (vsr.win * 100).toFixed(0) : "—";
                        const archLabel = archToLabel(m.arch);
                        const modelLabel = az
                          ? `AZ${archLabel ? "-" + archLabel : ""}_${m.iteration ?? "?"}`
                          : `PPO_gen_${m.generation ?? "?"}`;
                        const typeLabel = az
                          ? `AZ${archLabel ? " " + archLabel : ""}`
                          : "PPO";
                        return (
                          <tr key={m.id} style={{
                            borderBottom: "1px solid " + pageTheme.cardBorder,
                            background: isBest ? (globalTheme==="dark" ? "#1e3a5f" : "#dbeafe") : "transparent",
                            fontWeight: isBest ? 700 : 400,
                          }}>
                            <td style={{ padding: "6px 8px", textAlign: "center", color: isBest ? pageTheme.accent : pageTheme.textMuted }}>
                              {isBest ? "👑" : i + 1}
                            </td>
                            <td style={{ padding: "6px 8px", textAlign: "left" }}>
                              {modelLabel}
                            </td>
                            <td style={{ padding: "6px 8px", textAlign: "center", color: az ? "#a855f7" : "#3b82f6" }}>
                              {typeLabel}
                            </td>
                        <td style={{ padding: "6px 8px", textAlign: "center", color: isBest ? pageTheme.accent : pageTheme.text }}>
                          {m.eval?.elo?.toFixed(0) ?? "—"}
                        </td>
                        <td style={{ padding: "6px 8px", textAlign: "center", color: pageTheme.textMuted }}>
                          {winRate}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
            <div style={{ marginTop: "0.75rem", color: pageTheme.textMuted, fontSize: "0.7rem", textAlign: "center" }}>
              数据来自后端 Model Registry · 点击任意处关闭
            </div>
          </div>
        </div>
      )}

      {/* 错误提示 */}
      {error && state && (
        <div style={{ marginTop: "0.75rem", color: "#ef4444", fontSize: "0.85rem" }}>
          ⚠️ {error}
        </div>
      )}
    </div>
    </ErrorBoundary>
  );
}
