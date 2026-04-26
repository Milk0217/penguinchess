/**
 * 后端 API 客户端。
 * 所有游戏状态来自后端，前端仅负责渲染。
 */

const BASE = "/api";

export interface HexData {
  index: number;   // 动作 ID = hexes 数组索引
  q: number;
  r: number;
  s: number;
  state: 'active' | 'occupied' | 'used' | 'eliminated';
  points: number;   // 1/2/3 分值（仅 active 时有效）
}

export interface PieceData {
  id: number;
  owner: number;   // 0=Player1, 1=Player2
  q: number | null;
  r: number | null;
  s: number | null;
  index: number | null;  // hexes 数组索引（放置后有值）
  alive: boolean;
}

export interface GameState {
  session_id: string;
  hexes: HexData[];
  pieces: PieceData[];
  current_player: number;   // 0 或 1
  phase: "placement" | "movement";
  scores: [number, number];  // [p1, p2]
  legal_actions: number[];   // hex indices
  game_over: boolean;
  winner: number | null;     // None/0/1/2
  last_action: {
    player: number;
    action: number;
    hex: { q: number; r: number; s: number; state: string; points: number };
    phase_before: string;
  } | null;
  episode_steps: number;
  opponent_type: string;   // "human" | "ai"
}

/** 模型评估数据 */
export interface ModelEval {
  vs_random?: { win: number; lose: number; draw: number };
  vs_prev?: { win: number; lose: number; draw: number; opponent?: string };
  elo?: number;
}

/** 模型元数据 */
export interface ModelInfo {
  id: string;
  type: string;          // "ppo" | "alphazero"
  file: string;
  generation?: number;
  iteration?: number;
  created_at: string;
  evaluated_at?: string;
  eval?: ModelEval;
}

/** 棋盘元数据 */
export interface BoardInfo {
  id: string;
  name: string;
  hex_count: number;
  created_at: string;
}

interface ApiResponse {
  state: GameState;
  reward?: number;
  invalid?: boolean;
  error?: string;
}

async function request(path: string, options?: RequestInit): Promise<any> {
  const start = performance.now();
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const elapsed = (performance.now() - start).toFixed(1);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    console.warn(`[API] ${options?.method || "GET"} ${path} ${res.status} ${elapsed}ms`, err);
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  if (options?.method === "POST") {
    console.debug(`[API] ${options?.method || "GET"} ${path} ${res.status} ${elapsed}ms`);
  }
  return res.json();
}

export const api = {
  // === 游戏 API ===

  /** 创建新游戏（可选指定棋盘 ID 和对手类型） */
  createGame(opts?: { seed?: number; board_id?: string; opponent?: string }): Promise<{ state: GameState }> {
    return request("/game", {
      method: "POST",
      body: JSON.stringify(opts || {}),
    });
  },

  /** 获取当前状态 */
  getState(sessionId: string): Promise<{ state: GameState }> {
    return request(`/game/${sessionId}`);
  },

  /** 提交动作（hex index，可选指定 piece_id） */
  action(sessionId: string, action: number, pieceId?: number): Promise<ApiResponse> {
    const body: Record<string, unknown> = { action };
    if (pieceId !== undefined) body.piece_id = pieceId;
    return request(`/game/${sessionId}/action`, {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  /** AI 执行一次移动 */
  aiMove(sessionId: string): Promise<ApiResponse> {
    return request(`/game/${sessionId}/ai_move`, { method: "POST" });
  },

  /** 重置游戏 */
  reset(sessionId: string): Promise<{ state: GameState }> {
    return request(`/game/${sessionId}/reset`, { method: "POST" });
  },

  // === 棋盘 API ===

  /** 获取所有已保存棋盘 */
  getBoards(): Promise<BoardInfo[]> {
    return request("/boards");
  },

  /** 保存棋盘 */
  saveBoard(name: string, hexes: Array<{ q: number; r: number; s: number }>): Promise<{ id: string; name: string; hex_count: number }> {
    return request("/boards", {
      method: "POST",
      body: JSON.stringify({ name, hexes }),
    });
  },

  // === 模型 API ===

  /** 获取所有可用模型及评估数据 */
  getModels(): Promise<ModelInfo[]> {
    return request("/models");
  },

  /** 获取当前最优模型信息（基于 ELO） */
  getBestModel(): Promise<ModelInfo> {
    return request("/models/best");
  },

  // === 棋盘 API ===

  /** 获取单个棋盘详情 */
  getBoard(boardId: string): Promise<{ id: string; name: string; hexes: Array<{ q: number; r: number; s: number }> }> {
    return request(`/boards/${boardId}`);
  },

  /** 删除棋盘 */
  deleteBoard(boardId: string): Promise<void> {
    return request(`/boards/${boardId}`, { method: "DELETE" });
  },
};
