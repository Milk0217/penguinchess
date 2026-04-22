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
  value: number;   // 1/2/3=活跃, 0=被占据, -1=已消除
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
    hex: { q: number; r: number; s: number; value: number };
    phase_before: string;
  } | null;
  episode_steps: number;
}

interface ApiResponse {
  state: GameState;
  reward?: number;
  invalid?: boolean;
  error?: string;
}

async function request(path: string, options?: RequestInit): Promise<any> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  /** 创建新游戏 */
  createGame(seed?: number): Promise<{ state: GameState }> {
    return request("/game", {
      method: "POST",
      body: JSON.stringify(seed !== undefined ? { seed } : {}),
    });
  },

  /** 获取当前状态 */
  getState(sessionId: string): Promise<{ state: GameState }> {
    return request(`/game/${sessionId}`);
  },

  /** 提交动作（hex index） */
  action(sessionId: string, action: number): Promise<ApiResponse> {
    return request(`/game/${sessionId}/action`, {
      method: "POST",
      body: JSON.stringify({ action }),
    });
  },

  /** 重置游戏 */
  reset(sessionId: string): Promise<{ state: GameState }> {
    return request(`/game/${sessionId}/reset`, { method: "POST" });
  },
};
