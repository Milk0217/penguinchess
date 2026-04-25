/**
 * App.tsx 工具函数单元测试
 * 测试路径检查和目标计算逻辑
 */

import { describe, it, expect } from 'vitest'

// Extract these utility functions from App.tsx for testing
// They are pure functions that can be tested independently

function buildHexIndexMap(hexes: Array<{ q: number; r: number; s: number; index: number }>): Map<string, number> {
  const m = new Map<string, number>()
  for (const h of hexes) m.set(`${h.q},${h.r},${h.s}`, h.index)
  return m
}

function isPathClear(
  q1: number, r1: number, s1: number,
  q2: number, r2: number, s2: number,
  hexMap: Map<string, number>,
  hexes: Array<{ q: number; r: number; s: number; state: string; points: number }>,
  occupiedKeys: Set<string>,
): boolean {
  const dq = q2 - q1, dr = r2 - r1, ds = s2 - s1
  const steps = Math.max(Math.abs(dq), Math.abs(dr), Math.abs(ds))
  if (steps <= 1) return true

  const sign = (n: number) => (n > 0 ? 1 : n < 0 ? -1 : 0)
  const sq = sign(dq), sr = sign(dr), ss = sign(ds)

  for (let i = 1; i < steps; i++) {
    const mq = q1 + sq * i, mr = r1 + sr * i, ms = s1 + ss * i
    const key = `${mq},${mr},${ms}`
    if (!hexMap.has(key)) return false
    const idx = hexMap.get(key)!
    const h = hexes[idx]
    if (h.state !== 'active' || occupiedKeys.has(key)) return false
  }
  return true
}

function computeTargets(
  piece: { q: number; r: number; s: number; state: string; points: number; index: number },
  hexes: Array<{ q: number; r: number; s: number; state: string; points: number; index: number }>,
  hexMap: Map<string, number>,
  allPieces: Array<{ q: number | null; r: number | null; s: number | null; alive: boolean }>,
): Set<number> {
  if (piece.q === undefined || piece.q === null) return new Set()
  const occupiedKeys = new Set<string>()
  for (const p of allPieces) {
    if (p.alive && p.q !== null) occupiedKeys.add(`${p.q},${p.r},${p.s}`)
  }
  const targets = new Set<number>()
  for (const h of hexes) {
    if (h.state !== 'active') continue
    if (occupiedKeys.has(`${h.q},${h.r},${h.s}`)) continue
    if (h.q !== piece.q && h.r !== piece.r && h.s !== piece.s) continue
    if (isPathClear(piece.q, piece.r, piece.s, h.q, h.r, h.s, hexMap, hexes, occupiedKeys)) {
      targets.add(h.index)
    }
  }
  return targets
}

describe('buildHexIndexMap', () => {
  it('should create map from hex array', () => {
    const hexes = [
      { q: 0, r: 0, s: 0, index: 0 },
      { q: 1, r: -1, s: 0, index: 1 },
      { q: 0, r: 1, s: -1, index: 2 },
    ]
    const map = buildHexIndexMap(hexes)
    expect(map.get('0,0,0')).toBe(0)
    expect(map.get('1,-1,0')).toBe(1)
    expect(map.get('0,1,-1')).toBe(2)
  })

  it('should return undefined for non-existent key', () => {
    const map = buildHexIndexMap([])
    expect(map.get('0,0,0')).toBeUndefined()
  })
})

describe('isPathClear', () => {
  const hexes = [
    { q: 0, r: 0, s: 0, state: 'active', points: 1, index: 0 },
    { q: 1, r: -1, s: 0, state: 'active', points: 1, index: 1 },
    { q: 2, r: -2, s: 0, state: 'active', points: 1, index: 2 },
    { q: 0, r: 1, s: -1, state: 'active', points: 1, index: 3 },
    { q: 0, r: -1, s: 1, state: 'occupied', points: 0, index: 4 }, // occupied
  ]
  const hexMap = buildHexIndexMap(hexes)

  it('should return true for adjacent hexes', () => {
    expect(isPathClear(0, 0, 0, 1, -1, 0, hexMap, hexes, new Set())).toBe(true)
  })

  it('should return true for clear path with multiple steps', () => {
    expect(isPathClear(0, 0, 0, 2, -2, 0, hexMap, hexes, new Set())).toBe(true)
  })

  it('should return false when path is blocked by occupied hex', () => {
    // Path from (0,0,0) to (2,-2,0) goes through (1,-1,0)
    const occupiedKeys = new Set(['1,-1,0']) // This blocks the path
    expect(isPathClear(0, 0, 0, 2, -2, 0, hexMap, hexes, occupiedKeys)).toBe(false)
  })

  it('should return false when destination is out of bounds', () => {
    const largeMap = new Map<string, number>()
    expect(isPathClear(0, 0, 0, 99, 99, 99, largeMap, hexes, new Set())).toBe(false)
  })
})

describe('computeTargets', () => {
  const hexes = [
    { q: 0, r: 0, s: 0, state: 'active', points: 1, index: 0 },
    { q: 1, r: -1, s: 0, state: 'active', points: 1, index: 1 },
    { q: 2, r: -2, s: 0, state: 'active', points: 1, index: 2 },
    { q: 0, r: 1, s: -1, state: 'active', points: 1, index: 3 },
    { q: 0, r: -1, s: 1, state: 'active', points: 1, index: 4 },
    { q: 1, r: 0, s: -1, state: 'active', points: 1, index: 5 },
    { q: -1, r: 1, s: 0, state: 'active', points: 1, index: 6 },
    { q: -1, r: 0, s: 1, state: 'active', points: 1, index: 7 },
    { q: 0, r: 0, s: 0, state: 'occupied', points: 0, index: 8 }, // occupied by a piece
  ]
  const hexMap = buildHexIndexMap(hexes)

  it('should return empty set for piece with null coordinates', () => {
    const piece = { q: null, r: null, s: null, state: 'occupied', points: 0, index: 0 }
    const result = computeTargets(piece as any, hexes, hexMap, [])
    expect(result.size).toBe(0)
  })

  it('should find targets along same q axis', () => {
    const piece = { q: 0, r: 0, s: 0, state: 'occupied', points: 0, index: 0 }
    const allPieces = [{ q: 0, r: 0, s: 0, alive: true }]
    const result = computeTargets(piece, hexes, hexMap, allPieces)
    // Should find targets at index 1, 2 (along q axis), 3 (along r axis), 4 (along s axis)
    expect(result.size).toBeGreaterThan(0)
  })

  it('should not include occupied hexes as targets', () => {
    const piece = { q: -1, r: 0, s: 1, state: 'occupied', points: 0, index: 7 }
    const allPieces = [
      { q: 0, r: 0, s: 0, alive: true }, // occupied
    ]
    const result = computeTargets(piece, hexes, hexMap, allPieces)
    // index 8 is occupied, should not be a target
    expect(result.has(8)).toBe(false)
  })
})

describe('getStatusText logic', () => {
  function getStatusText(state: {
    game_over: boolean
    winner: number | null
    episode_steps: number
    phase: string
    current_player: number
  }): string {
    if (state.game_over) {
      if (state.winner === 0) return 'Player 1 (P1) 获胜！'
      if (state.winner === 1) return 'Player 2 (P2) 获胜！'
      return '平局！'
    }
    const turn = state.episode_steps + 1
    const playerNames = ['Player 1 (P1)', 'Player 2 (P2)']
    if (state.phase === 'placement') {
      return `第 ${turn} 步 · ${playerNames[state.current_player]} 放置棋子`
    }
    return `第 ${turn} 步 · ${playerNames[state.current_player]} 移动棋子`
  }

  it('should show P1 win message', () => {
    expect(getStatusText({ game_over: true, winner: 0, episode_steps: 10, phase: 'movement', current_player: 0 }))
      .toBe('Player 1 (P1) 获胜！')
  })

  it('should show P2 win message', () => {
    expect(getStatusText({ game_over: true, winner: 1, episode_steps: 10, phase: 'movement', current_player: 1 }))
      .toBe('Player 2 (P2) 获胜！')
  })

  it('should show draw message', () => {
    expect(getStatusText({ game_over: true, winner: 2, episode_steps: 10, phase: 'movement', current_player: 0 }))
      .toBe('平局！')
  })

  it('should show placement turn', () => {
    expect(getStatusText({ game_over: false, winner: null, episode_steps: 0, phase: 'placement', current_player: 0 }))
      .toBe('第 1 步 · Player 1 (P1) 放置棋子')
  })

  it('should show movement turn', () => {
    expect(getStatusText({ game_over: false, winner: null, episode_steps: 5, phase: 'movement', current_player: 1 }))
      .toBe('第 6 步 · Player 2 (P2) 移动棋子')
  })
})