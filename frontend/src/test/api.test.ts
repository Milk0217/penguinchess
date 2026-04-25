/**
 * API 客户端单元测试
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { api } from '../api'

// Mock fetch
const mockFetch = vi.fn()
global.fetch = mockFetch

describe('api', () => {
  beforeEach(() => {
    mockFetch.mockReset()
  })

  describe('createGame', () => {
    it('should call /api/game with POST', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          state: {
            session_id: 'test-session',
            hexes: [],
            pieces: [],
            current_player: 0,
            phase: 'placement',
            scores: [0, 0],
            legal_actions: [],
            game_over: false,
            winner: null,
            episode_steps: 0,
          },
        }),
      })

      const result = await api.createGame()
      expect(mockFetch).toHaveBeenCalledWith('/api/game', expect.objectContaining({
        method: 'POST',
      }))
      expect(result.state.session_id).toBe('test-session')
    })

    it('should pass seed and board_id options', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ state: { session_id: 'test' } }),
      })

      await api.createGame({ seed: 42, board_id: 'parallelogram' })
      expect(mockFetch).toHaveBeenCalledWith('/api/game', expect.objectContaining({
        body: JSON.stringify({ seed: 42, board_id: 'parallelogram' }),
      }))
    })
  })

  describe('action', () => {
    it('should call /api/game/:id/action with action', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          state: {
            session_id: 'session-123',
            hexes: [],
            pieces: [],
            current_player: 0,
            phase: 'placement',
            scores: [0, 0],
            legal_actions: [],
            game_over: false,
            winner: null,
            episode_steps: 1,
          },
        }),
      })

      await api.action('session-123', 5)
      expect(mockFetch).toHaveBeenCalledWith('/api/game/session-123/action', expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ action: 5 }),
      }))
    })

    it('should include piece_id when provided', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ state: {} }),
      })

      await api.action('session-123', 10, 4)
      expect(mockFetch).toHaveBeenCalledWith('/api/game/session-123/action', expect.objectContaining({
        body: JSON.stringify({ action: 10, piece_id: 4 }),
      }))
    })
  })

  describe('reset', () => {
    it('should call /api/game/:id/reset with POST', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ state: { session_id: 'session-123' } }),
      })

      await api.reset('session-123')
      expect(mockFetch).toHaveBeenCalledWith('/api/game/session-123/reset', expect.objectContaining({
        method: 'POST',
      }))
    })
  })

  describe('getBoards', () => {
    it('should return list of boards', async () => {
      const mockBoards = [
        { id: 'parallelogram', name: 'Parallelogram', hex_count: 60, created_at: '2024-01-01' },
        { id: 'hexagon', name: 'Hexagon', hex_count: 61, created_at: '2024-01-02' },
      ]
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockBoards,
      })

      const boards = await api.getBoards()
      // Verify the URL was called (method may not be explicitly set for GET)
      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('/api/boards'), expect.any(Object))
      expect(boards).toHaveLength(2)
      expect(boards[0].id).toBe('parallelogram')
    })
  })

  describe('error handling', () => {
    it('should throw error on non-ok response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ error: 'Internal Server Error' }),
      })

      await expect(api.getBoards()).rejects.toThrow('Internal Server Error')
    })
  })
})