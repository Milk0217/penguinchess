/**
 * 游戏全局配置
 * 所有魔法数字集中管理，便于调整游戏参数
 */
export const CONFIG = {
  // ============== 棋盘 ==============
  BOARD_RADIUS: 8,
  BOARD_WIDTH: 600,
  BOARD_HEIGHT: 600,

  // ============== 分数系统 ==============
  TOTAL_VALUE: 99,         // 所有格子分值总和
  HEX_COUNT: 60,           // 格子总数
  COUNT_OF_THREE_MIN: 8,   // 数字3最少出现次数
  COUNT_OF_THREE_MAX: 10,  // 数字3最多出现次数
  THREE_VALUE: 3,          // 数字3的分值

  // ============== 玩家棋子 ==============
  PLAYER_1_PIECES: [4, 6, 8],   // 玩家1的棋子ID（偶数）
  PLAYER_2_PIECES: [5, 7, 9],   // 玩家2的棋子ID（奇数）
  PIECES_PER_PLAYER: 3,        // 每位玩家棋子数量

  // ============== 棋子尺寸 ==============
  HEX_SIZE: 30,             // 六边形网格尺寸（cubeToPixel用）
  HEX_DIAMETER: 50,        // 六边形直径(px)
  HEX_OFFSET: 25,          // 像素偏移（用于中心对齐）
  PIECE_DIAMETER: 40,      // 棋子直径(px)

  // ============== 动画与回放 ==============
  REPLAY_DELAY_MS: 1000,   // 回放每步延迟(ms)
  POLL_INTERVAL_MS: 100,    // 轮询间隔(ms)

  // ============== 服务器 ==============
  SERVER_PORT: 8080,
};
