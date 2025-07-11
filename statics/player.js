export class Player {
  constructor(id, name='default', symbol = 0, isComputer = false) {
    this.id = id;
    this.name = name;          // 玩家名称
    this.symbol = symbol;      // 玩家棋子符号
    this.isComputer = isComputer; // 是否为电脑玩家
    this.score = 0;            // 玩家得分(本局)
    this.gamesWon = 0;         // 获胜次数
    this.gamesLost = 0;        // 失败次数
    this.gamesDrawn = 0;       // 平局次数
  }

  // 增加得分
  addScore(points = 1) {
    this.score += points;
  }

  // 记录获胜
  recordWin() {
    this.gamesWon++;
    this.addScore(1); // 默认获胜加1分
  }

  // 记录失败
  recordLoss() {
    this.gamesLost++;
  }

  // 记录平局
  recordDraw() {
    this.gamesDrawn++;
  }

  // 重置玩家数据（可选）
  reset() {
    this.score = 0;
    this.gamesWon = 0;
    this.gamesLost = 0;
    this.gamesDrawn = 0;
  }
}
