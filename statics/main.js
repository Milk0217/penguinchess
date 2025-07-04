import { createBoard } from './board.js';
import { createPiece } from './piece.js';

// 放置棋子
function placePieces() {
    // 添加开始选点前的提示
    updateGameStatus('玩家1请选择三个点放置棋子')
    let selectedCount = 0; // 记录当前玩家已选择的点数
    let currentPlayer = 1; // 当前玩家（1或2）
    let piecesPlaced = 0; // 已放置的棋子总数
    
    // 监听六边形的点击事件
    document.querySelectorAll('.hex').forEach(hex => {
        hex.addEventListener('click', function() {
            // 如果已放置6个棋子，结束放置
            if (piecesPlaced >= 6) {
                return;
            }
            
            // 如果六边形已有棋子，且是当前玩家的棋子，则选中该棋子
            if (this.dataset.player && parseInt(this.dataset.player) === currentPlayer) {
                selectedPiece = this.dataset.pieceValue;
                this.classList.add('selected');
                return;
            }
            
            // 如果六边形已有棋子且不是当前玩家的，则忽略
            if (this.dataset.player) {
                return;
            }
            
            // 更新已选择的点数和已放置的棋子总数
            selectedCount++;
            piecesPlaced++;
             
            // 创建棋子并放置
            const pieceId = currentPlayer === 1 
              ? 4 + (Math.floor((piecesPlaced - 1) / 2)) * 2  // 玩家1: 4,6,8
              : 5 + (Math.floor((piecesPlaced - 1) / 2)) * 2; // 玩家2: 5,7,9
            const piece = createPiece(pieceId);
            piece.placeToHex(this);
            
            // 添加选中样式
            this.classList.add('selected');
            
            // 标记该六边形属于当前玩家
            this.dataset.player = currentPlayer;
           
            // 如果当前切换到对手
            if (selectedCount % 2 === 1) {
                updateGameStatus('');
                currentPlayer = currentPlayer === 1 ? 2 : 1;
                selectedCount = 0
                
                // 如果所有棋子都已放置，结束
                if (piecesPlaced >= 6) {
                    return;
                }
                
                // 为下一个玩家添加提示
                updateGameStatus(`玩家${currentPlayer}请选择三个点放置棋子`)
            }
        });
    });
}

//移动棋子
function turn(turnNumber) {
    
}

// 检查游戏是否结束
function gameovercheck(){
    
    return true
}

// 游戏结束结算
function aftergame(){

}

// 初始化棋盘和棋子
function initializeGame() {
    createBoard();
    placePieces();
    
    let turnNumber = 1;
    while (true) {
        try {
            turn(turnNumber);
            if (gameovercheck()) {
                break;
            }
            turnNumber++;
        } catch (error) {
            console.error("An error occurred during the game:", error);
            break;
        }
    }
    aftergame();
}

// 初始化棋盘
window.addEventListener('DOMContentLoaded', () => {
    initializeGame();
});

// 重置棋盘
document.getElementById('reset-btn').addEventListener('click', () => {
    initializeGame();
    document.getElementById('selected-info').textContent = '当前未选中任何六边形。';
});


/**
 * 更新游戏状态显示
 * @param {string} message - 要显示的游戏状态消息
 */
function updateGameStatus(message) {
  const gameStatusElement = document.getElementById('game-status');
  if (gameStatusElement) {
    gameStatusElement.textContent = message;
  } else {
    console.error('游戏状态元素未找到');
  }
}
