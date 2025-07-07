import { createBoard } from './board.js';
import { createPiece } from './piece.js';

// 全局变量
let hexes = [];
let pieces = [];

const findHex = (hexes, q, r, s) => 
  hexes.find(hex => hex.q === q && hex.r === r && hex.s === s)
;

// 放置棋子
async function placePieces() {
    updateGameStatus('玩家1请选择三个点放置棋子');
    let selectedCount = 0;
    let currentPlayer = 1;
    let selectedPiece = null;

    // 返回一个Promise，等待用户完成放置棋子的操作
    return new Promise((resolve) => {
        const handleClick = (event) => {
            const hex = event.currentTarget;

            // 如果已放置6个棋子，结束放置
            if (pieces.length>= 6) {
                resolve(); // 解除Promise阻塞
                return;
            }

            // 如果六边形已有棋子且是当前玩家的棋子，则选中该棋子
            if (hex.dataset.player && parseInt(hex.dataset.player) === currentPlayer) {
                selectedPiece = hex.dataset.pieceValue;
                hex.classList.add('selected');
                return;
            }

            // 如果六边形已有棋子且不是当前玩家的，则忽略
            if (hex.dataset.player) {
                return;
            }

            // 更新已选择的点数和已放置的棋子总数
            selectedCount++;

            // 创建棋子并放置
            const pieceId = currentPlayer === 1 
                ? 4 + Math.floor(pieces.length / 2) * 2  // 玩家1: 4,6,8
                : 5 + Math.floor(pieces.length / 2) * 2; // 玩家2: 5,7,9
            const piece = createPiece(pieceId);

            // 找到对应的hex对象
            const targetHex = findHex(hexes, Number(hex.dataset.q), Number(hex.dataset.r), Number(hex.dataset.s));
            piece.placeToHex(targetHex);

            pieces.push(piece);
            // 添加选中样式
            hex.classList.add('selected');

            // 标记该六边形属于当前玩家
            hex.dataset.player = currentPlayer;

            // 如果当前切换到对手
            if (selectedCount % 2 === 1) {
                updateGameStatus('');
                currentPlayer = currentPlayer === 1 ? 2 : 1;
                selectedCount = 0;

                // 如果所有棋子都已放置，结束
                if (pieces.length>= 6) {
                    updateGameStatus('放置棋子结束');
                    document.getElementById('selected-info').textContent = "移动棋子阶段"
                    // 删除所有hex的初始选中功能
                    hex.classList.remove('selected');
                    hexes.forEach(hex => hex.removeClickHandler());
                    resolve(); // 解除Promise阻塞
                    return;
                }

                // 为下一个玩家添加提示
                updateGameStatus(`玩家${currentPlayer}请选择三个点放置棋子`);
            }
        };

        // 监听六边形的点击事件
        document.querySelectorAll('.hex').forEach(hex => {
            hex.addEventListener('click', handleClick);
        });
    });
}

// 移动棋子
async function turn(turnNumber) {
    try {
        // 输出当前回合信息
        updateGameStatus("第" + turnNumber + "回合\n现在是玩家"+ (turnNumber % 2 + 1) +"操作");
    } catch (error) {
        // 捕获并处理可能的错误
        console.error("An error occurred during the turn:", error);
        throw error; // 重新抛出错误，以便外层可以处理
    }
}

// 检查游戏是否结束
function gameovercheck(turnNumber){
    return turnNumber === 10;
}

// 游戏结束结算
function aftergame(){
    return true;
}

// 初始化棋盘和棋子
async function initializeGame() {
    hexes = createBoard();
    
    // 等待棋子放置完成
    await placePieces();

    let turnNumber = 1;
    while (true) {
        try {
            await turn(turnNumber);
            if (gameovercheck(turnNumber)) {
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
