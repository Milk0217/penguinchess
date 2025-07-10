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

        // 检查棋子是否需要移除
        pieces.forEach(piece => {
            const moves = calculatePossibleMoves(piece);
            if (moves.length === 0) {
                piece.destroySelf();
            }
        });
        console.log(pieces)

        const currentPlayer = turnNumber % 2 + 1;
        // 输出当前回合信息
        updateGameStatus("第" + (turnNumber+1) + "回合\n现在是玩家"+ currentPlayer +"操作");
        // 高亮当前用户的棋子
        //添加一个等待框，等待用户操作
        let chosenPiece = null;
        let nextHex = null;
        // 等待用户选择棋子
        while (!chosenPiece) {
            // 这里可以添加一些逻辑来等待用户操作，例如使用事件监听器或轮询
            // 假设我们有一个事件监听器来设置 chosenPiece
            highlightCurrentPlayerPieces(currentPlayer, (piece) => {
                if(chosenPiece === null){
                    chosenPiece = piece;
                    chosenPiece.element.classList.remove("highlighted");
                    chosenPiece.element.classList.add("selected");
                } else if (chosenPiece !== piece) {
                    chosenPiece.element.classList.add("highlighted");
                    chosenPiece.element.classList.remove("selected");
                    chosenPiece = piece;
                    chosenPiece.element.classList.remove("highlighted");
                    chosenPiece.element.classList.add("selected");
                }
            });
            await new Promise(resolve => setTimeout(resolve, 100)); // 简单的延迟以避免阻塞
        }


        // 等待用户选择目标位置
        while (!nextHex) {
            // 这里可以添加一些逻辑来等待用户操作
            // 高亮可移动的目标格子并等待选择
            highlightPossibleMoves(chosenPiece, (hex) => {
                nextHex = hex;
            });
            await new Promise(resolve => setTimeout(resolve, 100)); // 简单的延迟以避免阻塞
        }
        //移动棋子
        chosenPiece.moveToHex(nextHex);

        // 移除所有格子的高亮样式
        hexes.forEach(hex => {
            if (hex.element) {
                hex.element.classList.remove('highlighted');
                // 移除之前的点击事件监听器
                hex.element.removeEventListener('click', hex.onClick);
            }
        });
    } catch (error) {
        // 捕获并处理可能的错误
        console.error("An error occurred during the turn:",(turnNumber+1), error);
        throw error; // 重新抛出错误，以便外层可以处理
    }
}

// 检查游戏是否结束
function gameovercheck(turnNumber){
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

    let turnNumber = 0;
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
/**
 * 高亮当前玩家的棋子
 * @param {number} currentPlayer - 当前玩家编号 (1 或 2)
 * @param {function} setChosenPiece - 设置当前选中的棋子的函数
 */
function highlightCurrentPlayerPieces(currentPlayer, setChosenPiece) {
    // 移除所有棋子的高亮样式
    pieces.forEach(piece => {
        if (piece.element) {
            piece.element.classList.remove('highlighted');
            // 移除之前的点击事件监听器
            piece.element.removeEventListener('click', piece.onClick);
        }
    });

    // 高亮当前玩家的棋子
    const playerPieces = pieces.filter(piece => {
        // 棋子的ID可以区分玩家（奇数是玩家2，偶数是玩家1）
        return currentPlayer === 1 ? piece.id % 2 === 0 : piece.id % 2 === 1;
    });

    playerPieces.forEach(piece => {
        if (piece.element) {
            piece.element.classList.add('highlighted');
            // 为每个棋子添加点击事件监听器
            piece.onClick = () => {
                setChosenPiece(piece);
            };
            piece.element.addEventListener('click', piece.onClick);
        }
    });
}
/**
 * 高亮棋子可移动的目标格子
 * @param {Object} chosenPiece - 当前选中的棋子对象
 * @param {function} setNextHex - 设置目标格子的函数
 */
function highlightPossibleMoves(chosenPiece, setNextHex) {
    // 移除所有格子的高亮样式
    hexes.forEach(hex => {
        if (hex.element) {
            hex.element.classList.remove('highlighted');
            // 移除之前的点击事件监听器
            hex.element.removeEventListener('click', hex.onClick);
        }
    });

    // 计算并高亮可移动的目标格子
    const possibleMoves = calculatePossibleMoves(chosenPiece); // 假设此函数计算可移动的格子
    console.log(possibleMoves)

    possibleMoves.forEach(hex => {
        if (hex.element) {
            hex.element.classList.add('highlighted');
            // 为每个格子添加点击事件监听器
            hex.onClick = () => {
                setNextHex(hex);
            };
            hex.element.addEventListener('click', hex.onClick);
        }
    });
}
/**
 * 计算棋子可移动的目标格子
 * @param {Object} chosenPiece - 当前选中的棋子对象
 * @returns {Array} 可移动的格子数组
 */
function calculatePossibleMoves(chosenPiece) {
  if (!chosenPiece || !chosenPiece.hex) {
    console.error("Invalid chosenPiece or hex");
    return [];
  }

  const currentHex = chosenPiece.hex;
  const q = currentHex.q;
  const r = currentHex.r;
  const s = currentHex.s;

  // 筛选所有格子中 q 或 r 值与当前格子相同的格子
  let possibleMoves = hexes.filter(hex => {
    // 排除当前格子本身
    if (hex === currentHex) {
      return false;
    }
    
    // 检查 q 或 r 是否相同
    return hex.q === q || hex.r === r || (hex.q + hex.r) === (q + r);
  });

  // 排除已经被其他棋子占据的格子
  possibleMoves = possibleMoves.filter(hex => {
    return !pieces.some(piece => piece.hex === hex);
  });


  // 排除所有 value = 0 的格子
  possibleMoves = possibleMoves.filter(hex => {
    return !hexes.some(h => h.q === hex.q && h.r === hex.r && h.value === 0);
  });

  // 排除被其他棋子挡住行进路线的格子
  possibleMoves = possibleMoves.filter(targetHex => {
      const dq = targetHex.q - q;
      const dr = targetHex.r - r;

      const steps = Math.max(Math.abs(dq), Math.abs(dr));

      const signDq = Math.sign(dq);
      const signDr = Math.sign(dr);

      for (let i = 1; i < steps; i++) {
          const intermediateHex = {
              q: q + signDq * i,
              r: r + signDr * i,
          };
      
        // 检查中间格子是否被棋子或 value0 的格子占据
        const isBlocked = pieces.some(piece => 
            piece.hex.q === intermediateHex.q && 
            piece.hex.r === intermediateHex.r
        ) || hexes.some(hex => 
            hex.q === intermediateHex.q && 
            hex.r === intermediateHex.r && 
            hex.value === 0
        );

        if (isBlocked) {
            return false;
        }

      }

      return true;
  });

  return possibleMoves;
}
