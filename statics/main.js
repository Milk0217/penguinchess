import { createBoard } from './board.js';
import { createPiece } from './piece.js';
import { Player } from './player.js';

// 全局变量
let hexes = [];
let pieces = [];
let players = [];
let history = [];
const totalValue = 99; // 总分
/**
 * 放置棋子阶段，等待玩家轮流选择三个点放置棋子。
 * @returns {Promise<void>} 返回一个Promise，等待用户完成放置棋子的操作。
 */
async function placePieces() {
    let selectedCount = 0; // 记录当前玩家已选择的点数
    let currentPlayer = 1; // 当前玩家编号（1或2）
    let selectedPiece = null; // 当前选中的棋子（如果有）

    // 更新游戏状态提示当前玩家选择三个点放置棋子
    updateGameStatus(`请玩家${players[currentPlayer-1].name}选择三个点放置棋子`);
    updateScore(players); // 更新分数显示

    // 返回一个Promise，等待用户完成放置棋子的操作
    return new Promise((resolve) => {
        const handleClick = (event) => {
            const hex = event.currentTarget;

            // 如果已放置6个棋子（每个玩家3个），结束放置阶段
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
            // 给对应玩家计分,并更新
            players[currentPlayer-1].addScore(targetHex.value)
            updateScore(players)

            // 记录放置棋子的操作
            history.push({
                type: 'place',
                pieceId: pieceId,
                hex: { q: targetHex.q, r: targetHex.r, s: targetHex.s },
                player: players[currentPlayer - 1]
            });

            //放置棋子
            piece.placeToHex(targetHex);
            pieces.push(piece);

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
                updateGameStatus(`玩家${players[currentPlayer-1].name}请选择三个点放置棋子`);
            }
        };

        // 监听六边形的点击事件
        document.querySelectorAll('.hex').forEach(hex => {
            hex.addEventListener('click', handleClick);
        });
    });
}
/**
 * 执行游戏回合，处理棋子移动和回合逻辑。
 * @param {number} turnNumber - 当前回合编号。
 * @returns {Promise<boolean>} - 返回一个Promise，解析为游戏是否结束（true为结束，false为继续）。
 */
async function turn(turnNumber) {
    try {
        // 检查棋子是否需要移除
        pieces.forEach(piece => {
            const moves = calculatePossibleMoves(piece); // 计算棋子可能的移动位置
            if (moves.length === 0) {
                piece.destroySelf(); // 如果没有可移动位置，移除棋子
            }
        });
        checkAndRemoveHexes(); // 检查并移除无效的格子

        // 检查游戏是否结束
        if (gameovercheck(turnNumber)) {
            return true; // 游戏结束，返回true
        }

        const currentPlayer = turnNumber % 2 + 1; // 计算当前玩家编号（1或2）
        // 更新游戏状态显示
        updateGameStatus(`第${turnNumber+1}回合\n现在是玩家${players[currentPlayer-1].name}操作`);
        let chosenPiece = null;
        let nextHex = null;
        while (!chosenPiece) {
            // 高亮并等待当前用户选择棋子
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
            // 高亮可移动的目标格子并等待用户选择
            highlightPossibleMoves(chosenPiece, (hex) => {
                nextHex = hex;
            });
            await new Promise(resolve => setTimeout(resolve, 100)); // 简单的延迟以避免阻塞
        }

        // 记录移动棋子的操作
        history.push({
            type: 'move',
            pieceId: chosenPiece.id,
            fromHex: { q: chosenPiece.hex.q, r: chosenPiece.hex.r, s: chosenPiece.hex.s },
            toHex: { q: nextHex.q, r: nextHex.r, s: nextHex.s },
            player: players[currentPlayer - 1]
        });

        // 执行棋子移动逻辑
        players[currentPlayer-1].addScore(nextHex.value); // 玩家得分
        chosenPiece.moveToHex(nextHex); // 移动棋子到目标格子
        updateScore(players); // 更新并显示玩家得分

        // 清理高亮和事件监听器
        hexes.forEach(hex => {
            if (hex.element) {
                hex.element.classList.remove('highlighted');
                // 移除之前的点击事件监听器
                hex.element.removeEventListener('click', hex.onClick); // 移除点击事件监听器
            }
        });
        return false; // 回合结束，游戏继续

    } catch (error) {
        // 捕获并处理可能的错误
        console.error("An error occurred during the turn:",(turnNumber+1), error);
        throw error; // 重新抛出错误，以便外层可以处理
        return true; // 游戏结束（如果错误导致游戏无法继续）
    }
}
// 检查游戏是否结束
function gameovercheck(turnNumber){
    // 检查pieces中所有的piece.element的style.display是否为none，如果为none则是出局
    // 如果有一方的棋子全部出局，则游戏结束

    // 统计各方的存活棋子数量
    const aliveCounts = {0:0, 1:0};

    // 遍历所有棋子，统计存活数量
    pieces.forEach(piece => {
        if (!piece.element || !piece.element.style) {
            console.warn('棋子元素无效，跳过统计');
            return;
        }

        // 如果棋子可见（display !== 'none'），则统计其所属方
        if (piece.element.style.display !== 'none') {
            const playerSymbol = piece.id % 2; // 假设棋子有 playerSymbol 属性标识所属玩家
            if (!aliveCounts[playerSymbol]) {
                aliveCounts[playerSymbol] = 0;
            }
            aliveCounts[playerSymbol]++;
        }
    });

    // 检查是否有玩家的棋子全部出局（存活数量为0）
    let survivingPlayerSymbol = null;
    for (const playerSymbol in aliveCounts) {
        if (aliveCounts[playerSymbol] > 0) {
            if (survivingPlayerSymbol === null) {
                survivingPlayerSymbol = playerSymbol;
            } else {
                // 如果有多个玩家存活，则直接返回 false
                return false;
            }
        }
    }

    // 如果只有一个玩家存活，则处理得分和更新 hex 状态
    if (survivingPlayerSymbol !== null) {
        document.getElementById('selected-info').textContent = `玩家 ${players[survivingPlayerSymbol].name} 是唯一存活的玩家，游戏结束`;
        
        // 遍历所有 hex，更新得分和状态
        for (const hex of hexes) {
            if (hex.value > 0) {
                players[survivingPlayerSymbol].addScore(hex.value);
                updateScore(players);
                hex.updateStatus(-1);
            }
        }
        
        return true;
    }

    // 如果所有玩家都有存活棋子，游戏继续
    return false;
}
// 游戏结束结算
function aftergame(){
    console.log("aftergame")
}
// 初始化棋盘和棋子
async function initializeGame(type = 0){
    let valueSequence = []
    if (type === 0) {
        valueSequence = generateSequence(totalValue); // 生成值数组
        hexes = createBoard({valueSequence: valueSequence})
        const player1 = new Player(1, "Milky");
        const player2 = new Player(2, "test");
        players = [player1, player2];

        history.push({
            type: "initialize",
            valueSequence: valueSequence,
            players: players,
        });
    }

    // 等待棋子放置完成
    await placePieces();

    let turnNumber = 0;
    while (true) {
        try {
            const isGameOver = await turn(turnNumber);

            // 如果 turn 返回 true，表示游戏结束，退出循环
            if (isGameOver) {
                console.log("游戏结束");
                updateGameStatus("游戏结束");
                aftergame();
                break; // 退出 while 循环
            }
            
            turnNumber++;
        } catch (error) {
            console.error("An error occurred during the game:", error);
            break;
        }
    }
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
 * 更新得分信息
 * @param {Array} players - 玩家数组，包含所有玩家对象
 */
function updateScore(players) {
  const scoreElement = document.getElementById('score-board');
  if (scoreElement) {
    // 使用数组的map方法生成每个玩家的得分字符串，并用vs连接
    const scoreText = players.map(player => `${player.name}:${player.score}`).join(' vs ');
    scoreElement.textContent = scoreText;
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
  // 检查输入的棋子对象是否有效，以及是否包含 hex 属性
  if (!chosenPiece || !chosenPiece.hex) {
    console.error("Invalid chosenPiece or hex");
    return [];
  }

  // 获取当前棋子所在格子的坐标
  const currentHex = chosenPiece.hex;
  const q = currentHex.q;
  const r = currentHex.r;
  const s = currentHex.s;

  // 筛选所有格子中 q 或 r 值与当前格子相同的格子，排除当前格子本身
  let possibleMoves = hexes.filter(hex => {
    // 排除当前格子本身
    if (hex === currentHex) {
      return false;
    }
    
    // 检查 q 或 r 是否相同，或者 q + r 的和是否相同（用于六边形网格的移动规则）
    return hex.q === q || hex.r === r || (hex.q + hex.r) === (q + r);
  });

  // 排除已经被其他棋子占据的格子
  possibleMoves = possibleMoves.filter(hex => {
    return !pieces.some(piece => piece.hex === hex);
  });

  // 排除所有 value <= 0 的格子
  possibleMoves = possibleMoves.filter(hex => {
    return !hexes.some(h => h.q === hex.q && h.r === hex.r && h.value <= 0);
  });

  // 排除被其他棋子挡住行进路线的格子
  possibleMoves = possibleMoves.filter(targetHex => {
      const dq = targetHex.q - q; // 目标格子 q 坐标与当前格子的差值
      const dr = targetHex.r - r; // 目标格子 r 坐标与当前格子的差值

      const steps = Math.max(Math.abs(dq), Math.abs(dr)); // 计算移动的步数

      const signDq = Math.sign(dq); // q 方向的移动方向（1, 0, -1）
      const signDr = Math.sign(dr); // r 方向的移动方向（1, 0, -1）

      // 检查每一步的中间格子是否被阻挡
      for (let i = 1; i < steps; i++) {
          const intermediateHex = {
              q: q + signDq * i, // 计算中间格子的 q 坐标
              r: r + signDr * i, // 计算中间格子的 r 坐标
          };
      
        // 检查中间格子是否被棋子或空的格子占据
        const isBlocked = pieces.some(piece => 
            piece.hex.q === intermediateHex.q && 
            piece.hex.r === intermediateHex.r
        ) || hexes.some(hex => 
            hex.q === intermediateHex.q && 
            hex.r === intermediateHex.r && 
            hex.value < 0
        );

        if (isBlocked) {
            return false; // 如果中间格子被阻挡，则排除该目标格子
        }
      }

      return true; // 如果所有中间格子都未被阻挡，则保留该目标格子
  });

  return possibleMoves; // 返回所有可移动的格子
}
/**
 * 检查并移除不需要的格子
 * 遍历所有棋子，收集它们所在的格子、相邻格子以及相邻格子的相邻格子
 * 移除不在这些集合中的格子
 */
function checkAndRemoveHexes() {
    const connectedHexes = new Set(); // 使用 Set 避免重复添加相同的格子

    /**
     * 递归添加当前格子及其所有相邻格子到 connectedHexes 集合
     * @param {Object} hex - 当前要处理的格子
     */
    function addHexAndNeighbors(hex) {
      if (!hex) return; // 如果 hex 为空，直接返回

      // 添加当前格子到集合
      connectedHexes.add(hex);

      // 获取当前格子的所有相邻格子
      const adjacentHexes = hex.getConnectedHexes(hexes);
      
      // 递归添加相邻格子及其邻居
      adjacentHexes.forEach(neighborHex => {
        if (!connectedHexes.has(neighborHex)) { // 如果邻居还没有被添加过
          addHexAndNeighbors(neighborHex); // 递归添加
        }
      });
    }

    // 遍历所有棋子，调用 addHexAndNeighbors 函数，开始递归添加
    pieces.forEach(piece => {
      if (piece.hex) {
        addHexAndNeighbors(piece.hex); // 调用函数，开始递归添加
      }
    });

    // 遍历所有格子，移除不在 connectedHexes 中的格子
    hexes.forEach(hex => {
        if (!connectedHexes.has(hex)) {
            // 移除 DOM 元素并更新格子状态
            if (hex.element) {
                hex.updateStatus(-1); // 更新格子状态为不可用
            }
        }
    });
}
/**
 * 回放游戏历史记录，逐步重现棋盘状态和玩家操作。
 * @param {Array<Object>} history - 游戏历史记录数组，每个元素代表一个回合的操作。
 * @returns {void}
 */
function replayHistory(history) {
  let index = 0; // 当前回放操作的索引
  document.getElementById('selected-info').textContent = "回放模式"; // 更新UI显示回放状态

  // 初始化棋盘和游戏状态
  initializeGame(1, history); // 调用初始化函数，传入游戏模式和初始历史记录
  let hexes = createBoard({ valueSequence: history[0]?.valueSequence }); // 创建棋盘，使用历史记录中的初始值序列
  let players = history[0]?.players; // 获取初始玩家信息
  const player1 = new Player(players[0]?.id, players[0]?.name); // 创建玩家1对象
  const player2 = new Player(players[1]?.id, players[1]?.name); // 创建玩家2对象
  players = [player1, player2]; // 将玩家对象存入数组
  players.forEach(player => player.reset()); // 重置所有玩家的状态
  let pieces = []; // 初始化棋盘上的棋子列表

  replayNext(); // 开始执行回放

  // 执行下一个回放操作
  function replayNext() {
    if (index < history.length) { // 如果还有未回放的操作
      updateGameStatus(`第${index + 1}回合`); // 更新游戏状态显示当前回合（注意索引从0开始，显示时+1）
      const action = history[index]; // 获取当前要回放的操作
      index++; // 移动到下一个操作

      if (action.type === 'place') {
        // 处理放置棋子的操作
        const piece = createPiece(action.pieceId); // 创建棋子对象
        const targetHex = findHex(hexes, action.hex.q, action.hex.r, action.hex.s); // 找到目标格子
        players[action.player.id - 1].addScore(targetHex.value); // 玩家得分（假设玩家ID从1开始）
        piece.placeToHex(targetHex); // 将棋子放置到目标格子
        pieces.push(piece); // 将棋子添加到棋盘上的棋子列表
      } else if (action.type === 'move') {
        // 处理移动棋子的操作
        const piece = pieces.find(p => p.id === action.pieceId); // 找到要移动的棋子
        const fromHex = findHex(hexes, action.fromHex.q, action.fromHex.r, action.fromHex.s); // 找到起始格子
        const toHex = findHex(hexes, action.toHex.q, action.toHex.r, action.toHex.s); // 找到目标格子
        players[action.player.id - 1].addScore(toHex.value); // 玩家得分（移动到新格子可能得分）
        piece.moveToHex(toHex); // 将棋子移动到目标格子
        updateScore(players); // 更新并显示玩家得分
      }

      // 移除所有格子的高亮样式
      hexes.forEach(hex => {
        if (hex.element) {
          hex.element.classList.remove('highlighted');
          // 移除之前的点击事件监听器
          hex.element.removeEventListener('click', hex.onClick);
        }
      });

      // 延迟一段时间后继续回放下一个操作
      setTimeout(replayNext, 1000);
    } else {
      // 所有操作回放完毕
      document.getElementById('selected-info').textContent = "回放结束"; // 更新UI显示回放结束
    }
  }
}
/**
 * 随机生成一个数列，该数列的总和为指定值，并包含特定数量的3。
 * @param {number} totalSum - 目标数列的总和，默认为99。
 * @returns {Array<number>} 生成的随机数列。
 */
function generateSequence(totalSum = 99) {
    let sequence = []; // 存储最终生成的数列
    let count3 = 0; // 记录数字3在数列中的数量
    let remainingSum = totalSum; // 剩余需要分配的总和
    let remainingLength = 60; // 剩余需要填充的数组位置数量

    // 先随机确定3的数量（范围在8到10个之间）
    count3 = Math.floor(Math.random() * 3) + 8;
    
    // 分配确定的3的值到数列中
    for (let i = 0; i < count3; i++) {
        sequence.push(3); // 将数字3添加到数列末尾
        remainingSum -= 3; // 从剩余总和中减去3
        remainingLength--; // 减少剩余需要填充的位置计数
    }

    // 用1和2填充剩余的位置，直到数组长度达到60
    while (remainingLength > 0) {
        // 随机决定下一个要添加的数字是1还是2
        let nextNum = Math.random() < 0.5 ? 1 : 2;
        
        // 检查当前剩余总和和剩余长度是否允许放入这个随机数字
        if (remainingSum - nextNum >= 0 && remainingLength - 1 >= 0) {
            sequence.push(nextNum); // 将数字添加到数列
            remainingSum -= nextNum; // 更新剩余总和
            remainingLength--; // 更新剩余长度
        } else if (remainingSum - 2 >= 0 && remainingLength - 1 >= 0) {
            // 如果1放不下，尝试放入2
            sequence.push(2);
            remainingSum -= 2;
            remainingLength--;
        } else {
            // 如果1和2都放不下，说明之前的分配导致无法完成，需要重新生成
            return generateSequence(totalSum);
        }
    }

    // 验证生成的数列总和是否与目标总和一致
    if (sequence.reduce((a, b) => a + b, 0) !== totalSum) {
        // 如果不一致，递归调用自身重新生成
        return generateSequence(totalSum);
    }

    // 使用Fisher-Yates洗牌算法打乱数列的顺序，增加随机性
    for (let i = sequence.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1)); // 随机选择一个索引
        [sequence[i], sequence[j]] = [sequence[j], sequence[i]]; // 交换当前元素与随机选择元素的位置
    }

    return sequence; // 返回最终生成的随机数列
}
// 添加回放按钮的事件
document.getElementById('replay-btn').addEventListener('click', function(){
    replayHistory(history);
});
// 导出历史记录
document.getElementById('exportBtn').addEventListener('click', function() {
    try {
        const historyJson = JSON.stringify(history, null, 2);
        const blob = new Blob([historyJson], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'game-history.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showMessage('exportMessage', 'History exported successfully!', 'success');
    } catch (error) {
        showMessage('exportMessage', 'Error exporting history: ' + error.message, 'error');
    }
});
// 导入历史记录
document.getElementById('importBtn').addEventListener('click', function() {
    const fileInput = document.getElementById('importFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showMessage('importMessage', 'Please select a file to import', 'error');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const importedHistory = JSON.parse(e.target.result);
            history = importedHistory;
            console.log(history)
            replayHistory(history)
            showMessage('importMessage', 'History imported successfully!', 'success');
            // 这里可以添加其他处理逻辑，比如更新UI等
        } catch (error) {
            showMessage('importMessage', 'Error importing history: ' + error.message, 'error');
        }
    };
    reader.readAsText(file);
});
// 显示消息
function showMessage(elementId, message, type) {
    const messageElement = document.getElementById(elementId);
    messageElement.textContent = message;
    messageElement.className = 'message ' + type;
    messageElement.style.display = 'block';
    
    // 3秒后自动隐藏消息
    setTimeout(() => {
        messageElement.style.display = 'none';
    }, 3000);
}
// 用于在hexes中根据(q,r,s)坐标查找指定的hex
const findHex = (hexes, q, r, s) => 
  hexes.find(hex => hex.q === q && hex.r === r && hex.s === s)
;
