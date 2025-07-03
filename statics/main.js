import { createBoard } from './board.js';
import { createPiece } from './piece.js';

// 初始化棋盘和棋子
function initializeGame() {
    createBoard();
    const piece = createPiece(5); // 创建一个值为5的棋子
    const initialHex = document.querySelector('.hex'); // 选择第一个六边形作为初始位置
    if (initialHex) {
        piece.placeToHex(initialHex);
    }
}

// 初始化棋盘
window.addEventListener('DOMContentLoaded', () => {
    initializeGame();
});
