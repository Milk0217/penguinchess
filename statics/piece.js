// 棋子类
class Piece {
    constructor(value) {
        this.value = value;
        this.hex = null; // 当前所在的六边形
    }

    // 放置棋子到指定的六边形
    placeToHex(hex) {
        this.hex = hex;
        hex.dataset.pieceValue = this.value;
        hex.textContent = this.value;
    }

    // 移动棋子到相邻的六边形
    moveToHex(newHex) {
        if (this.hex) {
            delete this.hex.dataset.pieceValue;
            this.hex.textContent = this.hex.dataset.value;
        }
        this.placeToHex(newHex);
    }
}

// 创建棋子
export function createPiece(value) {
    return new Piece(value);
}

// 移动棋子
function movePiece(piece, targetHex) {
    if (targetHex.classList.contains('neighbors')) {
        piece.moveToHex(targetHex);
    } else {
        console.log('目标六边形不是相邻的！');
    }
}
