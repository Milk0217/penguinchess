// 棋子类
class Piece {
    constructor(id) {
        this.id = id;
        this.element = null; // 自身所指向的棋子div
        this.hex = null; // 当前所在的六边形
    }

    // 放置棋子到指定的六边形
    placeToHex(hex) {
      // 定义所在hex
 
      // 删除块的点击事件
      hex.removeClickHandler();
      hex.updateStatus(0)
      this.hex = hex

      // 清除之前的棋子（如果有）
      const existingPiece = document.querySelector(`.piece[data-hex-id="${hex.q},${hex.r},${hex.s}"]`);
      if (existingPiece) {
        existingPiece.remove();
      }

      //从 DOM 元素的 dataset 中获取中心位置
      const centerX = parseFloat(hex.centerX);
      const centerY = parseFloat(hex.centerY);

      // 4. 创建棋子元素
      const piece = document.createElement('div');
      piece.className = 'piece';
      piece.textContent = this.id;
      piece.style.width = '40px'; // 棋子直径
      piece.style.height = '40px';
      piece.style.borderRadius = '50%'; // 圆形
      piece.style.backgroundColor = 'black'; // 棋子颜色
      piece.style.color = 'white'; // 文字颜色
      piece.style.display = 'flex';
      piece.style.justifyContent = 'center';
      piece.style.alignItems = 'center';
      piece.style.position = 'absolute';
      piece.style.left = `${hex.left}px`; // 中心对齐（减去半径）
      piece.style.top = `${hex.top}px`; // 中心对齐（减去半径）
      piece.style.zIndex = '10'; // 确保棋子在六边形之上

      this.element = piece;
      // 5. 挂载到 board 容器
      const board = document.getElementById('board');
      board.appendChild(piece);
    }

    // 立方体坐标到屏幕坐标的转换
    cubeToPixel(q, r, size = 30) {
      const x = size * (3/2 * q);
      const y = size * (Math.sqrt(3)/2 * q + Math.sqrt(3) * r);
      return { x, y };
    }

    // 移动棋子到相邻的六边形
    moveToHex(newHex) {
        // 1. 清理当前 hex 的棋子信息
        if (this.hex) {
            // 设置当前 hex 的 value 为 -1
            this.hex.updateStatus(-1);
            // 移除当前 Piece 的 DOM 元素
            this.element.style.display = 'none';
        }
        // 2. 更新棋子的 hex 引用
        this.hex = newHex;

        // 3. 更新目标 hex 的棋子信息
        if (newHex) {
            // 设置目标 hex 的 pieceValue 属性
            newHex.pieceValue = this.id;
            // 更新目标 hex 的文本内容
            newHex.textContent = this.id;
            // 高亮目标 hex 的 DOM 元素（如果需要）
            if (newHex.element) {
                newHex.element.classList.add('highlighted');
            }
        }

        // 4. 调用 placeToHex 方法更新棋子的 DOM 位置
        this.placeToHex(newHex);
    }

    destroySelf(){
        this.element.style.display = "none" ;
        this.hex.updateStatus(-1);
    }
}

// 创建棋子
export function createPiece(value) {
    return new Piece(value);
}
