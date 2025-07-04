// 随机生成数列
function generateSequence(totalSum = 99) {
    let sequence = [];
    let count3 = 0;
    let remainingSum = totalSum;
    let remainingLength = 60;

    // 先随机确定3的数量（8到10个）
    count3 = Math.floor(Math.random() * 3) + 8;
    
    // 分配3的值
    for (let i = 0; i < count3; i++) {
        sequence.push(3);
        remainingSum -= 3;
        remainingLength--;
    }

    // 剩余的位置用1和2填充
    while (remainingLength > 0) {
        // 随机决定下一个是1还是2
        let nextNum = Math.random() < 0.5 ? 1 : 2;
        
        // 检查是否还能放入这个数
        if (remainingSum - nextNum >= 0 && remainingLength - 1 >= 0) {
            sequence.push(nextNum);
            remainingSum -= nextNum;
            remainingLength--;
        } else if (remainingSum - 2 >= 0 && remainingLength - 1 >= 0) {
            sequence.push(2);
            remainingSum -= 2;
            remainingLength--;
        } else {
            // 如果都放不下，说明前面分配有问题，重新开始
            return generateSequence(totalSum);
        }
    }

    // 检查总和是否正确
    if (sequence.reduce((a, b) => a + b, 0) !== totalSum) {
        return generateSequence(totalSum);
    }

    // 打乱数组顺序（Fisher-Yates洗牌算法）
    for (let i = sequence.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [sequence[i], sequence[j]] = [sequence[j], sequence[i]];
    }

    return sequence;
}

class Hex {
  constructor(q, r, s, value, showCoords, centerX, centerY, radius) {
    this.q = q;
    this.r = r;
    this.s = s;
    this.value = value;
    this.showCoords = showCoords;
    this.centerX = centerX;
    this.centerY = centerY;
    this.left = 0;
    this.top = 0;
    this.radius = radius;
    this.element = this.createHexElement();
    this.setupClickHandler(); // 初始化点击事件
  }

  createHexElement() {
    const hex = document.createElement('div');
    hex.classList.add('hex');

    // 存储立方体坐标和值
    hex.dataset.q = this.q;
    hex.dataset.r = this.r;
    hex.dataset.s = this.s;
    hex.dataset.value = this.value;

    // 调整后的像素坐标
    const { x, y } = this.cubeToPixel(this.q, this.r);
    hex.style.left = `${this.centerX + x - 25}px`;
    hex.style.top = `${this.centerY + y - 25}px`;

    this.left = `${this.centerX + x - 20}`;
    this.top = `${this.centerY + y - 20}`;

    // 设置基于坐标的颜色
    const hue = ((this.q + this.radius) / (2 * this.radius)) * 360;
    hex.style.backgroundColor = `hsl(${hue}, 70%, 80%)`;
    hex.style.zIndex = '5';

    return hex; // 不再直接绑定 click 事件，由 setupClickHandler 管理
  }

  // 初始化点击事件
  setupClickHandler() {
    const clickHandler = () => this.selectHex(this.element);
    this.element.addEventListener('click', clickHandler);
    this.clickHandler = clickHandler; // 保存引用以便移除
  }

  // 移除点击事件
  removeClickHandler() {
    if (this.clickHandler) {
      this.element.removeEventListener('click', this.clickHandler);
      this.clickHandler = null;
      console.log('Click handler removed');
    } else {
      console.log('No click handler to remove');
    }
  }

  // 选中六边形
  selectHex(hex) {
    // 清除所有选中状态
    document.querySelectorAll('.hex').forEach(h => {
      h.classList.remove('selected');
    });
    
    // 选中当前六边形
    hex.classList.add('selected');
    
    // 获取立方体坐标和值
    const { q, r, s, value } = hex.dataset;
    
    // 显示选中信息
    document.getElementById('selected-info').textContent = 
      `选中六边形: (${q}, ${r}, ${s}), 值: ${value}`;
  }

  // 立方体坐标到屏幕坐标的转换
  cubeToPixel(q, r, size = 30) {
    const x = size * (3/2 * q);
    const y = size * (Math.sqrt(3)/2 * q + Math.sqrt(3) * r);
    return { x, y };
  }

  appendToBoard(board) {
    board.appendChild(this.element);
    this.updateText();
  }

  updateText() {
    this.element.textContent = this.showCoords
      ? `${this.q},${this.r},${this.s}`
      : `${this.value}`;
  }
}

// 创建偏矩形棋盘
export function createBoard(radius = 8) {
  const board = document.getElementById('board');
  if (!board) return; // 增加错误处理

  board.innerHTML = '';

  const totalValue = 99; // 总分
  const valueSequence = generateSequence(totalValue); // 生成值数组
  let valueIndex = 0; // 值数组的索引

  // 预计算中心点坐标
  const centerX = board.clientWidth / 2;
  const centerY = board.clientHeight / 2;

  // 预计算qAdjustments
  const qAdjustments = {
    "1": -1,
    "2": -1,
    "3": -2,
    "-2": 1,
    "-3": 1,
    "-4": 2
  };

  // 预计算行范围
  const rowRanges = {
    even: { start: -4, end: 3 },
    odd: { start: -3, end: 3 }
  };

  const hexes = []; // 存储所有生成的 Hex 实例

  for (let q = -radius; q <= radius; q++) {
    // 跳过 q < -4 或 q > 3 的格子
    if (q < -4 || q > 3) continue;

    // 根据行号(q)的奇偶性确定列数(r)的范围
    const isEvenRow = q % 2 === 0;
    const { start, end } = isEvenRow ? rowRanges.even : rowRanges.odd;

    for (let r = start; r <= end; r++) {
      const s = -q - r;

      // 从生成的值数组中获取值
      const value = valueSequence[valueIndex++];
      if (valueIndex >= valueSequence.length) valueIndex = 0; // 防止越界

      // 只创建满足x + y + z = 0的六边形
      if (Math.abs(s) <= radius) {
        // 获取调整量，默认为 0（即不调整）
        const adjustment = qAdjustments[q] || 0;

        // 计算 adjustedR
        const adjustedR = r + adjustment;

        // 创建 Hex 实例并添加到棋盘
        const hex = new Hex(q, adjustedR, s, value, false, centerX, centerY, radius);
        hex.appendToBoard(board);
        hexes.push(hex); // 将 Hex 实例存入数组
      }
    }
  }
  return hexes; // 返回所有生成的 Hex 实例
}

// 切换坐标显示
let showCoords = false;
document.getElementById('toggle-coords-btn').addEventListener('click', () => {
    showCoords = !showCoords;
    document.querySelectorAll('.hex').forEach(hex => {
        hex.textContent = showCoords 
            ? `${hex.dataset.q},${hex.dataset.r},${hex.dataset.s}`
            : `${hex.dataset.value}`;
    });
});
