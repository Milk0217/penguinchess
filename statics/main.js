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
        } else {
            // 如果1放不下，尝试放2
            if (remainingSum - 2 >= 0 && remainingLength - 1 >= 0) {
                sequence.push(2);
                remainingSum -= 2;
                remainingLength--;
            } else {
                // 如果都放不下，说明前面分配有问题，重新开始
                return generateSequence(totalSum);
            }
        }
    }

    // 检查总和是否正确
    if (sequence.reduce((a, b) => a + b, 0) !== totalSum) {
        return generateSequence(totalSum);
    }

    // 打乱数组顺序
    for (let i = sequence.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [sequence[i], sequence[j]] = [sequence[j], sequence[i]];
    }

    return sequence;
}

// 立方体坐标到屏幕坐标的转换
function cubeToPixel(q, r, size = 30) {
    const x = size * (3/2 * q);
    const y = size * (Math.sqrt(3)/2 * q + Math.sqrt(3) * r);
    return { x, y };
}

// 创建偏矩形棋盘
function createBoard(radius = 8) {
    const board = document.getElementById('board');
    board.innerHTML = '';

    let totalValue = 99; // 总分
    let valueSequence = generateSequence(totalValue); // 生成值数组
    let valueIndex = 0; // 值数组的索引

    for (let q = -radius; q <= radius; q++) {
        // 跳过 q < -4 或 q > 3 的格子
        if (q < -4 || q > 3) continue;

        // 根据行号(q)的奇偶性确定列数(r)的范围
        let rStart, rEnd;
        if (q % 2 === 0) { // 偶数行
            rStart = -4;
            rEnd = 3;
        } else { // 奇数行
            rStart = -3;
            rEnd = 3;
        }
        for (let r = rStart; r <= rEnd; r++) {
            const s = -q - r;

            // 从生成的值数组中获取值
            let value = valueSequence[valueIndex++];
      
            // 定义 q 与调整量的映射关系
            const qAdjustments = {
              "1": -1,
              "2": -1,
              "3": -2,
              "-2": 1,
              "-3": 1,
              "-4": 2
            };

            // 只创建满足x + y + z = 0的六边形
            if (Math.abs(s) <= radius) {
                // 获取调整量，默认为 0（即不调整）
                const adjustment = qAdjustments[q] || 0;

                // 计算 adjustedR
                let adjustedR = r + adjustment;
               
                // 计算在容器中的位置
                const centerX = board.clientWidth / 2;
                const centerY = board.clientHeight / 2;
                
                const hex = document.createElement('div');
                hex.classList.add('hex');
                // 存储立方体坐标
                hex.dataset.q = q;
                hex.dataset.r = r;
                hex.dataset.s = s;
                hex.dataset.value = value; // 存储初始值
        
                // 调整后的像素坐标
                const { x, y } = cubeToPixel(q, adjustedR);
                hex.style.left = `${centerX + x - 25}px`;
                hex.style.top = `${centerY + y - 25}px`;
                

                // 设置基于坐标的颜色
                const hue = ((q + radius) / (2 * radius)) * 360;
                hex.style.backgroundColor = `hsl(${hue}, 70%, 80%)`;
                
                // 添加点击事件
                hex.addEventListener('click', () => selectHex(hex));
                
                board.appendChild(hex);
            }
        }
    }
}

// 选中六边形
function selectHex(hex) {
    // 清除所有选中状态
    document.querySelectorAll('.hex').forEach(h => {
        h.classList.remove('selected', 'neighbors');
    });
    
    // 选中当前六边形
    hex.classList.add('selected');
    
    // 获取立方体坐标
    const q = parseInt(hex.dataset.q);
    const r = parseInt(hex.dataset.r);
    const s = parseInt(hex.dataset.s);
    
    // 获取格子值
    const value = hex.dataset.value;
    
    // 显示选中信息
    document.getElementById('selected-info').textContent = 
        `选中六边形: (${q}, ${r}, ${s}), 值: ${value}`;
    
    // 计算并高亮相邻的六个六边形
    const neighbors = [
        {q: q+1, r: r-1, s: s}, // 右上
        {q: q+1, r: r, s: s-1}, // 右下
        {q: q, r: r+1, s: s-1}, // 左下
        {q: q-1, r: r+1, s: s}, // 左上
        {q: q-1, r: r, s: s+1}, // 左上
        {q: q, r: r-1, s: s+1}  // 右上
    ];
    
    // 查找并高亮相邻六边形
    neighbors.forEach(neighbor => {
        document.querySelectorAll('.hex').forEach(h => {
            if (parseInt(h.dataset.q) === neighbor.q && 
                parseInt(h.dataset.r) === neighbor.r && 
                parseInt(h.dataset.s) === neighbor.s) {
                h.classList.add('neighbors');
            }
        });
    });
}

// 重置棋盘
document.getElementById('reset-btn').addEventListener('click', () => {
    createBoard();
    document.getElementById('selected-info').textContent = '当前未选中任何六边形。';
});

// 切换坐标显示
let showCoords = false;
document.getElementById('toggle-coords-btn').addEventListener('click', () => {
    showCoords = !showCoords;
    document.querySelectorAll('.hex').forEach(hex => {
        if (showCoords) {
            hex.textContent = `${hex.dataset.q},${hex.dataset.r},${hex.dataset.s}`;
        } else {
            hex.textContent = `${hex.dataset.value}`;
        }
    });
});

// 初始化棋盘
window.addEventListener('DOMContentLoaded', () => {
    createBoard();
});
