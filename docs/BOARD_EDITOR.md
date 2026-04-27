# 企鹅棋 · 棋盘编辑器 v2 设计文档

## 概述

棋盘编辑器需要与游戏后端深度集成，允许用户：
1. 在可视化编辑器中设计 60 格正六边形棋盘
2. 将设计**直接保存到后端**
3. 在游戏设置中**直接选择已保存的棋盘**进行游戏

---

## 架构改动

### 2.1 数据流

```
编辑器 → [API: 保存棋盘] → 后端文件存储
                              ↓
游戏 ← [API: 创建游戏(选棋盘)] ← 棋盘列表
```

### 2.2 新增 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/boards` | GET | 获取所有已保存棋盘列表 |
| `/api/boards` | POST | 保存新棋盘（编辑器调用） |
| `/api/boards/<id>` | DELETE | 删除棋盘 |
| `/api/game` | POST | 创建游戏时可选 `board_id` 参数 |

### 2.3 棋盘存储格式

```json
// backend_data/boards/custom-abc123.json
{
  "id": "custom-abc123",
  "name": "我的棋盘",
  "created_at": "2026-04-23T20:00:00Z",
  "hex_count": 60,
  "hexes": [
    { "q": -4, "r": -4, "s": 8 },
    { "q": -3, "r": -4, "s": 7 },
    ...
  ]
}
```

### 2.4 目录结构

```
penguinchess/
├── server/
│   ├── app.py              # API 路由
│   └── game.py             # 游戏会话
├── backend_data/
│   └── boards/             # 保存的棋盘 JSON 文件
│       ├── parallelogram.json  # 内置默认棋盘
│       └── custom-xxx.json     # 用户自定义棋盘
└── frontend/src/
    └── editor/
        ├── BoardEditor.tsx
        ├── EditorCanvas.tsx
        └── Sidebar.tsx
```

---

## 功能规格

### 3.1 编辑器增强

#### 画布缩放与平移
- **鼠标滚轮** — 缩放画布（0.5x ~ 2x）
- **拖拽** — 按住空格+拖拽，或鼠标中键拖拽平移
- **适应屏幕** — 双击重置缩放和位置
- **缩放滑块** — 侧边栏提供缩放控制

#### 棋盘命名
- 输入框让用户为棋盘命名
- 默认名称：`我的棋盘 {timestamp}`

#### 保存到后端
- 点击"保存"按钮 → POST 到 `/api/boards`
- 成功后显示成功提示，可直接"开始游戏"

#### 加载已有棋盘
- 从后端获取已保存棋盘列表
- 选择一个加载到编辑器继续编辑

### 3.2 游戏集成

#### 棋盘选择下拉框
- 在游戏设置中显示所有已保存棋盘
- 包括内置棋盘（平行四边形、正六边形等）
- 选中后创建新游戏使用该棋盘

#### 创建游戏 API 扩展
```python
@app.route("/api/game", methods=["POST"])
def api_create_game():
    data = request.get_json(silent=True) or {}
    seed = data.get("seed")
    board_id = data.get("board_id")  # 新增参数

    session = create_session(seed=seed, board_id=board_id)
    return jsonify({"state": session.state()})
```

### 3.3 内置棋盘预设

| ID | 名称 | 描述 |
|----|------|------|
| `parallelogram` | 平行四边形 | 标准企鹅棋布局，60格 |
| `hexagon` | 正六边形 | 中心对称六边形，61格（需取消1个） |

---

## 前端改动

### 4.1 编辑器组件更新

#### EditorCanvas 新增 Props
```typescript
interface EditorCanvasProps {
  selected: Set<string>;
  onToggle: (key: string) => void;
  hexSize?: number;
  zoom?: number;        // 新增：缩放级别
  panX?: number;        // 新增：平移 X
  panY?: number;        // 新增：平移 Y
  onZoomChange?: (zoom: number) => void;
  onPanChange?: (x: number, y: number) => void;
}
```

#### Sidebar 新增功能
```typescript
interface SidebarProps {
  // ...现有props
  boardName: string;
  onBoardNameChange: (name: string) => void;
  onSave: () => void;
  onLoad: () => void;
  savedBoards: Array<{ id: string; name: string }>;
  isSaving: boolean;
  isLoading: boolean;
}
```

#### BoardEditor 状态
```typescript
const [boardName, setBoardName] = useState("我的棋盘");
const [zoom, setZoom] = useState(1);
const [pan, setPan] = useState({ x: 0, y: 0 });
const [savedBoards, setSavedBoards] = useState([]);
const [isSaving, setIsSaving] = useState(false);
```

### 4.2 API 模块扩展

```typescript
// frontend/src/api.ts

// 获取已保存棋盘列表
export async function getBoards(): Promise<Array<{ id: string; name: string; hex_count: number }>> {
  const res = await fetch("/api/boards");
  return res.json();
}

// 保存棋盘
export async function saveBoard(board: {
  name: string;
  hexes: Array<{ q: number; r: number; s: number }>;
}): Promise<{ id: string }> {
  const res = await fetch("/api/boards", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(board),
  });
  return res.json();
}

// 删除棋盘
export async function deleteBoard(id: string): Promise<void> {
  await fetch(`/api/boards/${id}`, { method: "DELETE" });
}

// 创建游戏（支持选择棋盘）
export async function createGame(opts?: { seed?: number; board_id?: string }): Promise<{ state: GameState }> {
  const res = await fetch("/api/game", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts || {}),
  });
  return res.json();
}
```

---

## 后端改动

### 5.1 棋盘存储模块

```python
# server/boards.py
"""
棋盘存储管理
"""

import json
import os
from pathlib import Path
from typing import List, Optional

BOARD_DIR = Path(__file__).parent.parent / "backend_data" / "boards"

def get_boards_dir() -> Path:
    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    return BOARD_DIR

def list_boards() -> List[dict]:
    """返回所有已保存棋盘（不含 hexes 数据）"""
    boards = []
    for f in get_boards_dir().glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
            boards.append({
                "id": data["id"],
                "name": data["name"],
                "hex_count": data["hex_count"],
                "created_at": data["created_at"],
            })
    return boards

def get_board(board_id: str) -> Optional[dict]:
    """根据 ID 获取完整棋盘数据"""
    path = get_boards_dir() / f"{board_id}.json"
    if not path.exists():
        return None
    with open(path) as fp:
        return json.load(fp)

def save_board(board_id: str, name: str, hexes: List[dict]) -> dict:
    """保存棋盘"""
    data = {
        "id": board_id,
        "name": name,
        "hex_count": len(hexes),
        "created_at": datetime.now().isoformat(),
        "hexes": hexes,
    }
    path = get_boards_dir() / f"{board_id}.json"
    with open(path, "w") as fp:
        json.dump(data, fp, indent=2)
    return {"id": board_id, "name": name, "hex_count": len(hexes)}

def delete_board(board_id: str) -> bool:
    """删除棋盘"""
    path = get_boards_dir() / f"{board_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False

# 内置棋盘初始化
def init_builtin_boards():
    """如果内置棋盘文件不存在，创建它们"""
    builtin_boards = {
        "parallelogram": {
            "name": "平行四边形",
            "hexes": [...]  # 60格
        },
        "hexagon": {
            "name": "正六边形",
            "hexes": [...]  # 61格
        }
    }
    # ... 实现
```

### 5.2 PenguinChessCore 扩展

需要修改 `PenguinChessCore.__init__` 支持自定义棋盘：

```python
class PenguinChessCore:
    def __init__(self, seed=None, custom_hexes=None):
        """
        custom_hexes: 可选，自定义格子列表。
                      如果为 None，使用默认的平行四边形 60 格。
        """
        self.seed = seed
        self._custom_hexes = custom_hexes
        self._init_game()

    def _init_game(self):
        if self._custom_hexes:
            self.hexes = [Hex(**h) for h in self._custom_hexes]
        else:
            self._init_default_board()  # 默认平行四边形
        self._build_hex_map()
        # ... 其他初始化
```

### 5.3 API 路由

```python
from datetime import datetime
from boards import list_boards, get_board, save_board, delete_board

@app.route("/api/boards", methods=["GET"])
def api_list_boards():
    return jsonify(list_boards())

@app.route("/api/boards", methods=["POST"])
def api_save_board():
    data = request.get_json()
    board_id = f"custom-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    result = save_board(board_id, data["name"], data["hexes"])
    return jsonify(result)

@app.route("/api/boards/<board_id>", methods=["DELETE"])
def api_delete_board(board_id: str):
    success = delete_board(board_id)
    if not success:
        return jsonify({"error": "board not found"}), 404
    return "", 204

@app.route("/api/game", methods=["POST"])
def api_create_game():
    data = request.get_json(silent=True) or {}
    seed = data.get("seed")
    board_id = data.get("board_id")

    session = create_session(seed=seed, board_id=board_id)
    return jsonify({"state": session.state()})
```

### 5.4 GameSession 扩展

```python
def create_session(seed=None, board_id=None):
    # ... 清理旧会话
    session_id = str(uuid.uuid4())[:8]

    # 获取棋盘配置
    if board_id:
        board_data = get_board(board_id)
        if board_data:
            custom_hexes = board_data["hexes"]
        else:
            custom_hexes = None  # 回退到默认
    else:
        custom_hexes = None

    session = GameSession(session_id=session_id, seed=seed, custom_hexes=custom_hexes)
    _sessions[session_id] = session
    return session

@dataclass
class GameSession:
    session_id: str
    seed: Optional[int]
    custom_hexes: Optional[List[dict]] = None  # 新增

    def __post_init__(self):
        self._core = PenguinChessCore(seed=self.seed, custom_hexes=self.custom_hexes)
        # ...
```

---

## 实现计划

### Phase 1: 后端基础（API + 存储）
- [x] 创建 `backend_data/boards/` 目录
- [x] 实现 `server/boards.py` 存储模块
- [x] 添加 API 路由（GET/POST/DELETE /api/boards）
- [x] 初始化内置棋盘文件

### Phase 2: PenguinChessCore 支持自定义棋盘
- [x] 修改 `PenguinChessCore.__init__` 接受 `custom_hexes`
- [x] 修改 `GameSession` 传递自定义棋盘
- [x] 测试默认棋盘和新自定义棋盘都能正常工作

### Phase 3: 前端 API 集成
- [x] 扩展 `api.ts` 添加 boards 相关函数
- [x] Sidebar 添加保存/加载功能
- [x] 添加棋盘命名输入框

### Phase 4: 编辑器画布增强
- [x] 添加缩放功能（滚轮 + 滑块）
- [x] 添加平移功能（拖拽）
- [x] 调整默认缩放使棋盘完全可见

### Phase 5: 游戏集成
- [x] 游戏设置中显示已保存棋盘列表
- [x] 选择棋盘后创建游戏
- [x] 测试完整流程

---

## 内置棋盘

| ID | 名称 | 描述 |
|----|------|------|
| `parallelogram` | 平行四边形 | 标准企鹅棋布局，60格 |
| `hexagon` | 正六边形 | 中心对称六边形，61格 |
| `default` | default | 默认自定义棋盘 |
| `custom-1777140498` | (用户自定义) | 用户设计的自定义 60 格棋盘 |

---

## 验证测试

1. 编辑器保存棋盘 → 刷新页面 → 棋盘列表中看到已保存棋盘
2. 编辑器加载棋盘 → 修改 → 再次保存
3. 游戏选择自定义棋盘 → 放置/移动阶段正常
4. 不同棋盘布局（60格）都能正常游戏
