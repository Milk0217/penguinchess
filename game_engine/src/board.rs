/// 六边形棋盘数据结构
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// 常量
// =============================================================================

pub const TOTAL_VALUE: i32 = 100;
pub const HEX_COUNT: usize = 60;
pub const PLAYER_1_PIECES: [i32; 3] = [4, 6, 8];
pub const PLAYER_2_PIECES: [i32; 3] = [5, 7, 9];
pub const PIECES_PER_PLAYER: usize = 3;

/// 六方向（立方体坐标）
pub const HEX_DIRECTIONS: [(i32, i32, i32); 6] = [
    (1, -1, 0), (1, 0, -1), (0, 1, -1),
    (-1, 1, 0), (-1, 0, 1), (0, -1, 1),
];

// =============================================================================
// 数据结构
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hex {
    pub q: i32,
    pub r: i32,
    pub s: i32,
}

impl Hex {
    pub fn new(q: i32, r: i32) -> Self {
        let s = -q - r;
        Hex { q, r, s }
    }

    pub fn neighbor(&self, dir: (i32, i32, i32)) -> Hex {
        Hex {
            q: self.q + dir.0,
            r: self.r + dir.1,
            s: self.s + dir.2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HexState {
    #[serde(rename = "active")]
    Active,
    #[serde(rename = "occupied")]
    Occupied,
    #[serde(rename = "used")]
    Used,
    #[serde(rename = "eliminated")]
    Eliminated,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HexCell {
    pub coord: Hex,
    pub state: HexState,
    pub points: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Piece {
    pub id: i32,
    pub alive: bool,
    pub hex_idx: Option<usize>,
    pub hex_value: i32,
}

impl Piece {
    pub fn owner(&self) -> usize {
        (self.id % 2) as usize
    }
}

/// 棋盘：60 个六边形格子的集合
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Board {
    pub cells: Vec<HexCell>,
    #[serde(skip)]
    pub hex_map: HashMap<(i32, i32, i32), usize>,
    #[serde(skip)]
    pub neighbors: Vec<Vec<usize>>,
}

impl Board {
    /// 从 cells 数据重建索引（JSON 反序列化后使用）
    pub fn rebuild_index_for_json(&mut self) {
        self.hex_map.clear();
        for (i, cell) in self.cells.iter().enumerate() {
            let c = cell.coord;
            self.hex_map.insert((c.q, c.r, c.s), i);
        }
        self.neighbors = vec![vec![]; self.cells.len()];
        for (i, cell) in self.cells.iter().enumerate() {
            for dir in &HEX_DIRECTIONS {
                let n = cell.coord.neighbor(*dir);
                if let Some(&ni) = self.hex_map.get(&(n.q, n.r, n.s)) {
                    self.neighbors[i].push(ni);
                }
            }
        }
    }
}

impl Board {
    /// 创建默认平行四边形棋盘（60格）
    pub fn new(value_seq: &[i32]) -> Self {
        let mut cells = Vec::with_capacity(HEX_COUNT);
        let mut idx = 0;

        for q in -4i32..=3 {
            let (start, end) = if q % 2 == 0 { (-4, 3) } else { (-3, 3) };
            for r in start..=end {
                let s = -q - r;
                if s.abs() > 8 {
                    continue;
                }
                cells.push(HexCell {
                    coord: Hex::new(q, r),
                    state: HexState::Active,
                    points: value_seq[idx],
                });
                idx += 1;
            }
        }

        let mut board = Board {
            cells,
            hex_map: HashMap::new(),
            neighbors: vec![],
        };
        board.build_index();
        board
    }

    /// 从自定义坐标创建棋盘
    pub fn from_coords(coords: &[(i32, i32, i32)], value_seq: &[i32]) -> Self {
        let cells = coords.iter().enumerate().map(|(i, &(q, r, s))| {
            HexCell {
                coord: Hex { q, r, s },
                state: HexState::Active,
                points: value_seq[i],
            }
        }).collect();

        let mut board = Board {
            cells,
            hex_map: HashMap::new(),
            neighbors: vec![],
        };
        board.build_index();
        board
    }

    fn build_index(&mut self) {
        // hex_map
        self.hex_map.clear();
        for (i, cell) in self.cells.iter().enumerate() {
            let c = cell.coord;
            self.hex_map.insert((c.q, c.r, c.s), i);
        }

        // neighbors
        self.neighbors = vec![vec![]; self.cells.len()];
        for (i, cell) in self.cells.iter().enumerate() {
            for dir in &HEX_DIRECTIONS {
                let n = cell.coord.neighbor(*dir);
                if let Some(&ni) = self.hex_map.get(&(n.q, n.r, n.s)) {
                    self.neighbors[i].push(ni);
                }
            }
        }
    }

    pub fn is_active(&self, idx: usize) -> bool {
        self.cells[idx].state == HexState::Active
    }

    pub fn is_occupied(&self, idx: usize) -> bool {
        self.cells[idx].state == HexState::Occupied
    }

    pub fn occupy(&mut self, idx: usize) {
        self.cells[idx].state = HexState::Occupied;
        self.cells[idx].points = 0;
    }

    pub fn mark_used(&mut self, idx: usize) {
        self.cells[idx].state = HexState::Used;
        self.cells[idx].points = 0;
    }

    pub fn eliminate(&mut self, idx: usize) {
        self.cells[idx].state = HexState::Eliminated;
        self.cells[idx].points = 0;
    }

    /// 获取某格子的合法移动目标（同轴 + 路径畅通）
    pub fn get_piece_moves(&self, piece_hex_idx: usize, occupied_set: &[usize]) -> Vec<usize> {
        let cur = &self.cells[piece_hex_idx].coord;
        let mut result = vec![];

        for (i, cell) in self.cells.iter().enumerate() {
            if cell.state != HexState::Active {
                continue;
            }
            if occupied_set.contains(&i) {
                continue;
            }
            let t = &cell.coord;
            if t.q != cur.q && t.r != cur.r && t.s != cur.s {
                continue;
            }
            if !self.path_clear(piece_hex_idx, i, occupied_set) {
                continue;
            }
            result.push(i);
        }
        result
    }

    /// 检查路径是否畅通
    pub fn path_clear(&self, from: usize, to: usize, occupied_set: &[usize]) -> bool {
        let f = &self.cells[from].coord;
        let t = &self.cells[to].coord;
        let dq = t.q - f.q;
        let dr = t.r - f.r;
        let ds = t.s - f.s;

        let steps = dq.abs().max(dr.abs()).max(ds.abs());
        if steps <= 1 {
            return true;
        }

        let sq = dq.signum();
        let sr = dr.signum();
        let ss = ds.signum();

        for i in 1..steps {
            let mq = f.q + sq * i;
            let mr = f.r + sr * i;
            let ms = f.s + ss * i;
            if let Some(&mid) = self.hex_map.get(&(mq, mr, ms)) {
                if occupied_set.contains(&mid) || self.cells[mid].state != HexState::Active {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}
