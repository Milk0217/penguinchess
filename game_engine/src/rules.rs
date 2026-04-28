/// 企鹅棋规则引擎
use crate::board::*;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

// =============================================================================
// 游戏状态
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Phase {
    #[serde(rename = "placement")]
    Placement,
    #[serde(rename = "movement")]
    Movement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub board: Board,
    pub pieces: Vec<Piece>,
    pub scores: [i32; 2],
    pub phase: Phase,
    pub current_player: usize, // 0=P1, 1=P2
    pub placement_count: usize,
    pub episode_steps: usize,
    pub terminated: bool,
    /// 前一步动作信息
    pub last_action: Option<ActionInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionInfo {
    pub action: usize,
    pub player: usize,
}

impl GameState {
    pub fn new(board: Board) -> Self {
        let mut pieces = Vec::with_capacity(6);
        for &pid in &PLAYER_1_PIECES {
            pieces.push(Piece { id: pid, alive: true, hex_idx: None, hex_value: 0 });
        }
        for &pid in &PLAYER_2_PIECES {
            pieces.push(Piece { id: pid, alive: true, hex_idx: None, hex_value: 0 });
        }

        GameState {
            board,
            pieces,
            scores: [0, 0],
            phase: Phase::Placement,
            current_player: 0,
            placement_count: 0,
            episode_steps: 0,
            terminated: false,
            last_action: None,
        }
    }

    /// 获取合法动作列表
    pub fn get_legal_actions(&self) -> Vec<usize> {
        if self.terminated {
            return vec![];
        }

        if self.phase == Phase::Placement {
            return self.board.cells.iter().enumerate()
                .filter(|(i, c)| {
                    c.state == HexState::Active
                    && c.points < 3  // 放置阶段禁止放在3分格上
                    && !self.pieces.iter().any(|p| p.hex_idx == Some(*i))
                })
                .map(|(i, _)| i)
                .collect();
        }

        // 移动阶段
        let mut ids = vec![];
        let occ = self.occupied_set();
        for piece in &self.pieces {
            if !piece.alive || piece.hex_idx.is_none() {
                continue;
            }
            if piece.owner() != self.current_player {
                continue;
            }
            let moves = self.board.get_piece_moves(piece.hex_idx.unwrap(), &occ);
            ids.extend(moves);
        }
        ids.sort();
        ids.dedup();
        ids
    }

    fn occupied_set(&self) -> Vec<usize> {
        self.pieces.iter()
            .filter(|p| p.alive && p.hex_idx.is_some())
            .map(|p| p.hex_idx.unwrap())
            .collect()
    }

    /// 执行一步动作。返回 (奖励, 是否终止)
    pub fn step(&mut self, action: usize) -> (f32, bool) {
        if self.terminated {
            return (0.0, true);
        }

        let prev_player = self.current_player;
        let _prev_pieces_alive: Vec<bool> = self.pieces.iter().map(|p| p.alive).collect();

        if self.phase == Phase::Placement {
            self.do_placement(action);
        } else {
            self.do_movement(action);
        }

        self.episode_steps += 1;

        // 消除 + 销毁
        if !self.terminated {
            self.eliminate_disconnected();
            self.destroy_immobile();
        }

        self.check_game_over();

        // 切换玩家
        if !self.terminated {
            self.current_player = 1 - prev_player;
        }

        let reward = 0.0; // 简化版奖励
        self.last_action = Some(ActionInfo { action, player: prev_player });
        (reward, self.terminated)
    }

    fn do_placement(&mut self, action: usize) {
        let hex_idx = action;
        let cell = &self.board.cells[hex_idx];
        let score = cell.points;

        // 分配棋子
        let player_pieces = if self.current_player == 0 { &PLAYER_1_PIECES } else { &PLAYER_2_PIECES };
        let piece_idx = self.placement_count / 2;
        let piece_id = player_pieces[piece_idx];

        let piece = self.pieces.iter_mut().find(|p| p.id == piece_id).unwrap();
        piece.hex_idx = Some(hex_idx);
        piece.hex_value = cell.points;

        self.scores[self.current_player] += score;
        self.board.occupy(hex_idx);
        self.placement_count += 1;

        if self.placement_count >= PIECES_PER_PLAYER * 2 {
            self.phase = Phase::Movement;
        }
    }

    fn do_movement(&mut self, action: usize) {
        let occ = self.occupied_set();
        // 找第一个能移动到目标格子的己方棋子
        let piece_idx = self.pieces.iter().position(|p| {
            if !p.alive || p.hex_idx.is_none() { return false; }
            if p.owner() != self.current_player { return false; }
            let moves = self.board.get_piece_moves(p.hex_idx.unwrap(), &occ);
            moves.contains(&action)
        });

        if let Some(pi) = piece_idx {
            let piece = &mut self.pieces[pi];
            let old_idx = piece.hex_idx.unwrap();
            let score = self.board.cells[action].points;

            // 旧格子标记为 used
            self.board.mark_used(old_idx);
            piece.hex_value = score;
            piece.hex_idx = Some(action);
            // 新格子 occupied
            self.board.occupy(action);
            self.scores[self.current_player] += score;
        }
    }

    fn eliminate_disconnected(&mut self) {
        // Flood fill from all alive pieces
        let mut connected = vec![false; self.board.cells.len()];
        let mut stack: Vec<usize> = vec![];

        for piece in &self.pieces {
            if piece.alive && piece.hex_idx.is_some() {
                stack.push(piece.hex_idx.unwrap());
            }
        }

        while let Some(idx) = stack.pop() {
            if connected[idx] { continue; }
            let state = self.board.cells[idx].state;
            if state == HexState::Eliminated || state == HexState::Used {
                continue;
            }
            connected[idx] = true;
            for &ni in &self.board.neighbors[idx] {
                if !connected[ni] {
                    let ns = self.board.cells[ni].state;
                    if ns == HexState::Active || ns == HexState::Occupied {
                        stack.push(ni);
                    }
                }
            }
        }

        for i in 0..self.board.cells.len() {
            if self.board.cells[i].state == HexState::Active && !connected[i] {
                self.board.eliminate(i);
            }
        }
    }

    fn destroy_immobile(&mut self) {
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..self.pieces.len() {
                if !self.pieces[i].alive || self.pieces[i].hex_idx.is_none() {
                    continue;
                }
                let hex_idx = self.pieces[i].hex_idx.unwrap();
                let has_move = self.board.neighbors[hex_idx].iter()
                    .any(|&ni| self.board.cells[ni].state == HexState::Active);
                if !has_move {
                    self.board.mark_used(hex_idx);
                    self.pieces[i].hex_idx = None;
                    self.pieces[i].alive = false;
                    changed = true;
                }
            }
        }
    }

    fn check_game_over(&mut self) {
        let p1_alive = self.pieces.iter().filter(|p| p.owner() == 0 && p.alive).count();
        let p2_alive = self.pieces.iter().filter(|p| p.owner() == 1 && p.alive).count();

        if p1_alive == 0 || p2_alive == 0 {
            let survivor = if p1_alive == 0 { 1 } else { 0 };
            for cell in &mut self.board.cells {
                if cell.state == HexState::Active {
                    self.scores[survivor] += cell.points;
                    cell.state = HexState::Used;
                    cell.points = 0;
                }
            }
            self.terminated = true;
            return;
        }

        let has_active = self.board.cells.iter().any(|c| c.state == HexState::Active);
        if !has_active {
            self.terminated = true;
        }
    }
}

pub fn generate_sequence(seed: u64) -> Vec<i32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut seq = vec![3; 10];
    seq.resize(30, 2);
    seq.resize(60, 1);
    seq.shuffle(&mut rng);
    seq
}
