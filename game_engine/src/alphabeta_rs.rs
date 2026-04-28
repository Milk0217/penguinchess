use crate::nnue_rs::NNUEWeights;
use crate::rules::GameState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const PIECE_IDS: [i32; 6] = [4, 6, 8, 5, 7, 9];

pub fn extract_sparse(state: &GameState) -> Vec<usize> {
    let mut features = Vec::with_capacity(6);
    for piece in &state.pieces {
        if piece.alive && piece.hex_idx.is_some() {
            let hex_idx = piece.hex_idx.unwrap();
            let piece_idx = PIECE_IDS.iter().position(|&pid| pid == piece.id).unwrap_or(0);
            features.push(piece_idx * 60 + hex_idx);
        }
    }
    features
}

pub fn extract_dense(state: &GameState) -> Vec<f32> {
    let mut dense = Vec::with_capacity(66);
    for cell in &state.board.cells {
        dense.push(match cell.state {
            crate::board::HexState::Active => cell.points as f32 / 3.0,
            _ => 0.0,
        });
    }
    let phase = if matches!(state.phase, crate::rules::Phase::Movement) { 0.5 } else { 0.0 };
    let p1_a = state.pieces.iter().filter(|p| p.owner() == 0 && p.alive).count() as f32 / 3.0;
    let p2_a = state.pieces.iter().filter(|p| p.owner() == 1 && p.alive).count() as f32 / 3.0;
    dense.extend_from_slice(&[
        state.scores[0] as f32 / 100.0, state.scores[1] as f32 / 100.0,
        phase, p1_a, p2_a, state.episode_steps as f32 / 500.0,
    ]);
    dense
}

// ─── Transposition Table ──────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum TTFlag { Exact, Lower, Upper }

struct TTEntry { depth: u8, score: f32, flag: TTFlag, best_move: usize, age: u32 }

struct TranspositionTable {
    entries: HashMap<(u64, u8), TTEntry>,
    age: u32, max_size: usize,
}

fn hash_state(state: &GameState) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for piece in &state.pieces { (piece.id, piece.hex_idx, piece.alive).hash(&mut h); }
    for cell in &state.board.cells { (cell.state as u8, cell.points).hash(&mut h); }
    state.current_player.hash(&mut h);
    h.finish()
}

impl TranspositionTable {
    fn new(max_size: usize) -> Self {
        Self { entries: HashMap::with_capacity(max_size.min(1 << 20)), age: 0, max_size }
    }

    fn lookup(&self, state: &GameState, depth: u8, alpha: f32, beta: f32) -> Option<(f32, usize)> {
        let entry = self.entries.get(&(hash_state(state), depth))?;
        if entry.age != self.age { return None; }
        match entry.flag {
            TTFlag::Exact => Some((entry.score, entry.best_move)),
            TTFlag::Lower if entry.score >= beta => Some((entry.score, entry.best_move)),
            TTFlag::Upper if entry.score <= alpha => Some((entry.score, entry.best_move)),
            _ => None,
        }
    }

    fn get_best_move(&self, state: &GameState) -> Option<usize> {
        let key = hash_state(state);
        for d in (1..=30u8).rev() {
            if let Some(e) = self.entries.get(&(key, d)) {
                if e.age == self.age { return Some(e.best_move); }
            }
        }
        None
    }

    fn store(&mut self, state: &GameState, depth: u8, score: f32, flag: TTFlag, best_move: usize) {
        if self.entries.len() >= self.max_size { self.entries.clear(); }
        self.entries.insert((hash_state(state), depth), TTEntry { depth, score, flag, best_move, age: self.age });
    }

    fn new_search(&mut self) { self.age += 1; }
}

// ─── Config & Result ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub max_depth: u8,
    pub time_limit_ms: u64,
    pub tt_size: usize,
    pub lmr_moves: u8,
    pub lmr_depth: u8,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self { max_depth: 6, time_limit_ms: 0, tt_size: 1 << 20, lmr_moves: 3, lmr_depth: 1 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub best_action: usize,
    pub score: f32,
    pub nodes_searched: u64,
    pub depth_reached: u8,
    pub time_ms: u64,
}

// ─── Search Engine ────────────────────────────────────────────

pub struct AlphaBetaSearch {
    pub weights: NNUEWeights,
    pub config: SearchConfig,
    tt: TranspositionTable,
    pub nodes_searched: u64,
    start_time: std::time::Instant,
}

impl AlphaBetaSearch {
    pub fn new(weights: NNUEWeights, config: SearchConfig) -> Self {
        Self {
            tt: TranspositionTable::new(config.tt_size), weights, config,
            nodes_searched: 0, start_time: std::time::Instant::now(),
        }
    }

    fn reset_stats(&mut self) {
        self.nodes_searched = 0;
        self.start_time = std::time::Instant::now();
        self.tt.new_search();
    }

    fn is_time_up(&self) -> bool {
        self.config.time_limit_ms > 0
            && self.start_time.elapsed().as_millis() as u64 >= self.config.time_limit_ms
    }

    fn terminal_value(state: &GameState) -> f32 {
        if !state.terminated { return 0.0; }
        let (s1, s2) = (state.scores[0] as f32, state.scores[1] as f32);
        if s1 > s2 { 1.0 } else if s2 > s1 { -1.0 } else { 0.0 }
    }

    fn evaluate(&self, state: &GameState) -> f32 {
        let sparse = extract_sparse(state);
        let dense = extract_dense(state);
        crate::nnue_rs::nnue_evaluate(&sparse, &dense, state.current_player, &self.weights)
    }

    fn is_terminal(state: &GameState) -> bool {
        state.terminated || state.get_legal_actions().is_empty()
    }

    fn order_moves(&self, state: &GameState, legal: &[usize], tt_best: Option<usize>)
        -> Vec<(usize, f32, bool)>
    {
        if legal.is_empty() { return vec![]; }
        if legal.len() == 1 { return vec![(legal[0], 0.0, false)]; }

        let mut scored: Vec<(usize, f32, bool)> = Vec::with_capacity(legal.len());

        if let Some(tbm) = tt_best {
            if legal.contains(&tbm) {
                let mut snap = state.clone();
                let (r, t) = snap.step(tbm);
                let ev = self.evaluate(&snap);
                scored.push((tbm, r + if t { Self::terminal_value(&snap) } else { ev }, t));
            }
        }

        for &a in legal {
            if Some(a) == tt_best { continue; }
            let mut snap = state.clone();
            let (r, t) = snap.step(a);
            let ev = self.evaluate(&snap);
            scored.push((a, r + if t { Self::terminal_value(&snap) } else { ev }, t));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(tbm) = tt_best {
            if let Some(pos) = scored.iter().position(|x| x.0 == tbm) {
                if pos != 0 { let entry = scored.remove(pos); scored.insert(0, entry); }
            }
        }
        scored
    }

    fn negamax(&mut self, state: &mut GameState, depth: u8,
                mut alpha: f32, mut beta: f32, is_pv: bool) -> (f32, Option<usize>)
    {
        self.nodes_searched += 1;
        if self.is_time_up() { return (self.evaluate(state), None); }

        if let Some((sc, bm)) = self.tt.lookup(state, depth, alpha, beta) {
            return (sc, Some(bm));
        }
        let tt_best = self.tt.get_best_move(state);

        if Self::is_terminal(state) || depth == 0 {
            let val = if state.terminated { Self::terminal_value(state) } else { self.evaluate(state) };
            self.tt.store(state, depth, val, TTFlag::Exact, 0);
            return (val, None);
        }

        let legal = state.get_legal_actions();
        if legal.is_empty() {
            let val = self.evaluate(state);
            self.tt.store(state, depth, val, TTFlag::Exact, 0);
            return (val, None);
        }

        let ordered = self.order_moves(state, &legal, tt_best);
        let mut best_action = ordered[0].0;
        let mut best_score = f32::NEG_INFINITY;
        let mut flag = TTFlag::Upper;

        for (i, &(action, _, term)) in ordered.iter().enumerate() {
            if self.is_time_up() { break; }
            let snap = state.clone();
            let (reward, _) = state.step(action);

            let score = if term {
                reward + Self::terminal_value(state)
            } else if depth == 1 {
                reward + self.evaluate(state)
            } else if i == 0 || is_pv {
                let (sc, _) = self.negamax(state, depth - 1, -beta, -alpha, is_pv);
                reward + (-sc)
            } else if depth > 2 && i >= self.config.lmr_moves as usize {
                let r = (depth - 1).saturating_sub(self.config.lmr_depth).max(1);
                let (sc, _) = self.negamax(state, r, -alpha - 1e-10, -alpha, false);
                let mut sc = -sc;
                if sc > alpha && sc < beta {
                    let (sc2, _) = self.negamax(state, depth - 1, -beta, -alpha, false);
                    sc = -sc2;
                }
                reward + sc
            } else {
                let (sc, _) = self.negamax(state, depth - 1, -alpha - 1e-10, -alpha, false);
                let mut sc = -sc;
                if sc > alpha && sc < beta {
                    let (sc2, _) = self.negamax(state, depth - 1, -beta, -alpha, false);
                    sc = -sc2;
                }
                reward + sc
            };

            *state = snap;
            if score > best_score { best_score = score; best_action = action; }
            alpha = alpha.max(best_score);
            if alpha >= beta { flag = TTFlag::Lower; break; }
        }

        self.tt.store(state, depth, best_score, flag, best_action);
        (best_score, Some(best_action))
    }

    pub fn search(&mut self, state: &GameState) -> SearchResult {
        self.reset_stats();
        let legal = state.get_legal_actions();
        if legal.is_empty() {
            return SearchResult { best_action: 0, score: 0.0, nodes_searched: 0, depth_reached: 0, time_ms: 0 };
        }
        if legal.len() == 1 {
            return SearchResult { best_action: legal[0], score: 0.0, nodes_searched: 1, depth_reached: 0, time_ms: 0 };
        }

        let ordered = self.order_moves(state, &legal, None);
        let mut best_action = ordered[0].0;
        let mut best_score = 0.0f32;
        let mut depth_reached = 0u8;

        for depth in 1..=self.config.max_depth {
            if self.is_time_up() { break; }
            let (mut alpha, mut beta) = if depth >= 3 && best_score.abs() > 0.01 {
                let w = (0.5 - 0.04 * depth as f32).max(0.1);
                ((best_score - w).max(-1.0), (best_score + w).min(1.0))
            } else { (-1.0, 1.0) };

            let mut result = self.negamax(&mut state.clone(), depth, alpha, beta, true);
            if result.0 <= alpha { result = self.negamax(&mut state.clone(), depth, -1.0, beta, true); }
            if result.0 >= beta { result = self.negamax(&mut state.clone(), depth, alpha, 1.0, true); }
            if let Some(a) = result.1 { best_action = a; best_score = result.0; }
            depth_reached = depth;
        }

        SearchResult {
            best_action, score: best_score,
            nodes_searched: self.nodes_searched, depth_reached,
            time_ms: self.start_time.elapsed().as_millis() as u64,
        }
    }
}
