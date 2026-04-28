use crate::nnue_rs::NNUEWeights;
use crate::rules::GameState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const PIECE_IDS: [i32; 6] = [4, 6, 8, 5, 7, 9];
const EVAL_BATCH_STRIDE: usize = 75; // 8 sparse + 66 dense + 1 stm

pub type EvalBatchFn = unsafe extern "C" fn(
    data: *const f32,   // (batch_size × EVAL_BATCH_STRIDE) flat array
    batch_size: i32,
    scores_out: *mut f32,
) -> i32;

static mut EVAL_CALLBACK: Option<EvalBatchFn> = None;

pub fn set_eval_callback(cb: EvalBatchFn) {
    unsafe { EVAL_CALLBACK = Some(cb); }
}

fn encode_for_batch(state: &GameState, buf: &mut [f32]) {
    let sparse = extract_sparse(state);
    for i in 0..8 {
        buf[i] = if i < sparse.len() { sparse[i] as f32 } else { -1.0 };
    }
    let dense = extract_dense(state);
    buf[8..74].copy_from_slice(&dense);
    buf[74] = state.current_player as f32;
}

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
        // Try callback first (CUDA accelerated)
        unsafe {
            if let Some(cb) = EVAL_CALLBACK {
                let mut buf = [0.0f32; EVAL_BATCH_STRIDE];
                encode_for_batch(state, &mut buf);
                let mut score = 0.0f32;
                cb(buf.as_ptr(), 1, &mut score as *mut f32);
                return score;
            }
        }
        // Fallback: Rust NNUE
        let sparse = extract_sparse(state);
        let dense = extract_dense(state);
        crate::nnue_rs::nnue_evaluate(&sparse, &dense, state.current_player, &self.weights)
    }

    fn batch_evaluate(&self, states: &[&GameState]) -> Vec<f32> {
        let n = states.len();
        if n == 0 { return vec![]; }

        // Try callback
        unsafe {
            if let Some(cb) = EVAL_CALLBACK {
                let mut batch = vec![0.0f32; n * EVAL_BATCH_STRIDE];
                for (i, state) in states.iter().enumerate() {
                    encode_for_batch(state, &mut batch[i * EVAL_BATCH_STRIDE..(i + 1) * EVAL_BATCH_STRIDE]);
                }
                let mut scores = vec![0.0f32; n];
                cb(batch.as_ptr(), n as i32, scores.as_mut_ptr());
                return scores;
            }
        }
        // Fallback: sequential Rust NNUE
        states.iter().map(|s| {
            let sparse = extract_sparse(s);
            let dense = extract_dense(s);
            crate::nnue_rs::nnue_evaluate(&sparse, &dense, s.current_player, &self.weights)
        }).collect()
    }

    fn is_terminal(state: &GameState) -> bool {
        state.terminated || state.get_legal_actions().is_empty()
    }

    fn nnue_ordering_depth(&self, state: &GameState) -> u8 {
        // Tactical positions (high stakes, many moves) → deeper NNUE ordering
        // Calm positions → shallower NNUE ordering for speed

        // Check for high-value hexes still available
        let has_high_value = state.board.cells.iter()
            .any(|c| c.state == crate::board::HexState::Active && c.points >= 3);

        // Check piece count — more pieces = more tactical
        let pieces_alive = state.pieces.iter().filter(|p| p.alive).count();

        if has_high_value && pieces_alive >= 4 {
            3  // Full tactical: NNUE at depth ≤ 3
        } else if pieces_alive >= 3 {
            2  // Moderate: NNUE at depth ≤ 2
        } else {
            1  // Endgame: NNUE at root only (depth ≤ 1)
        }
    }

    fn order_moves(&self, state: &GameState, legal: &[usize], tt_best: Option<usize>, node_depth: u8)
        -> Vec<(usize, f32, bool)>
    {
        if legal.is_empty() { return vec![]; }
        if legal.len() == 1 { return vec![(legal[0], 0.0, false)]; }

        // NNUE-based ordering: expensive, only for shallow nodes
        let nnue_depth = self.nnue_ordering_depth(state);
        if node_depth <= nnue_depth {
            let mut children: Vec<(usize, GameState, f32, bool)> = Vec::with_capacity(legal.len());
            for &a in legal {
                let mut child = state.clone();
                let (r, t) = child.step(a);
                children.push((a, child, r, t));
            }
            let child_refs: Vec<&GameState> = children.iter().map(|(_, s, _, _)| s).collect();
            let evals = self.batch_evaluate(&child_refs);
            let mut scored: Vec<(usize, f32, bool)> = children.iter().zip(evals.iter())
                .map(|((a, child, r, t), ev)| {
                    (*a, r + if *t { Self::terminal_value(child) } else { *ev }, *t)
                }).collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return scored;
        }

        // Fast heuristic ordering for deep nodes
        let mut scored: Vec<(usize, f32, bool)> = Vec::with_capacity(legal.len());

        // TT best move first (if any)
        if let Some(tbm) = tt_best {
            if legal.contains(&tbm) {
                let mut snap = state.clone();
                let (r, t) = snap.step(tbm);
                scored.push((tbm, r + if t { Self::terminal_value(&snap) } else { 0.0 }, t));
            }
        }

        // Score remaining moves by reward + heuristic
        let alive_before = state.pieces.iter().filter(|p| p.alive).count();
        let mut heur: Vec<(usize, f32, bool)> = Vec::with_capacity(legal.len());
        for &a in legal {
            if Some(a) == tt_best { continue; }
            let mut snap = state.clone();
            let (r, t) = snap.step(a);
            let alive_after = snap.pieces.iter().filter(|p| p.alive).count();
            let pieces_killed = alive_before - alive_after;
            let extra = pieces_killed as f32 * 10.0;
            heur.push((a, r + extra, t));
        }
        heur.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.extend(heur);
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

        let ordered = self.order_moves(state, &legal, tt_best, depth);
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

        let ordered = self.order_moves(state, &legal, None, 0); // root: depth 0 → NNUE ordering
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
