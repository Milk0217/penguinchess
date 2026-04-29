use crate::nnue_rs::NNUEWeights;
use crate::rules::GameState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const PIECE_IDS: [i32; 6] = [4, 6, 8, 5, 7, 9];
const EVAL_BATCH_STRIDE: usize = 75;
const MAX_LEGAL: usize = 60;
const MAX_DEPTH: usize = 64;

pub type EvalBatchFn = unsafe extern "C" fn(data: *const f32, batch_size: i32, scores_out: *mut f32) -> i32;
static mut EVAL_CALLBACK: Option<EvalBatchFn> = None;
pub fn set_eval_callback(cb: EvalBatchFn) { unsafe { EVAL_CALLBACK = Some(cb); } }

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

// ─── History Heuristic ────────────────────────────────────────

struct HistoryTable {
    scores: [i32; MAX_LEGAL],
    max_: i32,
}

impl HistoryTable {
    fn new() -> Self { Self { scores: [0; MAX_LEGAL], max_: 1 } }
    fn update(&mut self, action: usize, depth: u8) {
        let bonus = (depth as i32) * (depth as i32);
        self.scores[action] = self.scores[action].saturating_add(bonus);
        self.max_ = self.max_.max(self.scores[action]);
    }
    fn clear(&mut self) { self.scores.fill(0); self.max_ = 1; }
    fn get(&self, action: usize) -> i32 { self.scores[action] }
}

// ─── Killer Moves ─────────────────────────────────────────────

struct KillerTable {
    moves: Vec<[Option<usize>; 2]>,
}

impl KillerTable {
    fn new() -> Self { Self { moves: vec![[None, None]; MAX_DEPTH] } }
    fn add(&mut self, depth: usize, action: usize) {
        if depth >= MAX_DEPTH { return; }
        let slot = &mut self.moves[depth];
        if slot[0] != Some(action) { slot[1] = slot[0]; slot[0] = Some(action); }
    }
    fn is_killer(&self, depth: usize, action: usize) -> bool {
        self.moves.get(depth).map_or(false, |s| s[0] == Some(action) || s[1] == Some(action))
    }
    fn clear(&mut self) { self.moves.fill([None, None]); }
}

// ─── Ordered Child (stores result of single clone+step) ───────

struct OrderedChild {
    action: usize,
    reward: f32,
    is_terminal: bool,
    child_state: Option<GameState>,
    sort_score: f32, // for ordering only
}

// ─── Config & Result ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub max_depth: u8,
    pub time_limit_ms: u64,
    pub tt_size: usize,
    pub lmr_moves: u8,
    pub lmr_depth: u8,
    /// NNUE ordering: used at root all the time.
    /// For recursive nodes: used when depth_from_root <= this value (1=root only, 2=root+ply2, etc.)
    pub nnue_order_depth: u8,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self { max_depth: 6, time_limit_ms: 0, tt_size: 1 << 20, lmr_moves: 3, lmr_depth: 1, nnue_order_depth: 2 }
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
    history: HistoryTable,
    killers: KillerTable,
    pub nodes_searched: u64,
    start_time: std::time::Instant,
}

impl AlphaBetaSearch {
    pub fn new(weights: NNUEWeights, config: SearchConfig) -> Self {
        Self {
            tt: TranspositionTable::new(config.tt_size), weights, config,
            history: HistoryTable::new(), killers: KillerTable::new(),
            nodes_searched: 0, start_time: std::time::Instant::now(),
        }
    }

    fn reset_stats(&mut self) {
        self.nodes_searched = 0;
        self.start_time = std::time::Instant::now();
        self.tt.new_search();
        self.history.clear();
        self.killers.clear();
    }

    fn is_time_up(&self) -> bool {
        self.config.time_limit_ms > 0
            && self.start_time.elapsed().as_millis() as u64 >= self.config.time_limit_ms
    }

    fn terminal_value(state: &GameState) -> f32 {
        if !state.terminated { return 0.0; }
        if state.scores[0] > state.scores[1] { 1.0 } else if state.scores[1] > state.scores[0] { -1.0 } else { 0.0 }
    }

    fn evaluate_raw(&self, sparse: &[usize], dense: &[f32], stm: usize) -> f32 {
        crate::nnue_rs::nnue_evaluate(sparse, dense, stm, &self.weights)
    }

    fn evaluate(&self, state: &GameState) -> f32 {
        unsafe {
            if let Some(cb) = EVAL_CALLBACK {
                let mut buf = [0.0f32; EVAL_BATCH_STRIDE];
                encode_for_batch(state, &mut buf);
                let mut score = 0.0f32;
                cb(buf.as_ptr(), 1, &mut score as *mut f32);
                return score;
            }
        }
        let sparse = extract_sparse(state);
        let dense = extract_dense(state);
        crate::nnue_rs::nnue_evaluate(&sparse, &dense, state.current_player, &self.weights)
    }

    fn batch_evaluate(&self, states: &[&GameState]) -> Vec<f32> {
        let n = states.len();
        if n == 0 { return vec![]; }
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
        states.iter().map(|s| self.evaluate(s)).collect()
    }

    fn is_terminal(state: &GameState) -> bool {
        state.terminated || state.get_legal_actions().is_empty()
    }

    /// NNUE ordering depth boundary: how far from root to use NNUE ordering.
    /// depth_from_root=1 → root, =2 → first recursion, etc.
    fn nnue_ordering_depth(&self, state: &GameState) -> u8 {
        let base = self.config.nnue_order_depth;
        if base == 0 { return 0; }
        let alive = state.pieces.iter().filter(|p| p.alive).count();
        let has_high = state.board.cells.iter()
            .any(|c| c.state == crate::board::HexState::Active && c.points >= 3);
        if has_high && alive >= 4 { base }
        else if alive >= 3 { base.min(2) }
        else { 1u8.min(base) }
    }

    /// Build ordered children list.
    ///
    /// - `depth_from_root=0` (root ordering in search()): always NNUE
    /// - `depth_from_root>0` (negamax recursion): NNUE if <= nnue_ordering_depth, else heuristic
    ///
    /// NNUE branch: stores child_state (single clone+step+eval).
    /// Heuristic branch: no clone+step, uses TT best + killers + history.
    fn order_children(&mut self, state: &GameState, legal: &[usize],
                       tt_best: Option<usize>, depth_from_root: u8)
        -> Vec<OrderedChild>
    {
        if legal.is_empty() { return vec![]; }
        if legal.len() == 1 {
            return vec![OrderedChild {
                action: legal[0], reward: 0.0, is_terminal: false,
                child_state: None, sort_score: 0.0,
            }];
        }

        let nnue_max = self.nnue_ordering_depth(state);
        let use_nnue = depth_from_root == 0 || depth_from_root <= nnue_max;

        if use_nnue {
            // ── NNUE ordering: clone+step+eval all children ──
            let mut children: Vec<OrderedChild> = Vec::with_capacity(legal.len());

            // Single clone+step pass
            for &a in legal {
                let mut snap = state.clone();
                let (r, t) = snap.step(a);
                children.push(OrderedChild {
                    action: a, reward: r, is_terminal: t,
                    child_state: Some(snap), sort_score: 0.0,
                });
            }

            // Batch evaluate
            let child_refs: Vec<&GameState> = children.iter()
                .filter_map(|c| c.child_state.as_ref()).collect();
            let evals = self.batch_evaluate(&child_refs);

            // Compute combined score and store in sort_score
            for (i, child) in children.iter_mut().enumerate() {
                let ev = if child.is_terminal { 0.0 } else { evals[i] };
                child.sort_score = child.reward
                    + if child.is_terminal { Self::terminal_value(child.child_state.as_ref().unwrap()) } else { ev };
            }

            // Sort in-place by score descending
            children.sort_by(|a, b| b.sort_score.partial_cmp(&a.sort_score).unwrap_or(std::cmp::Ordering::Equal));
            children
        } else {
            // ── Heuristic ordering: no clone+step, use TT + killers + history ──
            let mut children: Vec<OrderedChild> = Vec::with_capacity(legal.len());
            for &a in legal {
                let mut h = 0i64;
                if Some(a) == tt_best { h += 10_000_000; }
                if self.killers.is_killer(depth_from_root as usize, a) { h += 5_000_000; }
                h += self.history.get(a) as i64;
                children.push(OrderedChild {
                    action: a, reward: 0.0, is_terminal: false,
                    child_state: None, sort_score: h as f32,
                });
            }
            children.sort_by(|a, b| b.sort_score.partial_cmp(&a.sort_score).unwrap_or(std::cmp::Ordering::Equal));
            children
        }
    }

    fn negamax(&mut self, state: &mut GameState, depth_remaining: u8,
                depth_from_root: u8, mut alpha: f32, beta: f32, is_pv: bool) -> (f32, Option<usize>)
    {
        self.nodes_searched += 1;
        if self.is_time_up() { return (self.evaluate(state), None); }

        if let Some((sc, bm)) = self.tt.lookup(state, depth_remaining, alpha, beta) {
            return (sc, Some(bm));
        }
        let tt_best = self.tt.get_best_move(state);

        if Self::is_terminal(state) || depth_remaining == 0 {
            let val = if state.terminated { Self::terminal_value(state) } else { self.evaluate(state) };
            self.tt.store(state, depth_remaining, val, TTFlag::Exact, 0);
            return (val, None);
        }

        let legal = state.get_legal_actions();
        if legal.is_empty() {
            let val = self.evaluate(state);
            self.tt.store(state, depth_remaining, val, TTFlag::Exact, 0);
            return (val, None);
        }

        let children = self.order_children(state, &legal, tt_best, depth_from_root);

        let mut best_action = children[0].action;
        let mut best_score = f32::NEG_INFINITY;
        let mut flag = TTFlag::Upper;

        for (i, child) in children.iter().enumerate() {
            if self.is_time_up() { break; }

            // Use pre-cloned child state (from NNUE ordering), or clone+step (heuristic)
            if let Some(ref cs) = child.child_state {
                *state = cs.clone();
            } else {
                state.step(child.action);
            }

            let score = if child.is_terminal {
                child.reward + Self::terminal_value(state)
            } else if depth_remaining == 1 {
                child.reward + self.evaluate(state)
            } else if i == 0 || is_pv {
                let (sc, _) = self.negamax(state, depth_remaining - 1, depth_from_root + 1,
                                          -beta, -alpha, true);
                child.reward + (-sc)
            } else if depth_remaining > 2 && i >= self.config.lmr_moves as usize {
                let r = (depth_remaining - 1).saturating_sub(self.config.lmr_depth).max(1);
                let (sc, _) = self.negamax(state, r, depth_from_root + 1,
                                          -alpha - 1e-10, -alpha, false);
                let mut sc = -sc;
                if sc > alpha && sc < beta {
                    let (sc2, _) = self.negamax(state, depth_remaining - 1, depth_from_root + 1,
                                               -beta, -alpha, false);
                    sc = -sc2;
                }
                child.reward + sc
            } else {
                let (sc, _) = self.negamax(state, depth_remaining - 1, depth_from_root + 1,
                                          -alpha - 1e-10, -alpha, false);
                let mut sc = -sc;
                if sc > alpha && sc < beta {
                    let (sc2, _) = self.negamax(state, depth_remaining - 1, depth_from_root + 1,
                                               -beta, -alpha, false);
                    sc = -sc2;
                }
                child.reward + sc
            };

            if score > best_score {
                best_score = score;
                best_action = child.action;
            }
            alpha = alpha.max(best_score);
            if alpha >= beta {
                flag = TTFlag::Lower;
                // History: update the move that caused cutoff
                if !is_pv {
                    self.history.update(child.action, depth_remaining);
                    self.killers.add(depth_from_root as usize, child.action);
                }
                break;
            }
        }

        self.tt.store(state, depth_remaining, best_score, flag, best_action);
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

        // Root ordering: always NNUE (depth_from_root=0 triggers NNUE in order_children)
        let root_children = self.order_children(state, &legal, None, 0);
        let mut best_action = root_children[0].action;
        let mut best_score = 0.0f32;
        let mut depth_reached = 0u8;

        if self.config.max_depth == 0 {
            return SearchResult {
                best_action, score: best_score,
                nodes_searched: self.nodes_searched, depth_reached: 0,
                time_ms: self.start_time.elapsed().as_millis() as u64,
            };
        }

        for depth in 1..=self.config.max_depth {
            if self.is_time_up() { break; }

            let (alpha, beta) = if depth >= 3 && best_score.abs() > 0.01 {
                let w = (0.5 - 0.04 * depth as f32).max(0.1);
                ((best_score - w).max(-1.0), (best_score + w).min(1.0))
            } else {
                (-1.0, 1.0)
            };

            let mut result = self.negamax(&mut state.clone(), depth, 1, alpha, beta, true);

            // Aspiration fail-low → full window
            if result.0 <= alpha {
                result = self.negamax(&mut state.clone(), depth, 1, -1.0, beta, true);
            }
            // Aspiration fail-high → full window
            if result.0 >= beta {
                result = self.negamax(&mut state.clone(), depth, 1, alpha, 1.0, true);
            }

            if let Some(a) = result.1 {
                best_action = a;
                best_score = result.0;
            }
            depth_reached = depth;
        }

        SearchResult {
            best_action, score: best_score,
            nodes_searched: self.nodes_searched, depth_reached,
            time_ms: self.start_time.elapsed().as_millis() as u64,
        }
    }
}
