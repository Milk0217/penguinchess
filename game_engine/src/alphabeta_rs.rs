use crate::nnue_rs::NNUEWeights;
use crate::rules::{GameState, Phase};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

const PIECE_IDS: [i32; 6] = [4, 6, 8, 5, 7, 9];
const EVAL_BATCH_STRIDE: usize = 75;
const MAX_LEGAL: usize = 60;
const MAX_DEPTH: usize = 64;

pub type EvalBatchFn = unsafe extern "C" fn(data: *const f32, batch_size: i32, scores_out: *mut f32) -> i32;
static mut EVAL_CALLBACK: Option<EvalBatchFn> = None;
pub fn set_eval_callback(cb: EvalBatchFn) { unsafe { EVAL_CALLBACK = Some(cb); } }

fn encode_for_batch(state: &GameState, buf: &mut [f32]) {
    let sparse = extract_sparse(state);
    for i in 0..8 { buf[i] = if i < sparse.len() { sparse[i] as f32 } else { -1.0 }; }
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
    let phase = if matches!(state.phase, Phase::Movement) { 0.5 } else { 0.0 };
    let p1_a = state.pieces.iter().filter(|p| p.owner() == 0 && p.alive).count() as f32 / 3.0;
    let p2_a = state.pieces.iter().filter(|p| p.owner() == 1 && p.alive).count() as f32 / 3.0;
    dense.extend_from_slice(&[state.scores[0] as f32 / 100.0, state.scores[1] as f32 / 100.0, phase, p1_a, p2_a, state.episode_steps as f32 / 500.0]);
    dense
}

// ─── Transposition Table (thread-safe) ────────────────────────

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum TTFlag { Exact, Lower, Upper }
#[allow(dead_code)]
struct TTEntry { depth: u8, score: f32, flag: TTFlag, best_move: usize, age: u32 }

pub struct TranspositionTable {
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
    pub fn new(max_size: usize) -> Self { Self { entries: HashMap::with_capacity(max_size.min(1 << 20)), age: 0, max_size } }
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

pub struct SharedTT { inner: RwLock<TranspositionTable> }

impl SharedTT {
    pub fn new(max_size: usize) -> Self { Self { inner: RwLock::new(TranspositionTable::new(max_size)) } }
    pub fn lookup(&self, state: &GameState, depth: u8, alpha: f32, beta: f32) -> Option<(f32, usize)> {
        self.inner.read().ok()?.lookup(state, depth, alpha, beta)
    }
    pub fn get_best_move(&self, state: &GameState) -> Option<usize> {
        self.inner.read().ok()?.get_best_move(state)
    }
    pub(crate) fn store(&self, state: &GameState, depth: u8, score: f32, flag: TTFlag, best_move: usize) {
        if let Ok(mut tt) = self.inner.write() { tt.store(state, depth, score, flag, best_move); }
    }
    pub fn new_search(&self) {
        if let Ok(mut tt) = self.inner.write() { tt.new_search(); }
    }
}

// ─── History + Killers ────────────────────────────────────────

struct HistoryTable { scores: [i32; MAX_LEGAL], max_: i32 }
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

struct KillerTable { moves: Vec<[Option<usize>; 2]> }
impl KillerTable {
    fn new() -> Self { Self { moves: vec![[None, None]; MAX_DEPTH] } }
    fn add(&mut self, depth: usize, action: usize) {
        if depth >= MAX_DEPTH { return; }
        let s = &mut self.moves[depth];
        if s[0] != Some(action) { s[1] = s[0]; s[0] = Some(action); }
    }
    fn is_killer(&self, depth: usize, action: usize) -> bool {
        self.moves.get(depth).map_or(false, |s| s[0] == Some(action) || s[1] == Some(action))
    }
    fn clear(&mut self) { self.moves.fill([None, None]); }
}

struct OrderedChild { action: usize, reward: f32, is_terminal: bool, child_state: Option<GameState>, sort_score: f32 }

// ─── Config & Result ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    pub max_depth: u8,
    pub time_limit_ms: u64,
    pub tt_size: usize,
    pub lmr_moves: u8,
    pub lmr_depth: u8,
    pub nnue_order_depth: u8,
    pub root_split: bool,
    pub num_threads: usize,
    pub null_move: bool,
    pub reuse: bool,
    pub epsilon: f32,  // epsilon-greedy exploration for data gen (0.0 = greedy)
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self { max_depth: 6, time_limit_ms: 0, tt_size: 1 << 20, lmr_moves: 3, lmr_depth: 1, nnue_order_depth: 2, num_threads: 1, null_move: true, root_split: false, reuse: false, epsilon: 0.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub best_action: usize, pub score: f32,
    pub nodes_searched: u64, pub depth_reached: u8, pub time_ms: u64,
}

// ─── Per-Thread Search Context ─────────────────────────────────

struct SearchContext<'a> {
    tt: &'a SharedTT,
    history: HistoryTable,
    killers: KillerTable,
    nodes_searched: u64,
    start_time: std::time::Instant,
    weights: &'a NNUEWeights,
    config: &'a SearchConfig,
    _thread_id: u32,
}

fn evaluate_static(state: &GameState, weights: &NNUEWeights) -> f32 {
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
    crate::nnue_rs::nnue_evaluate(&sparse, &dense, state.current_player, weights)
}

fn batch_evaluate_static(states: &[&GameState], weights: &NNUEWeights) -> Vec<f32> {
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
    states.iter().map(|s| evaluate_static(s, weights)).collect()
}

impl<'a> SearchContext<'a> {
    fn new(tt: &'a SharedTT, weights: &'a NNUEWeights, config: &'a SearchConfig, thread_id: u32) -> Self {
        Self { tt, history: HistoryTable::new(), killers: KillerTable::new(), nodes_searched: 0, start_time: std::time::Instant::now(), weights, config, _thread_id: thread_id }
    }

    fn is_time_up(&self) -> bool {
        self.config.time_limit_ms > 0 && self.start_time.elapsed().as_millis() as u64 >= self.config.time_limit_ms
    }

    fn terminal_value(state: &GameState) -> f32 {
        if !state.terminated { return 0.0; }
        if state.scores[0] > state.scores[1] { 1.0 } else if state.scores[1] > state.scores[0] { -1.0 } else { 0.0 }
    }

    fn evaluate(&self, state: &GameState) -> f32 {
        evaluate_static(state, self.weights)
    }

    fn batch_evaluate(&self, states: &[&GameState]) -> Vec<f32> {
        batch_evaluate_static(states, self.weights)
    }

    fn is_terminal(state: &GameState) -> bool {
        state.terminated || state.get_legal_actions().is_empty()
    }

    fn nnue_ordering_depth(&self, state: &GameState) -> u8 {
        let base = self.config.nnue_order_depth;
        if base == 0 { return 0; }
        let alive = state.pieces.iter().filter(|p| p.alive).count();
        let has_high = state.board.cells.iter().any(|c| c.state == crate::board::HexState::Active && c.points >= 3);
        if has_high && alive >= 4 { base } else if alive >= 3 { base.min(2) } else { 1u8.min(base) }
    }

    fn order_children(&mut self, state: &GameState, legal: &[usize], tt_best: Option<usize>, depth_from_root: u8) -> Vec<OrderedChild> {
        if legal.is_empty() { return vec![]; }
        if legal.len() == 1 { return vec![OrderedChild { action: legal[0], reward: 0.0, is_terminal: false, child_state: None, sort_score: 0.0 }]; }
        let use_nnue = depth_from_root == 0 || depth_from_root <= self.nnue_ordering_depth(state);
        if use_nnue {
            let mut children: Vec<OrderedChild> = Vec::with_capacity(legal.len());
            for &a in legal {
                let mut snap = state.clone();
                let (r, t) = snap.step(a);
                children.push(OrderedChild { action: a, reward: r, is_terminal: t, child_state: Some(snap), sort_score: 0.0 });
            }
            let child_refs: Vec<&GameState> = children.iter().filter_map(|c| c.child_state.as_ref()).collect();
            let evals = self.batch_evaluate(&child_refs);
            for (i, child) in children.iter_mut().enumerate() {
                let ev = if child.is_terminal { 0.0 } else { evals[i] };
                child.sort_score = child.reward + if child.is_terminal { Self::terminal_value(child.child_state.as_ref().unwrap()) } else { ev };
            }
            children.sort_by(|a, b| b.sort_score.partial_cmp(&a.sort_score).unwrap_or(std::cmp::Ordering::Equal));
            children
        } else {
            let mut children: Vec<OrderedChild> = Vec::with_capacity(legal.len());
            for &a in legal {
                let mut h = 0i64;
                if Some(a) == tt_best { h += 10_000_000; }
                if self.killers.is_killer(depth_from_root as usize, a) { h += 5_000_000; }
                h += self.history.get(a) as i64;
                children.push(OrderedChild { action: a, reward: 0.0, is_terminal: false, child_state: None, sort_score: h as f32 });
            }
            children.sort_by(|a, b| b.sort_score.partial_cmp(&a.sort_score).unwrap_or(std::cmp::Ordering::Equal));
            children
        }
    }

    fn negamax(&mut self, state: &mut GameState, depth_remaining: u8, depth_from_root: u8, mut alpha: f32, beta: f32, is_pv: bool) -> (f32, Option<usize>) {
        self.nodes_searched += 1;
        if self.is_time_up() { return (self.evaluate(state), None); }

        if let Some((sc, bm)) = self.tt.lookup(state, depth_remaining, alpha, beta) { return (sc, Some(bm)); }
        let mut tt_best = self.tt.get_best_move(state);
        // ── Internal Iterative Deepening ──
        // If TT miss at deep non-PV node, do a shallow search to get ordering info.
        if tt_best.is_none() && depth_remaining >= 4 && !is_pv && !Self::is_terminal(state) {
            let iid_target = (depth_remaining - 2).max(1);
            let mut clone = state.clone();
            self.negamax(&mut clone, iid_target, depth_from_root + 1, -beta, -alpha, false);
            tt_best = self.tt.get_best_move(state).or(tt_best);
        }

        if Self::is_terminal(state) || depth_remaining == 0 {
            let val = if state.terminated { Self::terminal_value(state) } else { self.evaluate(state) };
            self.tt.store(state, depth_remaining, val, TTFlag::Exact, 0);
            return (val, None);
        }

        // ── Null-move pruning ──
        if self.config.null_move && !is_pv && depth_remaining >= 3
            && matches!(state.phase, Phase::Movement)
            && state.pieces.iter().filter(|p| p.alive).count() >= 2
        {
            let r = 2u8;
            let mut null_state = state.clone();
            null_state.current_player = 1 - null_state.current_player;
            let (sc, _) = self.negamax(&mut null_state, depth_remaining - 1 - r, depth_from_root + 1, -beta, -beta + 1.0, false);
            if -sc >= beta { return (beta, None); }
        }

        let legal = state.get_legal_actions();
        if legal.is_empty() { let val = self.evaluate(state); self.tt.store(state, depth_remaining, val, TTFlag::Exact, 0); return (val, None); }

        let children = self.order_children(state, &legal, tt_best, depth_from_root);
        let mut best_action = children[0].action;
        let mut best_score = f32::NEG_INFINITY;
        let mut flag = TTFlag::Upper;

        for (i, child) in children.iter().enumerate() {
            if self.is_time_up() { break; }
            if let Some(ref cs) = child.child_state { *state = cs.clone(); }
            else { state.step(child.action); }

            let score = if child.is_terminal { child.reward + Self::terminal_value(state) }
            else if depth_remaining == 1 { child.reward + self.evaluate(state) }
            else if i == 0 || is_pv {
                let (sc, _) = self.negamax(state, depth_remaining - 1, depth_from_root + 1, -beta, -alpha, true);
                child.reward + (-sc)
            } else if depth_remaining > 2 && i >= self.config.lmr_moves as usize {
                let base_r = self.config.lmr_depth as u32;
                // More aggressive reduction for very late moves at deep depths
                let extra_r = if depth_remaining >= 6 && i >= 8 { 1u32 } else { 0u32 };
                let reduction = (base_r + extra_r).min((depth_remaining.saturating_sub(2)) as u32);
                let r = (depth_remaining - 1).saturating_sub(reduction as u8).max(1);
                let (mut sc, _) = self.negamax(state, r, depth_from_root + 1, -alpha - 1e-10, -alpha, false);
                sc = -sc;
                if sc > alpha && sc < beta {
                    let (sc2, _) = self.negamax(state, depth_remaining - 1, depth_from_root + 1, -beta, -alpha, false);
                    sc = -sc2;
                }
                child.reward + sc
            } else {
                let (mut sc, _) = self.negamax(state, depth_remaining - 1, depth_from_root + 1, -alpha - 1e-10, -alpha, false);
                sc = -sc;
                if sc > alpha && sc < beta {
                    let (sc2, _) = self.negamax(state, depth_remaining - 1, depth_from_root + 1, -beta, -alpha, false);
                    sc = -sc2;
                }
                child.reward + sc
            };

            if score > best_score { best_score = score; best_action = child.action; }
            alpha = alpha.max(best_score);
            if alpha >= beta {
                flag = TTFlag::Lower;
                if !is_pv { self.history.update(child.action, depth_remaining); self.killers.add(depth_from_root as usize, child.action); }
                break;
            }
        }
        self.tt.store(state, depth_remaining, best_score, flag, best_action);
        (best_score, Some(best_action))
    }

    fn search_root(&mut self, state: &GameState) -> SearchResult {
        self.tt.new_search();
        self.history.clear();
        self.killers.clear();
        self.nodes_searched = 0;
        self.start_time = std::time::Instant::now();

        let legal = state.get_legal_actions();
        if legal.is_empty() { return SearchResult { best_action: 0, score: 0.0, nodes_searched: 0, depth_reached: 0, time_ms: 0 }; }
        if legal.len() == 1 { return SearchResult { best_action: legal[0], score: 0.0, nodes_searched: 1, depth_reached: 0, time_ms: 0 }; }

        let root_children = self.order_children(state, &legal, None, 0);
        let mut best_action = root_children[0].action;
        let mut best_score = 0.0f32;
        let mut depth_reached = 0u8;

        if self.config.max_depth == 0 {
            return SearchResult { best_action, score: best_score, nodes_searched: self.nodes_searched, depth_reached: 0, time_ms: 0 };
        }

        for depth in 1..=self.config.max_depth {
            if self.is_time_up() { break; }
            let (alpha, beta) = if depth >= 3 && best_score.abs() > 0.01 {
                let w = (0.5 - 0.04 * depth as f32).max(0.1);
                ((best_score - w).max(-1.0), (best_score + w).min(1.0))
            } else { (-1.0, 1.0) };

            let mut result = self.negamax(&mut state.clone(), depth, 1, alpha, beta, true);
            if result.0 <= alpha { result = self.negamax(&mut state.clone(), depth, 1, -1.0, beta, true); }
            if result.0 >= beta { result = self.negamax(&mut state.clone(), depth, 1, alpha, 1.0, true); }
            if let Some(a) = result.1 { best_action = a; best_score = result.0; }
            depth_reached = depth;
        }
        SearchResult { best_action, score: best_score, nodes_searched: self.nodes_searched, depth_reached, time_ms: self.start_time.elapsed().as_millis() as u64 }
    }
}

// ─── Public API ───────────────────────────────────────────────

pub struct AlphaBetaSearch {
    pub weights: NNUEWeights,
    pub config: SearchConfig,
    pub persistent_tt: Option<SharedTT>,
}

fn pack_move_score(mov: usize, score: f32) -> u64 {
    ((score.to_bits() as u64) << 32) | (mov as u64 & 0xFFFF_FFFF)
}

fn unpack_move(packed: u64) -> usize { (packed & 0xFFFF_FFFF) as usize }
fn unpack_score(packed: u64) -> f32 { f32::from_bits((packed >> 32) as u32) }

impl AlphaBetaSearch {
    pub fn new(weights: NNUEWeights, config: SearchConfig) -> Self {
        Self { weights, config, persistent_tt: None }
    }

    /// Full search - TT persists across calls if config.reuse is true.
    pub fn search(&mut self, state: &GameState) -> SearchResult {
        let tt = self.persistent_tt.get_or_insert_with(|| SharedTT::new(self.config.tt_size));
        if !self.config.reuse { tt.inner.write().unwrap().new_search(); }
        if self.config.num_threads <= 1 {
            let mut ctx = SearchContext::new(tt, &self.weights, &self.config, 0);
            ctx.search_root(state)
        } else if self.config.root_split {
            self.search_root_split(state)
        } else {
            self.search_lazy_smp(state)
        }
    }

    /// Root-level parallelism: divide root moves among threads.
    /// Each thread has its OWN TT — no RwLock contention.
    pub fn search_root_split(&self, state: &GameState) -> SearchResult {
        let num_threads = self.config.num_threads.max(1) as usize;
        let start = std::time::Instant::now();
        let best = AtomicU64::new(pack_move_score(0, f32::NEG_INFINITY));

        // NNUE order all root moves on the calling thread
        let legal = state.get_legal_actions();
        if legal.is_empty() {
            return SearchResult { best_action: 0, score: 0.0, nodes_searched: 0, depth_reached: 0, time_ms: 0 };
        }
        if legal.len() == 1 {
            return SearchResult { best_action: legal[0], score: 0.0, nodes_searched: 1, depth_reached: 0, time_ms: 0 };
        }

        // Create a temporary context just for root ordering
        let ord_tt = SharedTT::new(256);
        let mut ord_ctx = SearchContext::new(&ord_tt, &self.weights, &self.config, 0);
        let root_children = ord_ctx.order_children(state, &legal, None, 0);
        let n_children = root_children.len();
        let children_ref = &root_children;

        std::thread::scope(|s| {
            for tid in 0..num_threads {
                let best_ref = &best;
                s.spawn(move || {
                    let tt_size = (self.config.tt_size / num_threads).max(64);
                    let tt = SharedTT::new(tt_size);
                    let mut ctx = SearchContext::new(&tt, &self.weights, &self.config, tid as u32);

                    for depth in 1..=self.config.max_depth {
                        if ctx.is_time_up() { break; }
                        let (alpha, beta) = (-1.0, 1.0);

                        for idx in (tid..n_children).step_by(num_threads) {
                            if ctx.is_time_up() { break; }
                            let mut child_state = state.clone();
                            if let Some(ref cs) = children_ref[idx].child_state {
                                child_state = cs.clone();
                            } else {
                                child_state.step(children_ref[idx].action);
                            }

                            let (score, _) = ctx.negamax(&mut child_state, depth, 1, alpha, beta, true);
                            let packed = pack_move_score(children_ref[idx].action, score);
                            let prev = best_ref.load(Ordering::Relaxed);
                            if score > unpack_score(prev) {
                                best_ref.store(packed, Ordering::Relaxed);
                            }
                        }
                    }
                });
            }
        });

        let packed = best.load(Ordering::Relaxed);
        SearchResult {
            best_action: unpack_move(packed),
            score: unpack_score(packed),
            nodes_searched: 0,
            depth_reached: self.config.max_depth,
            time_ms: start.elapsed().as_millis() as u64,
        }
    }

    pub fn search_lazy_smp(&self, state: &GameState) -> SearchResult {
        let tt = SharedTT::new(self.config.tt_size);
        let best = AtomicU64::new(pack_move_score(0, f32::NEG_INFINITY));
        let num_threads = self.config.num_threads.max(1) as usize;
        let start = std::time::Instant::now();
        let weights = &self.weights;
        let config = &self.config;

        std::thread::scope(|s| {
            for tid in 0..num_threads {
                let state_clone = state.clone();
                let best_ref = &best;
                let tt_ref = &tt;
                s.spawn(move || {
                    let mut ctx = SearchContext::new(tt_ref, weights, config, tid as u32);
                    let result = ctx.search_root(&state_clone);
                    let packed = pack_move_score(result.best_action, result.score);
                    let prev = best_ref.load(Ordering::Relaxed);
                    if result.score > unpack_score(prev) {
                        best_ref.store(packed, Ordering::Relaxed);
                    }
                });
            }
        });

        let packed = best.load(Ordering::Relaxed);
        SearchResult {
            best_action: unpack_move(packed),
            score: unpack_score(packed),
            nodes_searched: 0, // not easily aggregated
            depth_reached: self.config.max_depth,
            time_ms: start.elapsed().as_millis() as u64,
        }
    }
}
