/// Rust MCTS 搜索 — 支持 Python 回调和 ONNX 两种 eval 模式
use std::collections::HashMap;
use std::os::raw::c_char;
use rand::Rng;
use rand_distr::{Dirichlet, Distribution};
use crate::board::*;
use crate::rules::*;
#[cfg(feature = "ort")]
use crate::net_infer::NetInfer;


pub type EvalFn = extern "C" fn(*const f32, i32, *mut f32, i32) -> i32;

const DIRICHLET_ALPHA: f64 = 0.15;
const DIRICHLET_EPS: f64 = 0.25;

struct MCTSNode {
    visits: u32,
    total_value: f64,
    prior: f64,
    children: HashMap<usize, MCTSNode>,
}

impl MCTSNode {
    fn new(prior: f64) -> Self {
        MCTSNode { visits: 0, total_value: 0.0, prior, children: HashMap::new() }
    }
    fn ucb(&self, parent_visits: u32, c_puct: f64) -> f64 {
        if self.visits == 0 { return f64::INFINITY; }
        let q = self.total_value / self.visits as f64;
        let u = c_puct * self.prior * (parent_visits as f64).sqrt() / (1.0 + self.visits as f64);
        q + u
    }
}

fn node_by_path<'a>(root: &'a mut MCTSNode, path: &[usize]) -> &'a mut MCTSNode {
    let mut node = root;
    for &a in path { node = node.children.get_mut(&a).unwrap(); }
    node
}

// =============================================================================
// 编码观测：将 GameState 编码为 Flat float 数组 (206 dims)
// =============================================================================

fn encode_obs(state: &GameState, buf: &mut Vec<f32>) {
    // board: 60 cells × [q/8, r/8, value/3]
    for cell in &state.board.cells {
        let pts = if cell.state == HexState::Active || cell.state == HexState::Occupied {
            cell.points as f32 / 3.0
        } else {
            0.0
        };
        buf.push(cell.coord.q as f32 / 8.0);
        buf.push(cell.coord.r as f32 / 8.0);
        buf.push(pts);
    }
    // pieces: 6 pieces × [id/10, q/8, r/8, s/8]
    for piece in &state.pieces {
        if piece.alive && piece.hex_idx.is_some() {
            let cell = &state.board.cells[piece.hex_idx.unwrap()];
            buf.push(piece.id as f32 / 10.0);
            buf.push(cell.coord.q as f32 / 8.0);
            buf.push(cell.coord.r as f32 / 8.0);
            buf.push(cell.coord.s as f32 / 8.0);
        } else {
            buf.extend_from_slice(&[-1.0, 0.0, 0.0, 0.0]);
        }
    }
    // meta: [current_player, phase]
    buf.push(state.current_player as f32);
    buf.push(if matches!(state.phase, Phase::Placement) { 0.0 } else { 1.0 });
}

fn obs_dim() -> usize { 206 }
fn out_dim() -> usize { 61 }  // 60 logits + 1 value

// =============================================================================
// Dirichlet noise sampler
// =============================================================================

fn sample_dirichlet(alpha: f64, n: usize) -> Vec<f64> {
    let alphas = vec![alpha; n];
    let d = Dirichlet::new(&alphas).unwrap();
    let mut rng = rand::thread_rng();
    d.sample(&mut rng)
}

// =============================================================================
// Softmax + mask helper
// =============================================================================

fn masked_softmax(logits: &[f32], legal: &[usize]) -> Vec<f64> {
    let mut masked = vec![-1e9_f64; 60];
    for &a in legal {
        masked[a] = logits[a] as f64;
    }
    let max_l = masked.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = masked.iter().map(|x| (x - max_l).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 1e-12 {
        exps.iter().map(|x| x / sum).collect()
    } else {
        let mut v = vec![0.0; 60];
        for &a in legal { v[a] = 1.0 / legal.len() as f64; }
        v
    }
}

// =============================================================================
// 主搜索函数 (Core: 接受 &GameState, 返回 visit count JSON)
// =============================================================================

fn mcts_search_core(
    state: &GameState,
    num_simulations: i32,
    c_puct: f64,
    batch_size: i32,
    eval_fn: Option<EvalFn>,
) -> String {
    let mut root = MCTSNode::new(0.0);
    let bs = batch_size.max(1) as usize;

    // ----- Pre-expand root with Dirichlet noise -----
    let mut root_obs = Vec::with_capacity(obs_dim());
    encode_obs(state, &mut root_obs);
    let legal_root = state.get_legal_actions();

    if legal_root.is_empty() {
        return "{}".to_string();
    }

    if let Some(f) = eval_fn {
        let mut root_out = vec![0.0f32; out_dim()];
        unsafe { f(root_obs.as_ptr(), 1, root_out.as_mut_ptr(), root_out.len() as i32); }
        let policy = masked_softmax(&root_out[..60], &legal_root);
        let noise = sample_dirichlet(DIRICHLET_ALPHA, 60);
        for &a in &legal_root {
            let noisy_prior = (1.0 - DIRICHLET_EPS) * policy[a] + DIRICHLET_EPS * noise[a];
            root.children.insert(a, MCTSNode::new(noisy_prior));
        }
    } else {
        let uniform = 1.0 / legal_root.len() as f64;
        for &a in &legal_root {
            root.children.insert(a, MCTSNode::new(uniform));
        }
    }

    let mut pending_states: Vec<GameState> = vec![];
    let mut pending_legal: Vec<Vec<usize>> = vec![];
    let mut pending_paths: Vec<Vec<usize>> = vec![];
    let mut obs_buf: Vec<f32> = Vec::with_capacity(bs * obs_dim());
    let mut out_buf: Vec<f32> = Vec::with_capacity(bs * out_dim());

    for _ in 0..num_simulations.max(1) {
        let mut state_clone = state.clone();
        let mut path: Vec<usize> = vec![];

        // SELECT
        loop {
            let parent_visits = {
                let n = node_by_path(&mut root, &path);
                if n.children.is_empty() { break; }
                n.visits.max(1)
            };
            let best = {
                let n = node_by_path(&mut root, &path);
                let mut rng = rand::thread_rng();
                n.children.iter()
                    .map(|(a, child)| (a, child.ucb(parent_visits, c_puct) + rng.gen::<f64>() * 1e-12))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(a, _)| *a).unwrap()
            };
            state_clone.step(best);
            path.push(best);
            if state_clone.terminated { break; }
        }

        let terminated = state_clone.terminated;
        let legal = state_clone.get_legal_actions();

        if terminated {
            backup(&mut root, &path, terminal_value(&state_clone));
        } else if !legal.is_empty() {
            pending_states.push(state_clone);
            pending_legal.push(legal);
            pending_paths.push(path);
            if pending_states.len() >= bs {
                flush_batch_obs(&mut root, &mut pending_states, &mut pending_legal,
                                &mut pending_paths, eval_fn, &mut obs_buf, &mut out_buf);
            }
        }
    }
    if !pending_states.is_empty() {
        flush_batch_obs(&mut root, &mut pending_states, &mut pending_legal,
                        &mut pending_paths, eval_fn, &mut obs_buf, &mut out_buf);
    }

    let mut out = serde_json::Map::new();
    for (a, c) in &root.children {
        out.insert(a.to_string(), serde_json::Value::Number(serde_json::Number::from(c.visits)));
    }
    serde_json::to_string(&out).unwrap_or_default()
}

/// JSON entry point: deserialize state from JSON, run search, write output.
unsafe fn mcts_search_internal(
    state_json: *const c_char,
    num_simulations: i32,
    c_puct: f64,
    batch_size: i32,
    eval_fn: Option<EvalFn>,
    output_buf: *mut c_char,
    output_size: i32,
) -> i32 {
    let input = match std::ffi::CStr::from_ptr(state_json).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let mut state: GameState = match serde_json::from_str(input) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    state.board.rebuild_index_for_json();
    let result = mcts_search_core(&state, num_simulations, c_puct, batch_size, eval_fn);
    write_output(output_buf, output_size, &result);
    0
}

/// Exported: MCTS with Python eval callback (JSON input).
#[no_mangle]
pub unsafe extern "C" fn mcts_search_rust(
    state_json: *const c_char,
    num_simulations: i32,
    c_puct: f64,
    batch_size: i32,
    eval_fn: Option<EvalFn>,
    output_buf: *mut c_char,
    output_size: i32,
) -> i32 {
    mcts_search_internal(state_json, num_simulations, c_puct, batch_size,
                         eval_fn, output_buf, output_size)
}

/// Handle-based MCTS search: read state from GAMES[handle], run search.
/// This skips the JSON serialization/deserialization roundtrip.
/// Defined here and re-exported by ffi.rs.
pub unsafe fn mcts_search_on_handle(
    state: &GameState,
    num_simulations: i32,
    c_puct: f64,
    batch_size: i32,
    eval_fn: Option<EvalFn>,
) -> String {
    mcts_search_core(state, num_simulations, c_puct, batch_size, eval_fn)
}

// =============================================================================
// Batch flush
// =============================================================================
// Batch flush with float obs array (zero-copy to Python/NumPy)
// =============================================================================

fn flush_batch_obs(
    root: &mut MCTSNode,
    states: &mut Vec<GameState>,
    legal_list: &mut Vec<Vec<usize>>,
    paths: &mut Vec<Vec<usize>>,
    eval_fn: Option<EvalFn>,
    obs_buf: &mut Vec<f32>,
    out_buf: &mut Vec<f32>,
) {
    let n = states.len();
    obs_buf.clear();
    for state in states.iter() {
        encode_obs(state, obs_buf);
    }
    debug_assert_eq!(obs_buf.len(), n * obs_dim());

    if let Some(f) = eval_fn {
        out_buf.clear();
        out_buf.resize(n * out_dim(), 0.0f32);
        unsafe { f(obs_buf.as_ptr(), n as i32, out_buf.as_mut_ptr(), out_buf.len() as i32); }
    } else {
        // Uniform: fill out_buf with zeros (value=0 → uniform backup)
        out_buf.clear();
        out_buf.resize(n * out_dim(), 0.0f32);
    }

    for (i, (path, legal)) in paths.iter().zip(legal_list.iter()).enumerate() {
        let base = i * out_dim();
        let logits = &out_buf[base..base + 60];
        let value = out_buf[base + 60] as f64;
        let policy = masked_softmax(logits, legal);
        let node = node_by_path(root, path);
        for &a in legal {
            node.children.entry(a).or_insert(MCTSNode::new(policy[a]));
        }
        backup(root, path, value);
    }
    // NOTE: Uniform case (eval_fn=None and use_onnx=false) falls through here
    // with out_buf all zeros → masked_softmax produces uniform policy, backup uses 0 value
    // This matches the old behavior.

    states.clear();
    legal_list.clear();
    paths.clear();
}

fn backup(root: &mut MCTSNode, path: &[usize], mut value: f64) {
    for depth in (0..=path.len()).rev() {
        let n = node_by_path(root, &path[..depth]);
        n.visits += 1; n.total_value += value; value = -value;
    }
}

fn terminal_value(state: &GameState) -> f64 {
    if !state.terminated { return 0.0; }
    let cp = state.current_player;
    let (s0, s1) = (state.scores[0], state.scores[1]);
    if s0 > s1 { if cp == 0 { 1.0 } else { -1.0 } }
    else if s1 > s0 { if cp == 1 { 1.0 } else { -1.0 } }
    else { 0.0 }
}

fn write_output(buf: *mut c_char, size: i32, data: &str) {
    if size <= 0 { return; }
    let bytes = data.as_bytes();
    let len = bytes.len().min((size - 1) as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, len);
        *buf.add(len) = 0;
    }
}

// ────────────────────────────────────────────────────────────
// Parallel MCTS: splits simulations across N internal threads
// ────────────────────────────────────────────────────────────

pub fn mcts_search_parallel_core(
    state: &GameState,
    num_simulations: i32,
    c_puct: f64,
    batch_size: i32,
    eval_fn: Option<EvalFn>,
    num_workers: usize,
) -> String {
    if num_workers <= 1 {
        return mcts_search_core(state, num_simulations, c_puct, batch_size, eval_fn);
    }

    let sims_per = std::cmp::max(1, num_simulations / num_workers as i32);

    // Clone the shared state ONCE before spawning threads (avoids concurrent reads of static mut GAMES)
    let base_state = state.clone();

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let state_clone = base_state.clone();
            handles.push(s.spawn(move || {
                mcts_search_core(&state_clone, sims_per, c_puct, batch_size, eval_fn)
            }));
        }

        use std::collections::HashMap;
        let mut merged: HashMap<usize, u32> = HashMap::new();
        for h in handles {
            let json_str = h.join().unwrap();
            if let Ok(partial) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&json_str) {
                for (k, v) in partial {
                    if let Ok(action) = k.parse::<usize>() {
                        if let Some(visits) = v.as_u64() {
                            *merged.entry(action).or_insert(0) += visits as u32;
                        }
                    }
                }
            }
        }

        let out_map: serde_json::Map<String, serde_json::Value> = merged
            .iter()
            .map(|(k, v)| (k.to_string(), serde_json::Value::Number(serde_json::Number::from(*v))))
            .collect();
        serde_json::to_string(&out_map).unwrap_or_default()
    })
}

// ═══════════════════════════════════════════════════════════
// ONNX Runtime inference path (no Python callback)
// Disabled by default — Python callback is ~10x faster for small batches.
// Enable with: cargo build --release --features ort
// ═══════════════════════════════════════════════════════════
#[cfg(feature = "ort")]
fn flush_batch_obs_ort(
    root: &mut MCTSNode,
    states: &mut Vec<GameState>,
    legal_list: &mut Vec<Vec<usize>>,
    paths: &mut Vec<Vec<usize>>,
    model: &NetInfer,
    obs_buf: &mut Vec<f32>,
) {
    let n = states.len();
    obs_buf.clear();
    for state in states.iter() {
        encode_obs(state, obs_buf);
    }
    debug_assert_eq!(obs_buf.len(), n * obs_dim());

    let (logits_matrix, values_vec) = model.evaluate_batch(obs_buf, n).unwrap_or_else(|_| {
        let uniform_logits = vec![vec![0.0f32; 60]; n];
        let zero_values = vec![0.0f32; n];
        (uniform_logits, zero_values)
    });

    for (i, (path, legal)) in paths.iter().zip(legal_list.iter()).enumerate() {
        let logits = &logits_matrix[i];
        let value = values_vec[i] as f64;
        let policy = masked_softmax(logits, legal);
        let node = node_by_path(root, path);
        for &a in legal {
            node.children.entry(a).or_insert(MCTSNode::new(policy[a]));
        }
        backup(root, path, value);
    }
    states.clear();
    legal_list.clear();
    paths.clear();
}

#[cfg(feature = "ort")]
pub fn mcts_search_core_ort(
    state: &GameState,
    num_simulations: i32,
    c_puct: f64,
    batch_size: i32,
    model: &NetInfer,
) -> String {
    let mut root = MCTSNode::new(0.0);
    let bs = batch_size.max(1) as usize;

    let mut root_obs = Vec::with_capacity(obs_dim());
    encode_obs(state, &mut root_obs);
    let legal_root = state.get_legal_actions();
    if legal_root.is_empty() {
        return "{}".to_string();
    }

    // Root evaluation + Dirichlet noise
    {
        let (logits_vec, _) = model.evaluate_batch(&root_obs, 1).unwrap_or_else(|_| {
            (vec![vec![0.0f32; 60]], vec![0.0f32])
        });
        let policy = masked_softmax(&logits_vec[0], &legal_root);
        let noise = sample_dirichlet(DIRICHLET_ALPHA, 60);
        for &a in &legal_root {
            let noisy_prior = (1.0 - DIRICHLET_EPS) * policy[a] + DIRICHLET_EPS * noise[a];
            root.children.insert(a, MCTSNode::new(noisy_prior));
        }
    }

    let mut pending_states: Vec<GameState> = vec![];
    let mut pending_legal: Vec<Vec<usize>> = vec![];
    let mut pending_paths: Vec<Vec<usize>> = vec![];
    let mut obs_buf: Vec<f32> = Vec::with_capacity(bs * obs_dim());

    for _ in 0..num_simulations.max(1) {
        let mut state_clone = state.clone();
        let mut path: Vec<usize> = vec![];

        // SELECT
        loop {
            let parent_visits = {
                let n = node_by_path(&mut root, &path);
                if n.children.is_empty() {
                    break;
                }
                n.visits.max(1)
            };
            let best = {
                let n = node_by_path(&mut root, &path);
                let mut rng = rand::thread_rng();
                n.children
                    .iter()
                    .map(|(a, child)| {
                        (a, child.ucb(parent_visits, c_puct) + rng.gen::<f64>() * 1e-12)
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(a, _)| *a)
                    .unwrap()
            };
            state_clone.step(best);
            path.push(best);
            if state_clone.terminated {
                break;
            }
        }

        let terminated = state_clone.terminated;
        let legal = state_clone.get_legal_actions();

        if terminated {
            backup(&mut root, &path, terminal_value(&state_clone));
        } else if !legal.is_empty() {
            pending_states.push(state_clone);
            pending_legal.push(legal);
            pending_paths.push(path);
            if pending_states.len() >= bs {
                flush_batch_obs_ort(
                    &mut root,
                    &mut pending_states,
                    &mut pending_legal,
                    &mut pending_paths,
                    model,
                    &mut obs_buf,
                );
            }
        }
    }

    if !pending_states.is_empty() {
        flush_batch_obs_ort(
            &mut root,
            &mut pending_states,
            &mut pending_legal,
            &mut pending_paths,
            model,
            &mut obs_buf,
        );
    }

    let mut out = serde_json::Map::new();
    for (a, c) in &root.children {
        out.insert(
            a.to_string(),
            serde_json::Value::Number(serde_json::Number::from(c.visits)),
        );
    }
    serde_json::to_string(&out).unwrap_or_default()
}

#[cfg(feature = "ort")]
pub unsafe fn mcts_search_on_handle_ort(
    state: &GameState,
    num_simulations: i32,
    c_puct: f64,
    batch_size: i32,
    model: &NetInfer,
) -> String {
    mcts_search_core_ort(state, num_simulations, c_puct, batch_size, model)
}

#[cfg(feature = "ort")]
pub fn mcts_search_parallel_core_ort(
    state: &GameState,
    num_simulations: i32,
    c_puct: f64,
    batch_size: i32,
    model: &NetInfer,
    num_workers: usize,
) -> String {
    if num_workers <= 1 {
        return mcts_search_core_ort(state, num_simulations, c_puct, batch_size, model);
    }

    let sims_per = std::cmp::max(1, num_simulations / num_workers as i32);
    let base_state = state.clone();

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let state_clone = base_state.clone();
            handles.push(s.spawn(move || {
                mcts_search_core_ort(&state_clone, sims_per, c_puct, batch_size, model)
            }));
        }

        let mut merged: HashMap<usize, u32> = HashMap::new();
        for h in handles {
            let json_str = h.join().unwrap();
            if let Ok(partial) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&json_str) {
                for (k, v) in partial {
                    if let Ok(action) = k.parse::<usize>() {
                        if let Some(visits) = v.as_u64() {
                            *merged.entry(action).or_insert(0) += visits as u32;
                        }
                    }
                }
            }
        }

        let out_map: serde_json::Map<String, serde_json::Value> = merged
            .iter()
            .map(|(k, v)| (k.to_string(), serde_json::Value::Number(serde_json::Number::from(*v))))
            .collect();
        serde_json::to_string(&out_map).unwrap_or_default()
    })
}
