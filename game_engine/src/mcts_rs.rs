/// Rust MCTS 搜索 — 使用 float 观测数组替代 JSON，消除序列化开销
use std::collections::HashMap;
use std::os::raw::c_char;
use crate::board::*;
use crate::rules::*;

/// 新回调：接收 (obs_ptr, batch_size, output_ptr, output_capacity) — 纯 float，零拷贝
type EvalFn = extern "C" fn(*const f32, i32, *mut f32, i32) -> i32;

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
// 主搜���函数
// =============================================================================

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
    let input = match std::ffi::CStr::from_ptr(state_json).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let mut root_state: GameState = match serde_json::from_str(input) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    root_state.board.rebuild_index_for_json();

    let mut root = MCTSNode::new(0.0);
    let bs = batch_size.max(1) as usize;
    let mut pending_states: Vec<GameState> = vec![];
    let mut pending_legal: Vec<Vec<usize>> = vec![];
    let mut pending_paths: Vec<Vec<usize>> = vec![];

    // Pre-allocate obs/output buffers (reused across flushes)
    let mut obs_buf: Vec<f32> = Vec::with_capacity(bs * obs_dim());
    let mut out_buf: Vec<f32> = Vec::with_capacity(bs * out_dim());

    for _ in 0..num_simulations.max(1) {
        let mut state = root_state.clone();
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
                n.children.iter().max_by(|a, b| {
                    a.1.ucb(parent_visits, c_puct).partial_cmp(&b.1.ucb(parent_visits, c_puct)).unwrap()
                }).map(|(a, _)| *a).unwrap()
            };
            state.step(best);
            path.push(best);
            if state.terminated { break; }
        }

        let terminated = state.terminated;
        let legal = state.get_legal_actions();

        if terminated {
            backup(&mut root, &path, terminal_value(&state));
        } else if !legal.is_empty() {
            pending_states.push(state);
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
    write_output(output_buf, output_size, &serde_json::to_string(&out).unwrap_or_default());
    0
}

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
    } else {
        // Uniform: no neural network, directly backup with 0 value
        for (path, legal) in paths.iter().zip(legal_list.iter()) {
            let node = node_by_path(root, path);
            let uniform = 1.0 / legal.len() as f64;
            for &a in legal {
                node.children.entry(a).or_insert(MCTSNode::new(uniform));
            }
            backup(root, path, 0.0);
        }
    }

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
