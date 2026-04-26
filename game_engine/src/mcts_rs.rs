/// Rust MCTS 搜索 — 路径回溯（无父指针），批量回调评估
use std::collections::HashMap;
use std::os::raw::c_char;
use crate::board::*;
use crate::rules::*;

type EvalCallback = extern "C" fn(*const c_char, *mut c_char, i32) -> i32;

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

#[no_mangle]
pub unsafe extern "C" fn mcts_search_rust(
    state_json: *const c_char,
    num_simulations: i32,
    c_puct: f64,
    batch_size: i32,
    eval_cb: Option<EvalCallback>,
    output_buf: *mut c_char,
    output_size: i32,
) -> i32 {
    let input = std::ffi::CStr::from_ptr(state_json).to_str().map_err(|_| -1).unwrap_or("");
    let mut root_state: GameState = serde_json::from_str(input).map_err(|_| -1).unwrap_or_else(|_| std::process::exit(-1));
    root_state.board.rebuild_index_for_json();

    let mut root = MCTSNode::new(0.0);
    let bs = batch_size.max(1) as usize;
    let mut pending_states: Vec<String> = vec![];
    let mut pending_legal: Vec<Vec<usize>> = vec![];
    let mut pending_paths: Vec<Vec<usize>> = vec![];

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
            pending_states.push(serde_json::to_string(&state).unwrap_or_default());
            pending_legal.push(legal);
            pending_paths.push(path);
            if pending_states.len() >= bs {
                flush_batch(&mut root, &mut pending_states, &mut pending_legal, &mut pending_paths, eval_cb);
            }
        }
    }
    if !pending_states.is_empty() {
        flush_batch(&mut root, &mut pending_states, &mut pending_legal, &mut pending_paths, eval_cb);
    }

    let mut out = serde_json::Map::new();
    for (a, c) in &root.children {
        out.insert(a.to_string(), serde_json::Value::Number(serde_json::Number::from(c.visits)));
    }
    write_output(output_buf, output_size, &serde_json::to_string(&out).unwrap_or_default());
    0
}

fn backup(root: &mut MCTSNode, path: &[usize], mut value: f64) {
    for depth in (0..=path.len()).rev() {
        let n = node_by_path(root, &path[..depth]);
        n.visits += 1; n.total_value += value; value = -value;
    }
}

fn flush_batch(
    root: &mut MCTSNode, states: &mut Vec<String>, legal_list: &mut Vec<Vec<usize>>,
    paths: &mut Vec<Vec<usize>>, cb: Option<EvalCallback>,
) {
    let batch = format!("[{}]", states.join(","));
    let mut buf = vec![0u8; 65536];
    if let Some(eval_cb) = cb {
        unsafe { eval_cb(batch.as_ptr() as *const c_char, buf.as_mut_ptr() as *mut c_char, 65536); }
    }
    let result_str = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr() as *const c_char).to_str().unwrap_or("[]") };
    let results: Vec<serde_json::Value> = serde_json::from_str(result_str).unwrap_or_default();

    for (i, (path, legal)) in paths.iter().zip(legal_list.iter()).enumerate() {
        let value = if i < results.len() {
            let logits: Vec<f64> = results[i]["logits"].as_array()
                .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
                .unwrap_or_else(|| vec![0.0; 60]);
            let val = results[i]["value"].as_f64().unwrap_or(0.0);
            let mut masked = vec![-1e9_f64; 60];
            for &a in legal { masked[a] = logits[a]; }
            let max_l = masked.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = masked.iter().map(|x| (x - max_l).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let policy: Vec<f64> = if sum > 1e-12 {
                exps.iter().map(|x| x / sum).collect()
            } else {
                let mut v = vec![0.0; 60]; for &a in legal { v[a] = 1.0 / legal.len() as f64; } v
            };
            let n = node_by_path(root, path);
            for &a in legal { n.children.entry(a).or_insert(MCTSNode::new(policy[a])); }
            val
        } else { 0.0 };
        backup(root, path, value);
    }
    states.clear(); legal_list.clear(); paths.clear();
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
