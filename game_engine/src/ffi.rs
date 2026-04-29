/// C-compatible FFI API for Python integration via ctypes.
///
/// Protocol: all functions take and return JSON strings.
/// Input/Output via pre-allocated buffers to avoid memory management issues.
use std::ffi::CStr;
use std::os::raw::c_char;
#[cfg(feature = "ort")]
use std::sync::OnceLock;
#[cfg(feature = "ort")]
use crate::net_infer::NetInfer;
#[cfg(feature = "ort")]
static ORT_MODEL: OnceLock<NetInfer> = OnceLock::new();

use crate::board::*;
use crate::rules::*;
use crate::mcts_rs;

/// Evaluate a game state and return the observation + legal actions as JSON.
///
/// # Safety
/// `state_json` must be a valid null-terminated JSON string.
/// `output_buffer` must have at least `buffer_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn game_evaluate(
    state_json: *const c_char,
    output_buffer: *mut c_char,
    buffer_size: i32,
) -> i32 {
    let input = match unsafe { CStr::from_ptr(state_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let mut state: GameState = match serde_json::from_str(input) {
        Ok(s) => s,
        Err(e) => {
            let err = format!(r#"{{"error":"{}"}}"#, e);
            write_output(output_buffer, buffer_size, &err);
            return -1;
        }
    };
    state.board.rebuild_index_for_json();

    let legal = state.get_legal_actions();
    let obs = serde_json::to_string(&state).unwrap_or_default();
    let result = serde_json::json!({
        "state": serde_json::Value::String(obs),
        "legal_actions": legal,
        "current_player": state.current_player,
        "phase": state.phase,
        "scores": state.scores,
    });

    let output = serde_json::to_string(&result).unwrap_or_default();
    write_output(output_buffer, buffer_size, &output);
    0
}

/// Step a game state with an action and return the new state.
///
/// # Safety
/// Same as game_evaluate.
#[no_mangle]
pub unsafe extern "C" fn game_step(
    state_json: *const c_char,
    action: i32,
    output_buffer: *mut c_char,
    buffer_size: i32,
) -> i32 {
    let input = match unsafe { CStr::from_ptr(state_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let mut state: GameState = match serde_json::from_str(input) {
        Ok(s) => s,
        Err(e) => {
            let err = format!(r#"{{"error":"{}"}}"#, e);
            write_output(output_buffer, buffer_size, &err);
            return -1;
        }
    };
    state.board.rebuild_index_for_json();

    let (reward, terminated) = state.step(action as usize);
    let state_json = serde_json::to_string(&state).unwrap_or_default();
    let result = serde_json::json!({
        "state": serde_json::Value::String(state_json),
        "reward": reward,
        "terminated": terminated,
    });

    let output = serde_json::to_string(&result).unwrap_or_default();
    write_output(output_buffer, buffer_size, &output);
    0
}

/// Create a new game with the given seed.
///
/// # Safety
/// Same as game_evaluate.
#[no_mangle]
pub unsafe extern "C" fn game_new(
    seed: i64,
    output_buffer: *mut c_char,
    buffer_size: i32,
) -> i32 {
    let seq = generate_sequence(seed as u64);
    let board = Board::new(&seq);
    let state = GameState::new(board);
    let state_json = serde_json::to_string(&state).unwrap_or_default();
    let result = serde_json::json!({
        "state": serde_json::Value::String(state_json),
    });

    let output = serde_json::to_string(&result).unwrap_or_default();
    write_output(output_buffer, buffer_size, &output);
    0
}

fn write_output(buffer: *mut c_char, size: i32, data: &str) {
    if size <= 0 {
        return;
    }
    let bytes = data.as_bytes();
    let len = bytes.len().min((size - 1) as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer as *mut u8, len);
        *buffer.add(len) = 0; // null-terminate
    }
}

/// Get the API version for compatibility checking.
#[no_mangle]
pub extern "C" fn api_version() -> i32 {
    1
}

// =============================================================================
// Stateful Game API — 游戏状态留在 Rust 内存，避免 JSON 序列化整个状态
// =============================================================================

const MAX_GAMES: usize = 1024;
static mut GAMES: Vec<Option<GameState>> = Vec::new();

/// 初始化状态槽（第一次使用时）
fn ensure_slots() {
    unsafe {
        if GAMES.is_empty() {
            GAMES.resize(MAX_GAMES, None);
        }
    }
}

/// 找空闲槽位，返回 handle
fn alloc_slot(state: GameState) -> i32 {
    ensure_slots();
    unsafe {
        for i in 0..MAX_GAMES {
            if GAMES[i].is_none() {
                GAMES[i] = Some(state);
                return i as i32;
            }
        }
    }
    -1 // 满
}

/// Create a new stateful game. Returns handle (i32).
#[no_mangle]
pub unsafe extern "C" fn game_stateful_new(seed: i64) -> i32 {
    let seq = generate_sequence(seed as u64);
    let board = Board::new(&seq);
    let state = GameState::new(board);
    alloc_slot(state)
}

/// Step a stateful game by handle. Returns small JSON (no full state).
#[no_mangle]
pub unsafe extern "C" fn game_stateful_step(
    handle: i32,
    action: i32,
    output_buffer: *mut c_char,
    buffer_size: i32,
) -> i32 {
    let game = match GAMES.get_mut(handle as usize) {
        Some(Some(g)) => g,
        _ => return -1,
    };
    let (reward, terminated) = game.step(action as usize);
    let result = serde_json::json!({"reward": reward, "terminated": terminated}).to_string();
    write_ab_json(output_buffer, buffer_size, &result);
    0
}

// ─── AlphaZero Model Inference FFI ────────────────────────────

const MAX_AZ_MODELS: usize = 16;
static mut AZ_MODELS: Vec<Option<crate::az_model::AZModelWeights>> = Vec::new();

fn get_az_model(handle: i32) -> Option<&'static mut crate::az_model::AZModelWeights> {
    if handle < 0 || handle as usize >= MAX_AZ_MODELS { return None; }
    unsafe { AZ_MODELS[handle as usize].as_mut() }
}

#[no_mangle]
pub unsafe extern "C" fn ffi_az_create(
    json_config: *const c_char, output_buf: *mut c_char, output_size: i32,
) -> i32 {
    let config_str = match std::ffi::CStr::from_ptr(json_config).to_str() {
        Ok(s) => s, _ => { write_ab_json(output_buf, output_size, r#"{"error":"bad config"}"#); return -1; }
    };
    let config: serde_json::Value = match serde_json::from_str(config_str) {
        Ok(v) => v, _ => { write_ab_json(output_buf, output_size, r#"{"error":"bad json"}"#); return -2; }
    };
    let arches = config["arches"].as_str().unwrap_or("mlp");
    let _total_layers = config["total_layers"].as_u64().unwrap_or(0) as usize;
    let total_weights = config["total_weights"].as_u64().unwrap_or(0) as usize;
    let total_biases = config["total_biases"].as_u64().unwrap_or(0) as usize;
    let policy_idx = config["policy_idx"].as_u64().unwrap_or(0) as usize;
    let value1_idx = config["value1_idx"].as_u64().unwrap_or(0) as usize;
    let value2_idx = config["value2_idx"].as_u64().unwrap_or(0) as usize;
    let value_uses_obs = config["value_uses_obs"].as_bool().unwrap_or(false);

    if AZ_MODELS.is_empty() {
        for _ in 0..MAX_AZ_MODELS { AZ_MODELS.push(None); }
    }

    let weights = crate::az_model::AZModelWeights {
        arch: if arches == "resnet" { crate::az_model::AZArch::ResNet } else { crate::az_model::AZArch::MLP },
        layers: Vec::new(),
        weights: vec![0.0f32; total_weights],
        biases: vec![0.0f32; total_biases],
        layer_info: Vec::new(),
        policy_idx,
        value1_idx,
        value2_idx,
        value_uses_obs,
    };

    let mut handle = -1i32;
    for i in 0..MAX_AZ_MODELS {
        if AZ_MODELS[i].is_none() {
            AZ_MODELS[i] = Some(weights);
            handle = i as i32; break;
        }
    }
    let result = format!(r#"{{"handle":{}}}"#, handle);
    write_ab_json(output_buf, output_size, &result);
    0
}

#[no_mangle]
pub unsafe extern "C" fn ffi_az_set_layer_info(
    handle: i32,
    data: *const i32, count: i32,
) -> i32 {
    let model = match get_az_model(handle) { Some(m) => m, None => return -1 };
    let info = std::slice::from_raw_parts(data, count as usize);
    // Each layer: rows, cols, weight_offset, bias_offset, has_relu, is_residual (6 ints)
    let n = count as usize / 6;
    model.layers.clear();
    for i in 0..n {
        let base = i * 6;
        let rows = info[base] as usize;
        let cols = info[base + 1] as usize;
        let wo = info[base + 2] as usize;
        let bo = info[base + 3] as usize;
        let has_relu = info[base + 4] != 0;
        let is_res = info[base + 5] != 0;
        model.layers.push(crate::az_model::AZLayer {
            weight_offset: wo, bias_offset: bo,
            rows, cols, has_relu, is_residual: is_res,
        });
        model.layer_info.push((rows, cols));
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn ffi_az_set_weights(
    handle: i32, w_ptr: *const f32, w_count: i32,
    b_ptr: *const f32, b_count: i32,
) -> i32 {
    let model = match get_az_model(handle) { Some(m) => m, None => return -1 };
    let w_src = std::slice::from_raw_parts(w_ptr, w_count as usize);
    let b_src = std::slice::from_raw_parts(b_ptr, b_count as usize);
    model.weights.copy_from_slice(w_src);
    model.biases.copy_from_slice(b_src);
    0
}

#[no_mangle]
pub unsafe extern "C" fn ffi_az_evaluate(
    handle: i32, obs: *const f32, n_states: i32,
    logits_out: *mut f32, values_out: *mut f32,
) -> i32 {
    let model = match get_az_model(handle) { Some(m) => m, None => return -1 };
    let n = n_states as usize;
    let obs_slice = std::slice::from_raw_parts(obs, n * 206);
    for i in 0..n {
        let mut obs_arr = [0.0f32; 206];
        obs_arr.copy_from_slice(&obs_slice[i * 206..(i + 1) * 206]);
        let (logits, value) = model.forward(&obs_arr);
        for j in 0..60 {
            *logits_out.add(i * 60 + j) = logits[j];
        }
        *values_out.add(i) = value;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn ffi_az_free(handle: i32) -> i32 {
    if handle >= 0 && (handle as usize) < MAX_AZ_MODELS {
        AZ_MODELS[handle as usize] = None;
        0
    } else { -1 }
}

/// Get legal actions for a stateful game (JSON array).
#[no_mangle]
pub unsafe extern "C" fn game_stateful_get_legal(
    handle: i32,
    output_buffer: *mut c_char,
    buffer_size: i32,
) -> i32 {
    let game = match GAMES.get(handle as usize) {
        Some(Some(g)) => g,
        _ => return -1,
    };
    let legal = game.get_legal_actions();
    let result = serde_json::json!({ "legal_actions": legal });
    let output = serde_json::to_string(&result).unwrap_or_default();
    write_output(output_buffer, buffer_size, &output);
    0
}

/// Serialize a stateful game to JSON (for MCTS or inspection).
#[no_mangle]
pub unsafe extern "C" fn game_stateful_to_json(
    handle: i32,
    output_buffer: *mut c_char,
    buffer_size: i32,
) -> i32 {
    let game = match GAMES.get(handle as usize) {
        Some(Some(g)) => g,
        _ => return -1,
    };
    let state_json = serde_json::to_string(game).unwrap_or_default();
    write_output(output_buffer, buffer_size, &state_json);
    0
}

/// Free a stateful game slot.
#[no_mangle]
pub unsafe extern "C" fn game_stateful_free(handle: i32) -> i32 {
    unsafe {
        if (handle as usize) < MAX_GAMES {
            GAMES[handle as usize] = None;
            0
        } else {
            -1
        }
    }
}

/// Get scores for a stateful game.
#[no_mangle]
pub unsafe extern "C" fn game_stateful_scores(
    handle: i32,
    output_buffer: *mut c_char,
    buffer_size: i32,
) -> i32 {
    let game = match GAMES.get(handle as usize) {
        Some(Some(g)) => g,
        _ => return -1,
    };
    let result = serde_json::json!({ "scores": game.scores });
    let output = serde_json::to_string(&result).unwrap_or_default();
    write_output(output_buffer, buffer_size, &output);
    0
}

/// Get current game info without stepping.
#[no_mangle]
pub unsafe extern "C" fn game_stateful_get_info(
    handle: i32,
    output_buffer: *mut c_char,
    buffer_size: i32,
) -> i32 {
    let game = match GAMES.get(handle as usize) {
        Some(Some(g)) => g,
        _ => return -1,
    };
    let legal = game.get_legal_actions();
    let result = serde_json::json!({
        "legal_actions": legal,
        "scores": game.scores,
        "current_player": game.current_player,
        "phase": game.phase,
        "terminated": game.terminated,
    });
    let output = serde_json::to_string(&result).unwrap_or_default();
    write_output(output_buffer, buffer_size, &output);
    0
}

/// Handle-based MCTS search: read state from GAMES[handle], run search,
/// bypassing the JSON serialization/deserialization roundtrip.
/// Returns visit counts JSON (same format as mcts_search_rust).
#[no_mangle]
pub unsafe extern "C" fn mcts_search_rust_handle(
    handle: i32,
    num_simulations: i32,
    c_puct: f64,
    batch_size: i32,
    eval_fn: Option<mcts_rs::EvalFn>,
    output_buf: *mut c_char,
    output_size: i32,
) -> i32 {
    let game = match GAMES.get(handle as usize) {
        Some(Some(g)) => g,
        _ => return -1,
    };
    let result = mcts_rs::mcts_search_on_handle(
        game, num_simulations, c_puct, batch_size, eval_fn,
    );
    write_output(output_buf, output_size, &result);
    0
}

/// MCTS search using Rust-native AZ model (no Python callback).
/// Takes a stateful game handle + AZ model handle.
#[no_mangle]
pub unsafe extern "C" fn mcts_search_rust_handle_az(
    handle: i32,
    num_simulations: i32,
    c_puct: f64,
    batch_size: i32,
    az_handle: i32,
    output_buf: *mut c_char,
    output_size: i32,
) -> i32 {
    let game = match GAMES.get(handle as usize) {
        Some(Some(g)) => g,
        _ => return -1,
    };
    let model = match get_az_model(az_handle) {
        Some(m) => m,
        None => return -2,
    };
    let result = mcts_rs::mcts_search_core_az(
        game, num_simulations, c_puct, batch_size, model,
    );
    write_output(output_buf, output_size, &result);
    0
}

const MAX_AB_SEARCHES: usize = 64;
static mut AB_SEARCHES: Vec<Option<crate::alphabeta_rs::AlphaBetaSearch>> = Vec::new();

fn get_ab_search(handle: i32) -> Option<&'static mut crate::alphabeta_rs::AlphaBetaSearch> {
    if handle < 0 || handle as usize >= MAX_AB_SEARCHES {
        return None;
    }
    unsafe {
        AB_SEARCHES[handle as usize].as_mut()
    }
}

// ─── NNUE Training Data Generation ──────────────────────────

const NNUE_RECORD_FLOATS: usize = 73;

#[no_mangle]
pub unsafe extern "C" fn ffi_generate_nnue_data(
    num_games: i32, seed_offset: i32, output_path: *const c_char,
) -> i32 {
    use crate::alphabeta_rs::{extract_sparse, extract_dense};
    use crate::board::Board;
    use crate::rules::{GameState, generate_sequence};
    use rand::Rng;

    let path = match std::ffi::CStr::from_ptr(output_path).to_str() {
        Ok(s) => s, _ => return -1,
    };
    let mut file = match std::fs::File::create(path) {
        Ok(f) => f, _ => return -2,
    };

    let mut buf: Vec<u8> = Vec::with_capacity(8 + (num_games as usize) * 70 * NNUE_RECORD_FLOATS * 4);
    let count_pos = buf.len();
    buf.extend_from_slice(&(0u64).to_le_bytes());
    let mut total_written: u64 = 0;
    let mut rng = rand::thread_rng();

    for g in 0..num_games {
        let seed = (seed_offset + g) as u64;
        let board = Board::new(&generate_sequence(seed));
        let mut core = GameState::new(board);
        let mut game_states: Vec<GameState> = Vec::new();

        // Placement phase (6 moves)
        for _ in 0..6 {
            let legal = core.get_legal_actions();
            if legal.is_empty() { break; }
            game_states.push(core.clone());
            let idx = rng.gen_range(0..legal.len());
            core.step(legal[idx]);
            if core.terminated { break; }
        }

        // Movement phase
        while !core.terminated {
            let legal = core.get_legal_actions();
            if legal.is_empty() { break; }
            game_states.push(core.clone());
            let idx = rng.gen_range(0..legal.len());
            core.step(legal[idx]);
            if core.episode_steps > 200 { break; }
        }

        let final_outcome: f32 = {
            let s1 = core.scores[0] as f32;
            let s2 = core.scores[1] as f32;
            if s1 > s2 { 1.0 } else if s2 > s1 { -1.0 } else { 0.0 }
        };

        for state in &game_states {
            let sparse = extract_sparse(state);
            let dense = extract_dense(state);
            let outcome = if state.current_player == 0 { final_outcome } else { -final_outcome };

            for i in 0..6 {
                let val: i32 = if i < sparse.len() { sparse[i] as i32 } else { -1 };
                buf.extend_from_slice(&val.to_le_bytes());
            }
            for &v in &dense { buf.extend_from_slice(&v.to_le_bytes()); }
            buf.extend_from_slice(&outcome.to_le_bytes());
            total_written += 1;
        }
    }

    // Update position count at start of file
    let cb = total_written.to_le_bytes();
    for (i, &b) in cb.iter().enumerate() { buf[count_pos + i] = b; }
    let _ = std::io::Write::write_all(&mut file, &buf);
    0
}

fn write_ab_json(output_buf: *mut c_char, output_size: i32, result: &str) {
    let bytes = result.as_bytes();
    let n = (bytes.len() + 1).min(output_size as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output_buf as *mut u8, n - 1);
        *output_buf.add(n - 1) = 0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn ffi_ab_create(
    json_config: *const c_char,
    output_buf: *mut c_char,
    output_size: i32,
) -> i32 {
    let config_json = match CStr::from_ptr(json_config).to_str() {
        Ok(s) => s, Err(_) => {
            write_ab_json(output_buf, output_size, r#"{"error":"invalid config json"}"#);
            return -1;
        }
    };

    let config: crate::alphabeta_rs::SearchConfig = match serde_json::from_str(config_json) {
        Ok(c) => c, Err(e) => {
            let err = format!(r#"{{"error":"bad config: {}"}}"#, e);
            write_ab_json(output_buf, output_size, &err);
            return -2;
        }
    };

    if AB_SEARCHES.is_empty() {
        for _ in 0..MAX_AB_SEARCHES {
            AB_SEARCHES.push(None);
        }
    }

    let mut handle = -1;
    for i in 0..MAX_AB_SEARCHES {
        if AB_SEARCHES[i].is_none() {
            let weights = crate::nnue_rs::NNUEWeights::new_empty();
            let search = crate::alphabeta_rs::AlphaBetaSearch::new(weights, config);
            AB_SEARCHES[i] = Some(search);
            handle = i as i32;
            break;
        }
    }

    let result = format!(r#"{{"handle":{}}}"#, handle);
    write_ab_json(output_buf, output_size, &result);
    0
}

#[no_mangle]
pub unsafe extern "C" fn ffi_ab_set_weights(
    handle: i32,
    ptr: *const f32,
    count: i32,
    output_buf: *mut c_char,
    output_size: i32,
) -> i32 {
    let search = match get_ab_search(handle) {
        Some(s) => s, None => {
            write_ab_json(output_buf, output_size, r#"{"error":"invalid handle"}"#);
            return -1;
        }
    };

    let expected = crate::nnue_rs::NNUEWeights::total_floats();
    if count as usize != expected {
        let err = format!(r#"{{"error":"expected {} floats, got {}"}}"#, expected, count);
        write_ab_json(output_buf, output_size, &err);
        return -2;
    }

    let data = std::slice::from_raw_parts(ptr, expected);
    let weights = crate::nnue_rs::NNUEWeights::from_flat(data);
    search.weights = weights;

    write_ab_json(output_buf, output_size, r#"{"ok":true}"#);
    0
}

#[no_mangle]
pub unsafe extern "C" fn ffi_ab_search(
    handle: i32,
    json_state: *const c_char,
    max_depth: i32,
    output_buf: *mut c_char,
    output_size: i32,
) -> i32 {
    let search = match get_ab_search(handle) {
        Some(s) => s, None => {
            write_ab_json(output_buf, output_size, r#"{"error":"invalid handle"}"#);
            return -1;
        }
    };

    let state_json = match CStr::from_ptr(json_state).to_str() {
        Ok(s) => s, Err(_) => {
            write_ab_json(output_buf, output_size, r#"{"error":"bad state json"}"#);
            return -2;
        }
    };

    let state: crate::rules::GameState = match serde_json::from_str::<crate::rules::GameState>(state_json) {
        Ok(mut s) => {
            s.board.rebuild_index_for_json();
            s
        },
        Err(e) => {
            let err = format!(r#"{{"error":"bad state: {}"}}"#, e);
            write_ab_json(output_buf, output_size, &err);
            return -3;
        }
    };

    if max_depth > 0 {
        search.config.max_depth = max_depth as u8;
    }

    let result = search.search(&state);
    let json = serde_json::to_string(&result).unwrap_or_default();
    write_ab_json(output_buf, output_size, &json);
    0
}

#[no_mangle]
pub unsafe extern "C" fn ffi_ab_destroy(handle: i32) -> i32 {
    if handle >= 0 && (handle as usize) < MAX_AB_SEARCHES {
        AB_SEARCHES[handle as usize] = None;
        0
    } else {
        -1
    }
}

/// Set the batch evaluation callback for all AB search instances.
/// The callback receives ([75 f32 per state], n_states, scores_out).
#[no_mangle]
pub unsafe extern "C" fn ffi_ab_set_eval_callback(cb: Option<crate::alphabeta_rs::EvalBatchFn>) -> i32 {
    if let Some(f) = cb {
        crate::alphabeta_rs::set_eval_callback(f);
        0
    } else {
        -1
    }
}

