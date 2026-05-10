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
    -1
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
    let obs_dim = crate::az_model::OBS_DIM;
    let obs_slice = std::slice::from_raw_parts(obs, n * obs_dim);
    for i in 0..n {
        let mut obs_arr = [0.0f32; crate::az_model::OBS_DIM];
        obs_arr.copy_from_slice(&obs_slice[i * obs_dim..(i + 1) * obs_dim]);
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

// ─── AZ MCTS Tree Reuse Handles ───────────────────────────

const MAX_AZ_MCTS_TREES: usize = 128;
static mut AZ_MCTS_TREES: Vec<Option<(crate::rules::GameState, crate::mcts_rs::MCTSNode)>> = Vec::new();

unsafe fn init_az_mcts_trees() {
    if AZ_MCTS_TREES.is_empty() {
        for _ in 0..MAX_AZ_MCTS_TREES { AZ_MCTS_TREES.push(None); }
    }
}

unsafe fn get_az_mcts_tree(handle: i32) -> Option<&'static mut (crate::rules::GameState, crate::mcts_rs::MCTSNode)> {
    if handle < 0 || handle as usize >= MAX_AZ_MCTS_TREES { return None; }
    AZ_MCTS_TREES[handle as usize].as_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ffi_az_mcts_init(
    game_handle: i32, az_handle: i32,
    num_simulations: i32, c_puct: f64, batch_size: i32,
    output_buf: *mut c_char, output_size: i32,
) -> i32 {
    init_az_mcts_trees();
    let game = match GAMES.get(game_handle as usize) { Some(Some(g)) => g.clone(), _ => return -1 };
    let model = match get_az_model(az_handle) { Some(m) => m, None => return -2 };

    let (state, root) = crate::mcts_rs::az_mcts_build_tree(&game, num_simulations, c_puct, batch_size, &model);
    let visits = crate::mcts_rs::tree_to_visits_json(&root);

    for i in 0..MAX_AZ_MCTS_TREES {
        if AZ_MCTS_TREES[i].is_none() {
            AZ_MCTS_TREES[i] = Some((state, root));
            let result = serde_json::json!({"handle": i, "visits": serde_json::from_str::<serde_json::Value>(&visits).ok()});
            write_output(output_buf, output_size, &serde_json::to_string(&result).unwrap_or_default());
            return 0;
        }
    }
    -3
}

#[no_mangle]
pub unsafe extern "C" fn ffi_az_mcts_step(
    tree_handle: i32, action: i32, az_handle: i32,
    additional_sims: i32, c_puct: f64, batch_size: i32,
    output_buf: *mut c_char, output_size: i32,
) -> i32 {
    let (state, root) = match get_az_mcts_tree(tree_handle) { Some(t) => t, None => return -1 };
    let model = match get_az_model(az_handle) { Some(m) => m, None => return -2 };

    // Step the game
    let legal = state.get_legal_actions();
    if legal.contains(&(action as usize)) { state.step(action as usize); }

    // Find child node → new root
    let child = root.children.remove(&(action as usize)).unwrap_or(crate::mcts_rs::MCTSNode::new(0.0));
    root.children.clear(); // detach all other children from old root
    *root = child;

    // Run additional simulations on new root
    if additional_sims > 0 {
        crate::mcts_rs::az_mcts_search_on_root(state, root, additional_sims, c_puct, batch_size, &model);
    }

    let visits = crate::mcts_rs::tree_to_visits_json(root);
    write_output(output_buf, output_size, &visits);
    0
}

#[no_mangle]
pub unsafe extern "C" fn ffi_az_mcts_free(tree_handle: i32) -> i32 {
    if tree_handle >= 0 && (tree_handle as usize) < MAX_AZ_MCTS_TREES {
        AZ_MCTS_TREES[tree_handle as usize] = None;
        0
    } else { -1 }
}

#[no_mangle]
pub unsafe extern "C" fn ffi_az_mcts_search_parallel(
    game_handle: i32, az_handle: i32,
    num_simulations: i32, c_puct: f64, batch_size: i32, num_threads: i32,
    output_buf: *mut c_char, output_size: i32,
) -> i32 {
    let game = match GAMES.get(game_handle as usize) { Some(Some(g)) => g.clone(), _ => return -1 };
    let model = match get_az_model(az_handle) { Some(m) => m.clone(), None => return -2 };
    let result = crate::mcts_rs::mcts_search_core_az_parallel(&game, num_simulations, c_puct, batch_size, num_threads, &model);
    write_output(output_buf, output_size, &result);
    0
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
        game, num_simulations, c_puct, batch_size, eval_fn, false,  // AZ mode (272-dim)
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

/// MCTS search using NNUE eval callback (Python-side, 75-dim obs).
#[no_mangle]
pub unsafe extern "C" fn mcts_search_nnue_handle(
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
        game, num_simulations, c_puct, batch_size, eval_fn, true,
    );
    write_output(output_buf, output_size, &result);
    0
}

const MAX_NNUE_MCTS: usize = 64;
static mut NNUE_MCTS_MODELS: Vec<Option<crate::nnue_rs::NNUEMCTSWeights>> = Vec::new();

/// Create an NNUE MCTS model handle (no weights loaded yet).
#[no_mangle]
pub unsafe extern "C" fn nnue_mcts_create(
    output_buf: *mut c_char, output_size: i32,
) -> i32 {
    if NNUE_MCTS_MODELS.is_empty() {
        for _ in 0..MAX_NNUE_MCTS { NNUE_MCTS_MODELS.push(None); }
    }
    let mut handle = -1i32;
    for i in 0..MAX_NNUE_MCTS {
        if NNUE_MCTS_MODELS[i].is_none() {
            NNUE_MCTS_MODELS[i] = Some(crate::nnue_rs::NNUEMCTSWeights::new_empty());
            handle = i as i32; break;
        }
    }
    let result = format!(r#"{{"handle":{}}}"#, handle);
    write_ab_json(output_buf, output_size, &result);
    0
}

/// Set NNUEMCTS weights from flat float array.
#[no_mangle]
pub unsafe extern "C" fn nnue_mcts_set_weights(
    handle: i32, data: *const f32, count: i32,
    output_buf: *mut c_char, output_size: i32,
) -> i32 {
    let model = match NNUE_MCTS_MODELS.get_mut(handle as usize) {
        Some(Some(m)) => m,
        _ => { write_ab_json(output_buf, output_size, r#"{"error":"bad handle"}"#); return -1; }
    };
    let flat = std::slice::from_raw_parts(data, count as usize);
    let expected = 360 * crate::nnue_rs::MCTS_FT_DIM + crate::nnue_rs::MCTS_FT_DIM
        + crate::nnue_rs::MCTS_INPUT_DIM * crate::nnue_rs::MCTS_FC1_DIM + crate::nnue_rs::MCTS_FC1_DIM
        + crate::nnue_rs::MCTS_FC1_DIM + 1
        + crate::nnue_rs::MCTS_FC1_DIM * 60 + 60;
    if flat.len() < expected {
        write_ab_json(output_buf, output_size, r#"{"error":"short weights"}"#); return -3;
    }
    let mut off = 0;
    // ft_weight (360, MCTS_FT_DIM)
    let ft_sz = 360 * crate::nnue_rs::MCTS_FT_DIM;
    let fc1_sz = crate::nnue_rs::MCTS_INPUT_DIM * crate::nnue_rs::MCTS_FC1_DIM;
    model.ft_weight.copy_from_slice(&flat[off..off + ft_sz]); off += ft_sz;
    model.ft_bias.copy_from_slice(&flat[off..off + crate::nnue_rs::MCTS_FT_DIM]); off += crate::nnue_rs::MCTS_FT_DIM;
    model.fc1_weight_t.copy_from_slice(&flat[off..off + fc1_sz]); off += fc1_sz;
    model.fc1_bias.copy_from_slice(&flat[off..off + crate::nnue_rs::MCTS_FC1_DIM]); off += crate::nnue_rs::MCTS_FC1_DIM;
    model.fc2v_weight.copy_from_slice(&flat[off..off + crate::nnue_rs::MCTS_FC1_DIM]); off += crate::nnue_rs::MCTS_FC1_DIM;
    model.fc2v_bias.copy_from_slice(&flat[off..off + 1]); off += 1;
    model.fc2p_weight_row.copy_from_slice(&flat[off..off + crate::nnue_rs::MCTS_FC1_DIM * 60]); off += crate::nnue_rs::MCTS_FC1_DIM * 60;
    model.fc2p_bias.copy_from_slice(&flat[off..off + 60]);
    write_ab_json(output_buf, output_size, r#"{"ok":true}"#);
    0
}

/// NNUE-MCTS search: Rust-native, no Python callback.
#[no_mangle]
pub unsafe extern "C" fn mcts_search_nnue_native(
    game_handle: i32, model_handle: i32,
    num_simulations: i32, c_puct: f64,
    output_buf: *mut c_char, output_size: i32,
) -> i32 {
    let game = match GAMES.get(game_handle as usize) {
        Some(Some(g)) => g, _ => return -1,
    };
    let model = match NNUE_MCTS_MODELS.get(model_handle as usize) {
        Some(Some(m)) => m, _ => return -2,
    };
    let result = mcts_rs::mcts_search_core_nnue(
        game, num_simulations, c_puct, model,
    );
    write_output(output_buf, output_size, &result);
    0
}

// ─── MCTS Tree Reuse ──────────────────────────────────────────

use std::collections::HashMap;
const MAX_MCTS_TREES: usize = 64;
static mut MCTS_TREES: Vec<Option<mcts_rs::MCTSReuseTree>> = Vec::new();

/// Initialize MCTS tree for a game (runs first search batch).
#[no_mangle]
pub unsafe extern "C" fn nnue_mcts_tree_init(
    game_handle: i32,
    model_handle: i32,
    num_simulations: i32,
    c_puct: f64,
    output_buf: *mut c_char,
    output_size: i32,
) -> i32 {
    let game = match GAMES.get(game_handle as usize) {
        Some(Some(g)) => g,
        _ => { write_ab_json(output_buf, output_size, r#"{"error":"bad game"}"#); return -1; }
    };
    let model = match NNUE_MCTS_MODELS.get(model_handle as usize) {
        Some(Some(m)) => m,
        _ => { write_ab_json(output_buf, output_size, r#"{"error":"bad model"}"#); return -2; }
    };

    if MCTS_TREES.is_empty() {
        for _ in 0..MAX_MCTS_TREES { MCTS_TREES.push(None); }
    }

    let mut handle = -1i32;
    for i in 0..MAX_MCTS_TREES {
        if MCTS_TREES[i].is_none() { handle = i as i32; break; }
    }
    if handle < 0 { write_ab_json(output_buf, output_size, r#"{"error":"no slots"}"#); return -3; }

    let (state_clone, root) = mcts_rs::mcts_build_tree(game, num_simulations, c_puct, model);
    MCTS_TREES[handle as usize] = Some(mcts_rs::MCTSReuseTree { state: state_clone, root, model_handle });

    let visits = mcts_rs::tree_to_visits_json(&MCTS_TREES[handle as usize].as_ref().unwrap().root);
    write_ab_json(output_buf, output_size, &visits);
    0
}

/// Step the game, reuse tree, continue search.
#[no_mangle]
pub unsafe extern "C" fn nnue_mcts_tree_step(
    tree_handle: i32,
    action: i32,
    additional_sims: i32,
    c_puct: f64,
    output_buf: *mut c_char,
    output_size: i32,
) -> i32 {
    let tree = match MCTS_TREES.get_mut(tree_handle as usize) {
        Some(Some(t)) => t,
        _ => { write_ab_json(output_buf, output_size, r#"{"error":"bad handle"}"#); return -1; }
    };

    let model = match NNUE_MCTS_MODELS.get(tree.model_handle as usize) {
        Some(Some(m)) => m,
        _ => { write_ab_json(output_buf, output_size, r#"{"error":"bad model"}"#); return -2; }
    };

    // Step the game
    tree.state.step(action as usize);

    // Reuse child as new root
    if let Some(child) = tree.root.children.remove(&(action as usize)) {
        tree.root = child;
    } else {
        tree.root = mcts_rs::MCTSNode::new(0.0);
    }

    // Additional simulations on the new root
    mcts_rs::mcts_search_on_root(
        &mut tree.state, &mut tree.root, additional_sims, c_puct, model);

    let visits = mcts_rs::tree_to_visits_json(&tree.root);
    write_ab_json(output_buf, output_size, &visits);
    0
}

/// Free MCTS tree handle.
#[no_mangle]
pub unsafe extern "C" fn nnue_mcts_tree_free(
    tree_handle: i32,
) -> i32 {
    if tree_handle >= 0 && (tree_handle as usize) < MAX_MCTS_TREES {
        MCTS_TREES[tree_handle as usize] = None;
    }
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

const NNUE_RECORD_FLOATS: usize = 73; // 6 sparse + 66 dense + 1 outcome

#[no_mangle]
pub unsafe extern "C" fn ffi_ab_generate_random_data(
    ab_handle: i32,
    num_games: i32,
    seed_offset: i32,
    workers: i32,
    output_path: *const c_char,
) -> i64 {
    use crate::alphabeta_rs::{extract_sparse, extract_dense, AlphaBetaSearch};
    use crate::board::Board;
    use crate::rules::{GameState, generate_sequence};
    use rand::Rng;
    use std::fs::File;
    use std::io::Write;

    let path = match std::ffi::CStr::from_ptr(output_path).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    // Get weights + config from the handle
    let template = match get_ab_search(ab_handle) {
        Some(s) => s,
        None => return -2,
    };
    let weights = template.weights.clone();
    let mut gen_config = template.config.clone();
    // Use config's max_depth (no override)
    drop(template);

    let n_workers = workers.max(1) as usize;
    let games_per = (num_games as usize) / n_workers;
    let remainder = (num_games as usize) % n_workers;

    // Each thread generates its own data buffer
    let results: std::sync::Mutex<Vec<(Vec<u8>, u64)>> = std::sync::Mutex::new(Vec::new());

    std::thread::scope(|s| {
        for tid in 0..n_workers {
            let w = weights.clone();
            let cfg = gen_config.clone();
            let my_games = games_per + if tid < remainder { 1 } else { 0 };
            let result_ref = &results;

            s.spawn(move || {
                let mut search = AlphaBetaSearch::new(w, cfg);
                let mut rng = rand::thread_rng();
                let mut buf: Vec<u8> = Vec::with_capacity(my_games * 60 * 296);
                let mut count: u64 = 0;

                for g in 0..my_games {
                    let seed = (seed_offset + g as i32 + tid as i32 * 10000) as u64;
                    let seq = generate_sequence(seed);
                    let board = Board::new(&seq);
                    let mut state = GameState::new(board);

                    // Placement phase: random
                    for _ in 0..6 {
                        let legal = state.get_legal_actions();
                        if legal.is_empty() || state.terminated { break; }
                        // Record BEFORE the placement move
                        let sparse = extract_sparse(&state);
                        let dense = extract_dense(&state);
                        let result = search.search(&state);
                        for i in 0..6 { buf.extend_from_slice(&(if i < sparse.len() { sparse[i] as i32 } else { -1i32 }).to_le_bytes()); }
                        for &v in &dense { buf.extend_from_slice(&v.to_le_bytes()); }
                        buf.extend_from_slice(&result.score.to_le_bytes());
                        buf.extend_from_slice(&(state.current_player as i32).to_le_bytes());
                        count += 1;

                        let idx = rng.gen_range(0..legal.len());
                        state.step(legal[idx]);
                    }

                    // Movement phase: AB-guided
                    while !state.terminated && state.episode_steps < 200 {
                        let legal = state.get_legal_actions();
                        if legal.is_empty() { break; }
                        // Record
                        let sparse = extract_sparse(&state);
                        let dense = extract_dense(&state);
                        let result = search.search(&state);
                        for i in 0..6 { buf.extend_from_slice(&(if i < sparse.len() { sparse[i] as i32 } else { -1i32 }).to_le_bytes()); }
                        for &v in &dense { buf.extend_from_slice(&v.to_le_bytes()); }
                        buf.extend_from_slice(&result.score.to_le_bytes());
                        buf.extend_from_slice(&(state.current_player as i32).to_le_bytes());
                        count += 1;

                        let idx = rng.gen_range(0..legal.len());
                        state.step(legal[idx]);
                    }
                }
                result_ref.lock().unwrap().push((buf, count));
            });
        }
    });

    // Merge all thread buffers
    let merged = results.lock().unwrap();
    let total_count: u64 = merged.iter().map(|(_, c)| c).sum();
    let total_size: usize = merged.iter().map(|(b, _)| b.len()).sum();
    let mut final_buf: Vec<u8> = Vec::with_capacity(8 + total_size);
    final_buf.extend_from_slice(&total_count.to_le_bytes());
    for (buf, _) in merged.iter() {
        final_buf.extend_from_slice(buf);
    }

    if let Ok(mut file) = File::create(path) {
        file.write_all(&final_buf).ok();
    }
    total_count as i64
}

/// Generate self-play data: uses AB search for both evaluation AND move selection.
#[no_mangle]
pub unsafe extern "C" fn ffi_ab_generate_selfplay_data(
    ab_handle: i32, num_games: i32, seed_offset: i32, workers: i32, output_path: *const c_char,
) -> i64 {
    use crate::alphabeta_rs::{extract_sparse, extract_dense, AlphaBetaSearch};
    use crate::board::Board;
    use crate::rules::{GameState, generate_sequence};
    use rand::Rng;
    use std::fs::File;
    use std::io::Write;

    let path = match std::ffi::CStr::from_ptr(output_path).to_str() {
        Ok(s) => s, Err(_) => return -1,
    };
    let template = match get_ab_search(ab_handle) {
        Some(s) => s, None => return -2,
    };
    let weights = template.weights.clone();
    let mut gen_config = template.config.clone();
    drop(template);

    let n_workers = workers.max(1) as usize;
    let games_per = (num_games as usize) / n_workers;
    let remainder = (num_games as usize) % n_workers;
    let results: std::sync::Mutex<Vec<(Vec<u8>, u64)>> = std::sync::Mutex::new(Vec::new());

    std::thread::scope(|s| {
        for tid in 0..n_workers {
            let w = weights.clone();
            let cfg = gen_config.clone();
            let my_games = games_per + if tid < remainder { 1 } else { 0 };
            let result_ref = &results;

            s.spawn(move || {
                let mut rng = rand::thread_rng();
                let mut search = AlphaBetaSearch::new(w, cfg);
                let mut buf: Vec<u8> = Vec::with_capacity(my_games * 60 * 296);
                let mut count: u64 = 0;

                for g in 0..my_games {
                    let seed = (seed_offset + g as i32 + tid as i32 * 10000) as u64;
                    let seq = generate_sequence(seed);
                    let board = Board::new(&seq);
                    let mut state = GameState::new(board);

                    // Placement phase: AB-guided (epsilon-greedy)
                    for _ in 0..6 {
                        let legal = state.get_legal_actions();
                        if legal.is_empty() || state.terminated { break; }
                        let sparse = extract_sparse(&state);
                        let dense = extract_dense(&state);
                        let result = search.search(&state);
                        for i in 0..6 { buf.extend_from_slice(&(if i < sparse.len() { sparse[i] as i32 } else { -1i32 }).to_le_bytes()); }
                        for &v in &dense { buf.extend_from_slice(&v.to_le_bytes()); }
                        buf.extend_from_slice(&result.score.to_le_bytes());
                        buf.extend_from_slice(&(state.current_player as i32).to_le_bytes());
                        count += 1;
                        let a = if rng.gen::<f32>() < search.config.epsilon { legal[rng.gen_range(0..legal.len())] } else { result.best_action };
                        state.step(a);
                    }
                    
                    // Movement phase: AB-guided (epsilon-greedy)
                    while !state.terminated && state.episode_steps < 200 {
                        let legal = state.get_legal_actions();
                        if legal.is_empty() { break; }
                        let sparse = extract_sparse(&state);
                        let dense = extract_dense(&state);
                        let result = search.search(&state);
                        for i in 0..6 { buf.extend_from_slice(&(if i < sparse.len() { sparse[i] as i32 } else { -1i32 }).to_le_bytes()); }
                        for &v in &dense { buf.extend_from_slice(&v.to_le_bytes()); }
                        buf.extend_from_slice(&result.score.to_le_bytes());
                        buf.extend_from_slice(&(state.current_player as i32).to_le_bytes());
                        count += 1;
                        let a = if rng.gen::<f32>() < search.config.epsilon { legal[rng.gen_range(0..legal.len())] } else { result.best_action };
                        state.step(a);
                    }
                }
                result_ref.lock().unwrap().push((buf, count));
            });
        }
    });

    let merged = results.lock().unwrap();
    let total_count: u64 = merged.iter().map(|(_, c)| c).sum();
    let total_size: usize = merged.iter().map(|(b, _)| b.len()).sum();
    let mut final_buf: Vec<u8> = Vec::with_capacity(8 + total_size);
    final_buf.extend_from_slice(&total_count.to_le_bytes());
    for (buf, _) in merged.iter() { final_buf.extend_from_slice(buf); }
    if let Ok(mut file) = File::create(path) { file.write_all(&final_buf).ok(); }
    total_count as i64
}

/// Generate self-play data with AZ-format observations (272-dim).
/// Binary format: n_records(u64) + [obs(272×f32) + action(i32) + outcome(f32) + stm(i32)] × n
#[no_mangle]
pub unsafe extern "C" fn ffi_ab_generate_az_data(
    ab_handle: i32,
    num_games: i32,
    seed_offset: i32,
    workers: i32,
    output_path: *const c_char,
) -> i64 {
    use crate::alphabeta_rs::AlphaBetaSearch;
    use crate::az_model::encode_obs;
    use crate::board::Board;
    use crate::rules::{GameState, Phase, generate_sequence};
    use rand::Rng;
    use std::fs::File;
    use std::io::Write;

    let path = match std::ffi::CStr::from_ptr(output_path).to_str() { Ok(s) => s, _ => return -1 };
    let search = match AB_SEARCHES.get(ab_handle as usize) {
        Some(Some(s)) => s,
        _ => return -2,
    };
    let cfg = search.config.clone();
    let weights = search.weights.clone();
    drop(search); // release the borrow before using cfg in the closure
    let nw = workers.max(1) as usize;
    let mut file = match File::create(path) { Ok(f) => f, _ => return -3 };
    let mut buf: Vec<u8> = Vec::with_capacity(1_000_000);
    let count_pos = buf.len();
    buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder
    let mut total: u64 = 0;

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(nw);
        for w in 0..nw {
            let w_cfg = cfg.clone(); let w_weights = weights.clone();
            handles.push(s.spawn(move || {
                let mut search = AlphaBetaSearch::new(w_weights, w_cfg);
                let mut local_buf: Vec<u8> = Vec::new();
                let mut local_count: u64 = 0;

                let games_per_worker = num_games / nw as i32;
                let extra = (num_games % nw as i32) as usize;
                let start = w as i32 * games_per_worker;

                let n_my_games = games_per_worker + if w < extra { 1 } else { 0 };

                for g in 0..n_my_games {
                    let seed = (seed_offset + start + g) as u64;
                    let seq = generate_sequence(seed);
                    let board = Board::new(&seq);
                    let mut state = GameState::new(board);
                    let mut game_states: Vec<(GameState, i32)> = Vec::new();

                    while !state.terminated && state.episode_steps < 200 {
                        let legal = state.get_legal_actions();
                        if legal.is_empty() { break; }
                        game_states.push((state.clone(), -1));
                        let result = search.search(&state);
                        let action = result.best_action;
                        if legal.contains(&action) { state.step(action); }
                        if let Some(last) = game_states.last_mut() { last.1 = action as i32; }
                    }

                    let s1 = state.scores[0]; let s2 = state.scores[1];
                    let outcome: f32 = if s1 > s2 { 1.0 } else if s2 > s1 { -1.0 } else { 0.0 };

                    for (st, act) in &game_states {
                        let bn = &st.board.cells; let ps = &st.pieces; let cp = st.current_player;
                        let ph = if st.phase == crate::rules::Phase::Movement { 1u8 } else { 0u8 };
                        let obs = encode_obs(bn, ps, cp, ph);
                        for &v in obs.iter() { local_buf.extend_from_slice(&v.to_le_bytes()); }
                        local_buf.extend_from_slice(&act.to_le_bytes());
                        let ao = if st.current_player == 0 { outcome } else { -outcome };
                        local_buf.extend_from_slice(&ao.to_le_bytes());
                        local_buf.extend_from_slice(&(st.current_player as i32).to_le_bytes());
                        local_count += 1;
                    }
                }
                (local_buf, local_count)
            }));
        }

        for h in handles {
            if let Ok((lb, lc)) = h.join() {
                buf.extend_from_slice(&lb);
                total += lc;
            }
        }
    });

    let cb = total.to_le_bytes();
    for (i, &b) in cb.iter().enumerate() { buf[count_pos + i] = b; }
    let _ = file.write_all(&buf);
    total as i64
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

/// Run NNUE training. Takes weights (flat in/out), data path, JSON config.
/// Returns JSON with final_loss, best_loss, epochs.
#[no_mangle]
pub unsafe extern "C" fn ffi_nnue_train(
    weights_flat: *mut f32, weight_count: i32,
    data_path: *const c_char, config_json: *const c_char,
    output_buf: *mut c_char, output_size: i32,
) -> i32 {
    let path = match CStr::from_ptr(data_path).to_str() { Ok(s) => s, _ => { write_ab_json(output_buf, output_size, r#"{"error":"bad path"}"#); return -1; }};
    let cjson = match CStr::from_ptr(config_json).to_str() { Ok(s) => s, _ => { write_ab_json(output_buf, output_size, r#"{"error":"bad config"}"#); return -1; }};

    if weight_count as usize != crate::nnue_rs::NNUEWeights::total_floats() {
        let err = format!(r#"{{"error":"expected {} floats, got {}"}}"#, crate::nnue_rs::NNUEWeights::total_floats(), weight_count);
        write_ab_json(output_buf, output_size, &err); return -2;
    }
    let flat = std::slice::from_raw_parts_mut(weights_flat, weight_count as usize);
    let mut weights = crate::nnue_rs::NNUEWeights::from_flat(flat);
    let records = crate::nnue_train::load_records(path);

    let (lr, wd, bs, ep, mn) = match serde_json::from_str::<serde_json::Value>(cjson) {
        Ok(v) => (
            v.get("lr").and_then(|x| x.as_f64()).unwrap_or(3e-4) as f32,
            v.get("wd").and_then(|x| x.as_f64()).unwrap_or(1e-4) as f32,
            v.get("batch_size").and_then(|x| x.as_u64()).unwrap_or(4096) as usize,
            v.get("epochs").and_then(|x| x.as_u64()).unwrap_or(50) as usize,
            v.get("max_norm").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32,
        ), Err(_) => { write_ab_json(output_buf, output_size, r#"{"error":"bad json"}"#); return -3; }
    };
    let cfg = crate::nnue_train::TrainingConfig { learning_rate: lr, weight_decay: wd, batch_size: bs, n_epochs: ep, max_norm: mn };
    let result = crate::nnue_train::train(&mut weights, &records, &cfg);

    let new_flat = crate::nnue_rs::NNUEWeights::flatten(&weights);
    for (i, &v) in new_flat.iter().enumerate() { flat[i] = v; }
    let out = format!(r#"{{"final_loss":{},"best_loss":{},"epochs":{}}}"#, result.epoch_losses.last().unwrap_or(&0.0), result.best_loss, ep);
    write_ab_json(output_buf, output_size, &out);
    0
}

/// Train NNUE FC layers using Candle (correct Adam). Keeps FT fixed.
#[no_mangle]
pub unsafe extern "C" fn ffi_nnue_train_candle(
    weights_flat: *mut f32, weight_count: i32,
    data_path: *const c_char, config_json: *const c_char,
    output_buf: *mut c_char, output_size: i32,
) -> i32 {
    let path = match CStr::from_ptr(data_path).to_str() { Ok(s) => s, _ => { write_ab_json(output_buf, output_size, r#"{"error":"bad path"}"#); return -1; }};
    let cjson = match CStr::from_ptr(config_json).to_str() { Ok(s) => s, _ => { write_ab_json(output_buf, output_size, r#"{"error":"bad config"}"#); return -1; }};

    if weight_count as usize != crate::nnue_rs::NNUEWeights::total_floats() {
        let err = format!(r#"{{"error":"expected {} floats, got {}"}}"#, crate::nnue_rs::NNUEWeights::total_floats(), weight_count);
        write_ab_json(output_buf, output_size, &err); return -2;
    }

    let flat = std::slice::from_raw_parts_mut(weights_flat, weight_count as usize);
    let mut weights = crate::nnue_rs::NNUEWeights::from_flat(flat);
    let ft = weights.clone();  // Keep frozen FT
    let records = crate::nnue_train::load_records(path);

    let v = serde_json::from_str::<serde_json::Value>(cjson).unwrap_or_default();
    let lr = v.get("lr").and_then(|x| x.as_f64()).unwrap_or(1e-4) as f32;
    let wd = v.get("wd").and_then(|x| x.as_f64()).unwrap_or(1e-4) as f32;
    let bs = v.get("batch_size").and_then(|x| x.as_u64()).unwrap_or(4096) as usize;
    let ep = v.get("epochs").and_then(|x| x.as_u64()).unwrap_or(50) as usize;

    let loss = crate::nnue_candle::train_all(&records, &mut weights, lr, wd, bs, ep);

    let new_flat = crate::nnue_rs::NNUEWeights::flatten(&weights);
    for (i, &v) in new_flat.iter().enumerate() { flat[i] = v; }
    let out = format!(r#"{{"final_loss":{},"epochs":{}}}"#, loss, ep);
    write_ab_json(output_buf, output_size, &out);
    0
}

