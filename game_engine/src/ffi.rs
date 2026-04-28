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
    let legal = game.get_legal_actions();
    let result = serde_json::json!({
        "reward": reward,
        "terminated": terminated,
        "legal_actions": legal,
        "scores": game.scores,
        "current_player": game.current_player,
        "phase": game.phase,
    });
    let output = serde_json::to_string(&result).unwrap_or_default();
    write_output(output_buffer, buffer_size, &output);
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
        game, num_simulations, c_puct, batch_size, eval_fn,
    );
    write_output(output_buf, output_size, &result);
    0
}

/// Get flat observation from stateful game (for PPO Agent).
/// Returns JSON array of floats in the same format as PenguinChessCore.get_observation().
#[no_mangle]
pub unsafe extern "C" fn game_stateful_get_obs(
    handle: i32,
    output_buffer: *mut c_char,
    buffer_size: i32,
) -> i32 {
    let game = match GAMES.get(handle as usize) {
        Some(Some(g)) => g,
        _ => return -1,
    };
    // board: 60 cells × [q/8, r/8, value/3 or 0]
    let mut board: Vec<Vec<f64>> = Vec::with_capacity(60);
    for cell in &game.board.cells {
        let val = if cell.state == HexState::Active || cell.state == HexState::Occupied {
            cell.points as f64 / 3.0
        } else {
            0.0
        };
        board.push(vec![cell.coord.q as f64 / 8.0, cell.coord.r as f64 / 8.0, val]);
    }
    // pieces: 6 pieces × [id/10, q/8, r/8, s/8]
    let mut pieces: Vec<Vec<f64>> = Vec::with_capacity(6);
    for piece in &game.pieces {
        if piece.alive && piece.hex_idx.is_some() {
            let idx = piece.hex_idx.unwrap();
            let cell = &game.board.cells[idx];
            pieces.push(vec![
                piece.id as f64 / 10.0,
                cell.coord.q as f64 / 8.0,
                cell.coord.r as f64 / 8.0,
                cell.coord.s as f64 / 8.0,
            ]);
        } else {
            pieces.push(vec![-1.0, 0.0, 0.0, 0.0]);
        }
    }
    let result = serde_json::json!({
        "board": board,
        "pieces": pieces,
        "current_player": game.current_player as f64,
        "phase": if matches!(game.phase, Phase::Placement) { 0.0 } else { 1.0 },
    });
    let output = serde_json::to_string(&result).unwrap_or_default();
    write_output(output_buffer, buffer_size, &output);
    0
}

// ────────────────────────────────────────────────────────────
// Parallel MCTS — single FFI call, internal thread parallelism
// ────────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn mcts_search_rust_handle_parallel(
    handle: i32,
    num_simulations: i32,
    c_puct: f64,
    batch_size: i32,
    num_workers: i32,
    eval_fn: Option<mcts_rs::EvalFn>,
    output_buf: *mut c_char,
    output_size: i32,
) -> i32 {
    let game = match GAMES.get(handle as usize) {
        Some(Some(g)) => g,
        _ => return -1,
    };
    let result = mcts_rs::mcts_search_parallel_core(
        game,
        num_simulations,
        c_puct,
        batch_size,
        eval_fn,
        num_workers as usize,
    );
    write_output(output_buf, output_size, &result);
    0
}

// ═══════════════════════════════════════════════════════════
// ONNX Runtime: disabled by default (Python callback is faster)
// Enable with: cargo build --release --features ort
// ═══════════════════════════════════════════════════════════
#[cfg(feature = "ort")]
mod ffi_ort {
    use std::ffi::CStr;
    use std::os::raw::c_char;
    use std::sync::OnceLock;
    use crate::net_infer::NetInfer;
    use crate::mcts_rs;

    static ORT_MODEL: OnceLock<NetInfer> = OnceLock::new();

    #[no_mangle]
    pub unsafe extern "C" fn ort_init(model_path: *const c_char) -> i32 {
        let path = match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        };
        let model = NetInfer::new(path);
        let _ = ORT_MODEL.set(model);
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn mcts_search_rust_handle_parallel_ort(
        handle: i32, num_simulations: i32, c_puct: f64, batch_size: i32,
        num_workers: i32, output_buf: *mut c_char, output_size: i32,
    ) -> i32 {
        let model = match ORT_MODEL.get() { Some(m) => m, None => return -2 };
        let game = match crate::GAMES.get(handle as usize) {
            Some(Some(g)) => g, _ => return -1,
        };
        let result = mcts_rs::mcts_search_parallel_core_ort(
            game, num_simulations, c_puct, batch_size, model, num_workers as usize,
        );
        crate::write_output(output_buf, output_size, &result);
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn mcts_search_rust_handle_ort(
        handle: i32, num_simulations: i32, c_puct: f64, batch_size: i32,
        output_buf: *mut c_char, output_size: i32,
    ) -> i32 {
        let model = match ORT_MODEL.get() { Some(m) => m, None => return -2 };
        let game = match crate::GAMES.get(handle as usize) {
            Some(Some(g)) => g, _ => return -1,
        };
        let result = mcts_rs::mcts_search_core_ort(
            game, num_simulations, c_puct, batch_size, model,
        );
        crate::write_output(output_buf, output_size, &result);
        0
    }
}

// ─── Alpha-Beta Search FFI ────────────────────────────────────

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

