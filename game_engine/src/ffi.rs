/// C-compatible FFI API for Python integration via ctypes.
///
/// Protocol: all functions take and return JSON strings.
/// Input/Output via pre-allocated buffers to avoid memory management issues.
use std::ffi::{CStr, CString};
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

