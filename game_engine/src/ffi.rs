/// C-compatible FFI API for Python integration via ctypes.
///
/// Protocol: all functions take and return JSON strings.
/// Input/Output via pre-allocated buffers to avoid memory management issues.
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use crate::board::*;
use crate::rules::*;

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
