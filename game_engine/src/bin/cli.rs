/// CLI 工具：通过 stdin/stdout JSON 协议提供游戏核心服务
///
/// 协议：
///   {"cmd": "new_game", "seed": 42} → {"board": ..., "pieces": ..., ...}
///   {"cmd": "step", "action": 5, "state": ...} → {"state": ..., "reward": 0.0, "terminated": false}
///   {"cmd": "legal_actions", "state": ...} → {"actions": [0,1,2,...]}

use std::io::{self, BufRead, Write};
use game_engine::*;
use serde_json::{json, Value};

fn deserialize_state(value: Value) -> Option<GameState> {
    let mut state = serde_json::from_value::<GameState>(value).ok()?;
    state.board.rebuild_index_for_json();
    Some(state)
}

fn handle_command(cmd: &Value) -> Value {
    let cmd_type = cmd["cmd"].as_str().unwrap_or("");

    match cmd_type {
        "new_game" => {
            let seed = cmd["seed"].as_u64().unwrap_or(42);
            let seq = generate_sequence(seed);
            let board = Board::new(&seq);
            let state = GameState::new(board);
            let state_json = serde_json::to_value(&state).unwrap_or(json!({}));
            json!({"ok": true, "state": state_json})
        }

        "step" => {
            if let Some(mut state) = deserialize_state(cmd["state"].clone()) {
                let action = cmd["action"].as_u64().unwrap_or(0) as usize;
                let (reward, terminated) = state.step(action);
                let state_json = serde_json::to_value(&state).unwrap_or(json!({}));
                json!({"ok": true, "state": state_json, "reward": reward, "terminated": terminated})
            } else {
                json!({"ok": false, "error": "invalid state"})
            }
        }

        "legal_actions" => {
            if let Some(state) = deserialize_state(cmd["state"].clone()) {
                let actions = state.get_legal_actions();
                json!({"ok": true, "actions": actions})
            } else {
                json!({"ok": false, "error": "invalid state"})
            }
        }

        "evaluate" => {
            if let Some(state) = deserialize_state(cmd["state"].clone()) {
                let legal_count = state.get_legal_actions().len();
                json!({"ok": true, "legal_count": legal_count, "scores": state.scores})
            } else {
                json!({"ok": false, "error": "invalid state"})
            }
        }

        _ => json!({"ok": false, "error": format!("unknown cmd: {}", cmd_type)}),
    }
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();

    for line in stdin.lock().lines() {
        match line {
            Ok(text) => {
                let text = text.trim().to_string();
                if text.is_empty() || text == "exit" {
                    break;
                }
                match serde_json::from_str::<Value>(&text) {
                    Ok(cmd) => {
                        let result = handle_command(&cmd);
                        let output = serde_json::to_string(&result).unwrap_or_default();
                        let mut out = stdout.lock();
                        let _ = writeln!(out, "{}", output);
                        let _ = out.flush();
                    }
                    Err(e) => {
                        let mut out = stdout.lock();
                        let _ = writeln!(out, r#"{{"ok": false, "error": "{}"}}"#, e);
                        let _ = out.flush();
                    }
                }
            }
            Err(_) => break,
        }
    }
}
