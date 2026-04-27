"""
Model Registry — 持久化模型元数据和评估结果。
提供基于 ELO / 胜率的智能模型选择，供 server 和训练脚本共用。

Registry 文件位置: models/model_registry.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

# 默认 registry 结构
_DEFAULT_REGISTRY: dict[str, Any] = {
    "version": 1,
    "models": [],
}


# =============================================================================
# Registry I/O
# =============================================================================

def get_registry() -> dict[str, Any]:
    """加载 registry JSON。文件不存在时返回默认空结构。"""
    if not REGISTRY_PATH.exists():
        return dict(_DEFAULT_REGISTRY)
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT_REGISTRY)


def save_registry(registry: dict[str, Any]) -> None:
    """写回 registry JSON。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =============================================================================
# 模型注册与更新
# =============================================================================

def register_model(
    model_id: str,
    model_type: str,  # "ppo" | "alphazero"
    file_path: str,
    generation: Optional[int] = None,
    iteration: Optional[int] = None,
    arch: Optional[str] = None,  # "mlp" | "resnet" (仅 alphazero)
) -> dict[str, Any]:
    """
    注册一个新模型（若 model_id 已存在则更新基本信息）。

    返回该模型的 entry dict。
    """
    registry = get_registry()
    models = registry.setdefault("models", [])

    # 找已有 entry
    entry = None
    for m in models:
        if m.get("id") == model_id:
            entry = m
            break

    if entry is None:
        entry = {
            "id": model_id,
            "type": model_type,
            "file": file_path,
            "created_at": _now_iso(),
        }
        models.append(entry)

    # 更新基本信息
    entry["type"] = model_type
    entry["file"] = file_path
    if generation is not None:
        entry["generation"] = generation
    if iteration is not None:
        entry["iteration"] = iteration
    if arch is not None:
        entry["arch"] = arch
    entry.setdefault("created_at", _now_iso())

    save_registry(registry)
    return entry


def update_evaluation(
    model_id: str,
    eval_data: dict[str, Any],
    *,
    append_history: bool = True,
) -> Optional[dict[str, Any]]:
    """
    更新模型的评估结果。
    eval_data 示例:
    {
        "vs_random": {"win": 0.95, "lose": 0.03, "draw": 0.02},
        "vs_prev": {"win": 0.55, "lose": 0.30, "draw": 0.15, "opponent": "ppo_gen_4"},
        "elo": 1248,
    }

    当 append_history=True 且 eval_data 包含 "elo" 字段时，
    自动将 ELO 快照追加到该模型的 elo_history 数组中，用于追踪 ELO 变化趋势。
    返回更新后的 entry，若模型不存在则返回 None。
    """
    registry = get_registry()
    for m in registry.setdefault("models", []):
        if m.get("id") == model_id:
            m["eval"] = eval_data
            m["evaluated_at"] = _now_iso()

            # ELO 历史追踪
            if append_history:
                elo = eval_data.get("elo")
                if elo is not None:
                    history = m.setdefault("elo_history", [])
                    history.append({
                        "elo": round(float(elo), 1),
                        "timestamp": _now_iso(),
                    })
                    # 保留最近 100 条记录，防止无限增长
                    if len(history) > 100:
                        m["elo_history"] = history[-100:]

            save_registry(registry)
            return m
    return None


# =============================================================================
# 查询
# =============================================================================

def get_best_model(criteria: str = "elo") -> Optional[tuple[str, str]]:
    """
    基于 criteria 返回最优模型的 (file_path, model_type)。

    选择策略（三级 fallback）:
    1. criteria="elo": 有 eval.elo 字段的 → 取 ELO 最高
    2. 没 ELO: 有 eval.vs_random.win → 取 vs_random 胜率最高
    3. 都没评估数据: 取 generation/iteration 编号最大（兼容旧行为）

    返回 None 表示无任何模型。
    """
    registry = get_registry()
    models = registry.get("models", [])
    if not models:
        return None

    # --- 策略 1: ELO 最高 ---
    if criteria == "elo":
        scored = []
        for m in models:
            elo = _get_elo(m)
            if elo is not None:
                scored.append((elo, m))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][1]
            return (best["file"], best["type"])

    # --- 策略 2: vs_random 胜率最高 ---
    scored = []
    for m in models:
        wr = _get_vs_random_winrate(m)
        if wr is not None:
            scored.append((wr, m))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        return (best["file"], best["type"])

    # --- 策略 3: 编号最大 ---
    return _find_highest_numbered(models)


def get_best_model_info(criteria: str = "elo") -> Optional[dict[str, Any]]:
    """返回最优模型的完整 info dict（供 API 使用）。"""
    registry = get_registry()
    models = registry.get("models", [])
    if not models:
        return None

    if criteria == "elo":
        scored = []
        for m in models:
            elo = _get_elo(m)
            if elo is not None:
                scored.append((elo, m))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return dict(scored[0][1])

    scored = []
    for m in models:
        wr = _get_vs_random_winrate(m)
        if wr is not None:
            scored.append((wr, m))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return dict(scored[0][1])

    best_file = _find_highest_numbered(models)
    if best_file:
        for m in models:
            if m["file"] == best_file[0]:
                return dict(m)
    return None


def list_models() -> list[dict[str, Any]]:
    """返回所有已注册模型的列表（按创建时间降序）。"""
    registry = get_registry()
    models = list(registry.get("models", []))
    models.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return models


def get_model(model_id: str) -> Optional[dict[str, Any]]:
    """按 ID 查询单个模型。"""
    registry = get_registry()
    for m in registry.get("models", []):
        if m.get("id") == model_id:
            return dict(m)
    return None


# =============================================================================
# 内部工具
# =============================================================================

def _get_elo(entry: dict) -> Optional[float]:
    """从 entry 中提取 ELO，不存在返回 None。"""
    ev = entry.get("eval")
    if ev and isinstance(ev, dict):
        elo = ev.get("elo")
        if elo is not None:
            return float(elo)
    return None


def _get_vs_random_winrate(entry: dict) -> Optional[float]:
    """从 entry 中提取 vs_random 胜率，不存在返回 None。"""
    ev = entry.get("eval")
    if ev and isinstance(ev, dict):
        vr = ev.get("vs_random")
        if vr and isinstance(vr, dict):
            win = vr.get("win")
            if win is not None:
                return float(win)
    return None


def _find_highest_numbered(models: list[dict]) -> Optional[tuple[str, str]]:
    """
    按 generation 或 iteration 编号取最大（兼容旧逻辑）。
    先按 gen 排，再按 iter 排。
    """
    best = None
    best_priority = -1

    for m in models:
        gen = m.get("generation")
        it = m.get("iteration")
        prio = gen if gen is not None else (it if it is not None else -1)
        if prio > best_priority:
            best = (m["file"], m["type"])
            best_priority = prio

    return best


def _now_iso() -> str:
    """返回 ISO 8601 时间戳字符串。"""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
