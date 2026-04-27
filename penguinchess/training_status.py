"""
Training Status — shared module for tracking active training progress.

Training scripts (train_ppo.py, train_alphazero.py) update this file periodically.
The server API reads it to show real-time training progress on the frontend dashboard.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_STATUS_FILE: Path = Path(__file__).resolve().parent.parent / "models" / "training_status.json"


def update_status(**kwargs: Any) -> None:
    """更新训练状态文件。训练脚本在每次迭代/评估后调用。"""
    data = {
        "is_training": True,
        "last_updated": _now_iso(),
        **kwargs,
    }
    _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATUS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def clear_status() -> None:
    """清除训练状态（训练完成时调用）。"""
    data = {"is_training": False, "last_updated": _now_iso()}
    _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATUS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_status() -> dict[str, Any]:
    """获取当前训练状态。文件不存在或损坏时返回 idle 状态。"""
    try:
        return json.loads(_STATUS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"is_training": False}


def get_training_metrics() -> dict[str, Any]:
    """
    从 Model Registry 聚合训练指标，用于前端图表展示。
    返回:
    {
        "generations": [1, 2, 3, ...],
        "elos": [1200, 1214, 1218, ...],
        "win_rates": [0.5, 0.575, 0.61, ...],
        "models": [...]
    }
    """
    from penguinchess.model_registry import get_registry

    registry = get_registry()
    models = registry.get("models", [])

    # 按类型和编号排序
    ppo_models = []
    az_models = []
    for m in models:
        if m.get("type") == "ppo" and m.get("generation") is not None:
            ppo_models.append(m)
        elif m.get("type") == "alphazero" and m.get("iteration") is not None:
            az_models.append(m)

    ppo_models.sort(key=lambda m: m["generation"])
    az_models.sort(key=lambda m: m["iteration"])

    def _extract_metrics(model_list: list[dict]) -> dict:
        gens = []
        elos = []
        win_rates = []
        for m in model_list:
            ev = m.get("eval") or {}
            gen = m.get("generation") or m.get("iteration") or 0
            gens.append(gen)
            elos.append(ev.get("elo"))
            vr = ev.get("vs_random") or {}
            win_rates.append(vr.get("win"))
        return {"generations": gens, "elos": elos, "win_rates": win_rates}

    result = {"ppo": _extract_metrics(ppo_models), "alphazero": _extract_metrics(az_models)}
    result["models"] = [
        {"id": m.get("id"), "type": m.get("type"), "elo": _get_elo_from_entry(m)}
        for m in models
    ]
    return result


def _get_elo_from_entry(entry: dict) -> float | None:
    ev = entry.get("eval")
    if ev and isinstance(ev, dict):
        return ev.get("elo")
    return None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
