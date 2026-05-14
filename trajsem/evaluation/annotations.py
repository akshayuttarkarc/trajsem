from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import pandas as pd

from trajsem.schemas import SemanticEvent


def load_expert_annotations(path_or_file) -> list[dict]:
    if hasattr(path_or_file, "read"):
        raw = path_or_file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
    else:
        raw = Path(path_or_file).read_text()
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def interval_iou(a, b) -> float:
    a0, a1 = float(a[0]), float(a[1])
    b0, b1 = float(b[0]), float(b[1])
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union else 0.0


def compare_events(predicted: list[SemanticEvent], expert: list[dict], iou_threshold: float = 0.25) -> tuple[pd.DataFrame, dict]:
    rows = []
    matched_pred = set()
    matched_exp = set()
    for ei, ann in enumerate(expert):
        best = None
        for pi, ev in enumerate(predicted):
            if ev.event_type != ann.get("event_type"):
                continue
            score = interval_iou(ev.time_window_ns, ann.get("time_window_ns", [0, 0]))
            if best is None or score > best[0]:
                best = (score, pi, ev)
        if best and best[0] >= iou_threshold:
            matched_exp.add(ei)
            matched_pred.add(best[1])
            rows.append({"expert_event_type": ann.get("event_type"), "predicted_event_type": best[2].event_type, "expert_time_window_ns": ann.get("time_window_ns"), "predicted_time_window_ns": best[2].time_window_ns, "iou": round(best[0], 3), "status": "matched"})
        else:
            rows.append({"expert_event_type": ann.get("event_type"), "predicted_event_type": None, "expert_time_window_ns": ann.get("time_window_ns"), "predicted_time_window_ns": None, "iou": 0.0, "status": "missed"})
    for pi, ev in enumerate(predicted):
        if pi not in matched_pred:
            rows.append({"expert_event_type": None, "predicted_event_type": ev.event_type, "expert_time_window_ns": None, "predicted_time_window_ns": ev.time_window_ns, "iou": 0.0, "status": "extra_prediction"})
    tp = len(matched_pred)
    fp = len(predicted) - tp
    fn = len(expert) - len(matched_exp)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return pd.DataFrame(rows), {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3), "iou_threshold": iou_threshold}


def annotation_template() -> str:
    return json.dumps([
        {"event_type": "ligand_pose_drift", "time_window_ns": [55.0, 82.0], "entities": {"ligand": "LIG"}, "notes": "Expert rationale here."},
        {"event_type": "transient_pocket_opening", "time_window_ns": [61.0, 76.0], "entities": {"region": "binding_site"}, "notes": "Expert rationale here."}
    ], indent=2)
