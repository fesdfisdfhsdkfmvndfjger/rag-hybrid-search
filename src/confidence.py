from __future__ import annotations
import numpy as np

def is_confident(scores, abs_threshold: float = 0.25, gap_threshold: float = 0.04) -> bool:
    """Returns True if retrieval quality is sufficient to attempt an answer."""
    if len(scores) == 0:
        return False
    best = float(scores[0])
    if best < abs_threshold:
        return False
    if gap_threshold > 0 and len(scores) >= 2:
        second = float(scores[1])
        if (best - second) < gap_threshold and best < 0.50:
            return False
    return True

def confidence_label(score: float) -> str:
    if score >= 0.65: return "Very High"
    if score >= 0.50: return "High"
    if score >= 0.35: return "Medium"
    if score >= 0.25: return "Low"
    return "Very Low"

def confidence_color(score: float) -> str:
    if score >= 0.65: return "#22c55e"
    if score >= 0.50: return "#84cc16"
    if score >= 0.35: return "#f59e0b"
    if score >= 0.25: return "#f97316"
    return "#ef4444"