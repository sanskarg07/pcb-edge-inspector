"""
inspector_core.py
Turns raw detector output into a structured inspection record:
severity per defect, a pass/review/fail result, and a JSON-ready dict.
"""
import json
from datetime import datetime

# ---------- Our severity rules (document these in the report) ----------
LEVELS = ["Low", "Medium", "High"]

BASE_SEVERITY = {
    "open_circuit":    "High",    # broken track: circuit is dead
    "short":           "High",    # unintended connection: can damage the board
    "missing_hole":    "Medium",
    "mouse_bite":      "Medium",
    "spurious_copper": "Medium",  # extra copper: may cause a short
    "spur":            "Low",     # small copper spike: often cosmetic
}

MIN_CONF = 0.25      # ignore detections the model is very unsure about
LARGE_AREA = 0.02    # box covers >= 2% of the image -> bump severity up
EDGE_MARGIN = 0.05   # centre within 5% of the border -> bump severity down


def bump(level, steps):
    """Move a severity level up (+1) or down (-1), staying inside Low..High."""
    i = LEVELS.index(level) + steps
    return LEVELS[max(0, min(len(LEVELS) - 1, i))]


def describe_location(cx, cy):
    """Turn a normalised centre point (0..1) into words like 'top-left'."""
    col = ["left", "center", "right"][min(int(cx * 3), 2)]
    row = ["top", "middle", "bottom"][min(int(cy * 3), 2)]
    return "center" if (row, col) == ("middle", "center") else f"{row}-{col}"


def rate_defect(det, frame_w, frame_h):
    """det = {'defect': str, 'confidence': float, 'bbox': [x1, y1, x2, y2]} in pixels."""
    x1, y1, x2, y2 = det["bbox"]
    cx, cy = ((x1 + x2) / 2) / frame_w, ((y1 + y2) / 2) / frame_h
    area = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)

    level = BASE_SEVERITY[det["defect"]]
    reasons = [f"{det['defect']} is normally {level}"]

    if area >= LARGE_AREA:
        level = bump(level, +1)
        reasons.append("large defect, raised one level")

    # near_edge = cx < EDGE_MARGIN or cx > 1 - EDGE_MARGIN or cy < EDGE_MARGIN or cy > 1 - EDGE_MARGIN
    # if near_edge:
    #     level = bump(level, -1)
    #     reasons.append("near board edge, lowered one level")

    return {
        "defect": det["defect"],
        "severity": level,
        "confidence": round(float(det["confidence"]), 2),
        "bbox": [int(v) for v in det["bbox"]],
        "location": describe_location(cx, cy),
        "area_pct": round(area * 100, 2),
        "reason": "; ".join(reasons),
    }


def board_result(defects):
    """Overall decision for the board."""
    levels = {d["severity"] for d in defects}
    if "High" in levels:
        return "FAIL"
    if "Medium" in levels:
        return "REVIEW"
    return "PASS"


def build_record(board_id, detections, frame_w, frame_h):
    """Main function: raw detections in, structured record out."""
    kept = [d for d in detections if d["confidence"] >= MIN_CONF]
    defects = [rate_defect(d, frame_w, frame_h) for d in kept]
    defects.sort(key=lambda d: LEVELS.index(d["severity"]), reverse=True)  # worst first
    return {
        "board_id": board_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "result": board_result(defects),
        "defect_count": len(defects),
        "defects": defects,
    }


if __name__ == "__main__":
    # Fake detector output, just to test the rules (no model needed)
    fake = [
        {"defect": "open_circuit", "confidence": 0.91, "bbox": [300, 200, 360, 240]},
        {"defect": "spur",         "confidence": 0.62, "bbox": [10, 300, 40, 320]},
        {"defect": "mouse_bite",   "confidence": 0.12, "bbox": [500, 500, 530, 530]},  # too unsure, dropped
    ]
    print(json.dumps(build_record("pcb_001", fake, 640, 640), indent=2))