"""
logger.py
Appends every inspection record to a local JSONL file and reads it back.
"""
import json, os

LOG_PATH = "logs/inspections.jsonl"


def log_record(record, advice="", timings=None, path=LOG_PATH):
    """Save one inspection. 'advice' is Llama's text; 'timings' holds latency numbers."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = dict(record)
    entry["advice"] = advice
    entry["timings_ms"] = timings or {}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_log(path=LOG_PATH, last_n=None):
    """Return all saved inspections (or just the last N), oldest first."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[-last_n:] if last_n else rows


if __name__ == "__main__":
    from inspector_core import build_record
    fake = [{"defect": "short", "confidence": 0.88, "bbox": [100, 100, 160, 150]}]
    rec = build_record("pcb_test", fake, 640, 640)
    log_record(rec, advice="(advice will go here)", timings={"detect": 42, "llama": 0})
    print("saved. entries in log:", len(read_log()))