"""advisor.py - operator advice from an inspection record.

Backends (env var ADVISOR_BACKEND):
  "ollama"      Llama 3.2 1B through Ollama: LAPTOP DEV STAND-IN, not ExecuTorch
  "placeholder" simple rules
The final on-device backend (Llama via ExecuTorch llama_main) is added on the Pi.
"""
import os
import requests
from advisor_prompt import build_messages

BACKEND = os.environ.get("ADVISOR_BACKEND", "ollama")
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

ACTIONS = {
    "open_circuit": "Remove the board and inspect or repair the broken track before use.",
    "short": "Remove the board and check for a solder or copper bridge between conductors.",
    "missing_hole": "Check the drilling step; the board may not accept its component.",
    "mouse_bite": "Inspect the track edge; reject if the notch is deep.",
    "spurious_copper": "Check for extra copper near neighbouring tracks that could cause a short.",
    "spur": "Low risk; clean up if convenient.",
}


def placeholder_advice(record):
    if record["defect_count"] == 0:
        return "SUMMARY: No defects found.\nACTION: Board can continue to the next stage."
    top = record["defects"][0]
    return (f"SUMMARY: {record['defect_count']} defect(s); worst is {top['defect']} "
            f"({top['severity']}) at {top['location']}.\nACTION: {ACTIONS[top['defect']]}")


def _ask_ollama(record):
    r = requests.post(OLLAMA_URL, timeout=120, json={
        "model": OLLAMA_MODEL,
        "messages": build_messages(record),
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 80},
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _keep_required_format(text):
    """Keep only the SUMMARY/ACTION part; return None if the format is missing."""
    if "SUMMARY:" in text and "ACTION:" in text:
        return text[text.index("SUMMARY:"):].strip()
    return None


def get_advice(record):
    """Returns (advice_text, source_label)."""
    if record["defect_count"] == 0:           # nothing to explain, skip the model
        return placeholder_advice(record), "rule (no defects)"
    if BACKEND == "ollama":
        try:
            text = _keep_required_format(_ask_ollama(record))
            if text:
                return text, "Llama 3.2 1B via Ollama (laptop dev stand-in, not ExecuTorch)"
            return placeholder_advice(record), "placeholder (Llama reply was not in the required format)"
        except Exception as e:
            return placeholder_advice(record), f"placeholder (Llama unavailable: {type(e).__name__})"
    return placeholder_advice(record), "placeholder"


if __name__ == "__main__":
    import time
    from inspector_core import build_record
    fake = [{"defect": "open_circuit", "confidence": 0.91, "bbox": [300, 200, 360, 240]}]
    rec = build_record("pcb_test", fake, 640, 640)
    t0 = time.time()
    text, source = get_advice(rec)
    print(f"{(time.time() - t0) * 1000:.0f} ms | {source}\n{text}")