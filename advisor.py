"""advisor.py - produces operator advice from an inspection record.
PLACEHOLDER (rule-based). Will be replaced by Llama 3.2 via ExecuTorch."""

ACTIONS = {
    "open_circuit": "Remove the board and inspect or repair the broken track before use.",
    "short": "Remove the board and check for a solder or copper bridge between conductors.",
    "missing_hole": "Check the drilling step; the board may not accept its component.",
    "mouse_bite": "Inspect the track edge; reject if the notch is deep.",
    "spurious_copper": "Check for extra copper near neighbouring tracks that could cause a short.",
    "spur": "Low risk; clean up if convenient.",
}


def get_advice(record):
    """Returns (advice_text, source_label)."""
    if record["defect_count"] == 0:
        return "SUMMARY: No defects found.\nACTION: Board can continue to the next stage.", "placeholder"
    top = record["defects"][0]
    summary = (f"SUMMARY: {record['defect_count']} defect(s); worst is {top['defect']} "
               f"({top['severity']}) at {top['location']}.")
    return summary + "\nACTION: " + ACTIONS[top["defect"]], "placeholder"