"""
advisor_prompt.py
Builds the text prompt we give Llama from an inspection record.
Llama never sees pixels, only these structured facts.
"""

SYSTEM = (
    "You are a PCB quality-control assistant. You receive structured inspection "
    "findings. Use only the facts given. Do not invent defects, causes or measurements. "
    "Be short and specific."
)

# One-line plain-English meaning of each defect, so Llama stays grounded.
DEFECT_NOTES = {
    "open_circuit":    "a break in a copper track, so current cannot flow",
    "short":           "an unintended connection between two conductors",
    "missing_hole":    "a drilled hole that is absent",
    "mouse_bite":      "small notches along the board or track edge",
    "spurious_copper": "extra copper where there should be none",
    "spur":            "a small copper spike off a track",
}


def build_messages(record):
    """Return chat messages (system + user) for Llama."""
    if record["defect_count"] == 0:
        facts = "No defects were found. Board result: PASS."
    else:
        lines = []
        for i, d in enumerate(record["defects"], 1):
            lines.append(
                f"{i}. {d['defect']} ({DEFECT_NOTES[d['defect']]}), "
                f"severity {d['severity']}, confidence {d['confidence']}, "
                f"location {d['location']}, size {d['area_pct']}% of image."
            )
        facts = f"Board result: {record['result']}.\n" + "\n".join(lines)

    user = (
        f"Findings for board {record['board_id']}:\n{facts}\n\n"
        "Reply in exactly this format, with no extra text:\n"
        "SUMMARY: <one sentence on what was found>\n"
        "ACTION: <one sentence telling the operator what to do next>\n"
        "Keep the whole reply under 40 words."
    )
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


if __name__ == "__main__":
    from inspector_core import build_record
    fake = [{"defect": "open_circuit", "confidence": 0.91, "bbox": [300, 200, 360, 240]}]
    for m in build_messages(build_record("pcb_001", fake, 640, 640)):
        print(f"[{m['role']}]\n{m['content']}\n")