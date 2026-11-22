"""app.py - operator UI. Run with: streamlit run app.py"""
import glob, os, time
import cv2, numpy as np, streamlit as st
from detector import Detector
from inspector_core import build_record
from advisor import get_advice
from logger import log_record, read_log

COLORS = {"High": (0, 0, 255), "Medium": (0, 165, 255), "Low": (0, 200, 0)}  # BGR

st.set_page_config(page_title="PCB Inspector", layout="wide")
st.title("On-device PCB quality inspector")


@st.cache_resource
def load_detector():
    return Detector()


det = load_detector()

src = st.sidebar.radio("Image source", ["Sample tiles", "Upload image", "Camera snapshot"])
img, name = None, ""

if src == "Sample tiles":
    only_defects = st.sidebar.checkbox("Only tiles that contain defects", True)
    files = sorted(glob.glob("data/tiled/images/val/*.jpg"))
    if only_defects:
        files = [f for f in files
                 if os.path.getsize(f.replace("images", "labels").replace(".jpg", ".txt")) > 0]
    name = st.sidebar.selectbox("Tile", files, format_func=lambda p: os.path.basename(p))
    if name:
        img = cv2.imread(name)
elif src == "Upload image":
    up = st.sidebar.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"])
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        name = up.name
else:
    snap = st.sidebar.camera_input("Point the camera at the board")
    if snap:
        img = cv2.imdecode(np.frombuffer(snap.getvalue(), np.uint8), 1)
        name = "camera_" + time.strftime("%H%M%S") + ".jpg"

if img is not None and st.button("Inspect", type="primary"):
    t0 = time.time()
    dets = det.detect_tiled(img)
    t1 = time.time()
    rec = build_record(os.path.basename(name), dets, img.shape[1], img.shape[0])
    advice, source = get_advice(rec)
    t2 = time.time()
    log_record(rec, advice=advice, timings={"detect": round((t1 - t0) * 1000, 1),
                                            "advice": round((t2 - t1) * 1000, 1)})

    shown = img.copy()
    for d in rec["defects"]:
        x1, y1, x2, y2 = d["bbox"]
        c = COLORS[d["severity"]]
        cv2.rectangle(shown, (x1, y1), (x2, y2), c, 2)
        cv2.putText(shown, f"{d['defect']} {d['confidence']}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

    left, right = st.columns(2)
    left.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), caption=name, use_container_width=True)
    with right:
        msg = f"Board result: {rec['result']}"
        {"PASS": st.success, "REVIEW": st.warning, "FAIL": st.error}[rec["result"]](msg)
        st.subheader("Advice")
        st.text(advice)
        st.caption(f"Advice source: {source} | detection {(t1 - t0) * 1000:.0f} ms | advice {(t2 - t1) * 1000:.0f} ms")
        if rec["defects"]:
            st.dataframe([{k: d[k] for k in ("defect", "severity", "confidence", "location", "reason")}
                          for d in rec["defects"]])

st.subheader("Inspection history (from local log)")
rows = read_log(last_n=15)
st.dataframe([{"time": r["timestamp"], "board": r["board_id"], "result": r["result"],
               "defects": r["defect_count"]} for r in reversed(rows)])