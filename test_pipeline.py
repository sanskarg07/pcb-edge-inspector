"""test_pipeline.py - detector -> rules -> log, on a few val tiles."""
import glob, time, cv2
from detector import Detector
from inspector_core import build_record
from logger import log_record

det = Detector()
tiles = sorted(glob.glob("data/tiled/images/val/*.jpg"))[::40][:10]
for path in tiles:
    img = cv2.imread(path)
    t0 = time.time()
    dets = det.detect(img)
    ms = (time.time() - t0) * 1000
    rec = build_record(path.split("/")[-1], dets, img.shape[1], img.shape[0])
    log_record(rec, advice="(not yet)", timings={"detect": round(ms, 1)})
    print(rec["board_id"], rec["result"], rec["defect_count"], f"{ms:.0f} ms")