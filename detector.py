"""detector.py - runs best.onnx with ONNX Runtime (vision pipeline)."""
import cv2, numpy as np, onnxruntime as ort

NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']


class Detector:
    def __init__(self, path="models/best.onnx", conf=0.25, iou=0.45, size=640):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.conf, self.iou, self.size = conf, iou, size

    def detect(self, img):
        """img: BGR image (numpy). Returns [{'defect','confidence','bbox':[x1,y1,x2,y2]}]."""
        h, w = img.shape[:2]
        s = self.size
        r = min(s / h, s / w)
        nh, nw = round(h * r), round(w * r)
        top, left = (s - nh) // 2, (s - nw) // 2
        canvas = np.full((s, s, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        out = self.sess.run(None, {self.inp: blob})[0][0].T      # (8400, 10)
        scores = out[:, 4:]
        cls, conf = scores.argmax(1), scores.max(1)
        keep = conf >= self.conf
        out, cls, conf = out[keep], cls[keep], conf[keep]
        if len(out) == 0:
            return []

        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        x1 = (cx - bw / 2 - left) / r
        y1 = (cy - bh / 2 - top) / r
        x2 = (cx + bw / 2 - left) / r
        y2 = (cy + bh / 2 - top) / r
        boxes = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
        idx = cv2.dnn.NMSBoxes(boxes, conf.tolist(), self.conf, self.iou)

        dets = []
        for i in np.array(idx).flatten():
            dets.append({"defect": NAMES[int(cls[i])], "confidence": float(conf[i]),
                         "bbox": [float(max(0, x1[i])), float(max(0, y1[i])),
                                  float(min(w, x2[i])), float(min(h, y2[i]))]})
        return dets