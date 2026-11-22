import cv2, time, sys
from detector import Detector

path = sys.argv[1]
img = cv2.imread(path)
print("image size:", img.shape[1], "x", img.shape[0])
det = Detector()
det.detect(img[:640, :640])            # warm-up
t0 = time.time()
n = len(det.detect_tiled(img))
print(f"detect_tiled: {(time.time()-t0)*1000:.0f} ms, {n} defects")
for stride in (480, 560, 640):
    t0 = time.time()
    det.detect_tiled(img, stride=stride)
    print(f"stride {stride}: {(time.time()-t0)*1000:.0f} ms")