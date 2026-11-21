import os, glob, cv2

SRC = "data/raw/PKU-Market-PCB"
DST = "data/tiled"
TILE, STRIDE = 640, 480          # 160 px overlap
MIN_VISIBLE = 0.6                # keep a box if >=60% of it is inside the tile
KEEP_EMPTY = 0.15                # keep 15% of defect-free tiles as negatives

def starts(length):
    pos = list(range(0, max(length - TILE, 0) + 1, STRIDE))
    if pos[-1] + TILE < length:
        pos.append(length - TILE)
    return pos

def read_labels(path):
    boxes = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 5:
                c, x, y, w, h = int(p[0]), *map(float, p[1:])
                boxes.append((c, x, y, w, h))
    return boxes

import random
random.seed(0)
stats = {}
for split in ("train", "val"):
    os.makedirs(f"{DST}/images/{split}", exist_ok=True)
    os.makedirs(f"{DST}/labels/{split}", exist_ok=True)
    n_tiles = n_boxes = 0
    for img_path in sorted(glob.glob(f"{SRC}/images/{split}/*.jpg")):
        name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        boxes = [(c, (x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H)
                 for c, x, y, w, h in read_labels(f"{SRC}/labels/{split}/{name}.txt")]
        for ty in starts(H):
            for tx in starts(W):
                out = []
                for c, x1, y1, x2, y2 in boxes:
                    ix1, iy1 = max(x1, tx), max(y1, ty)
                    ix2, iy2 = min(x2, tx+TILE), min(y2, ty+TILE)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    if (ix2-ix1)*(iy2-iy1) / ((x2-x1)*(y2-y1)) < MIN_VISIBLE:
                        continue
                    out.append(f"{c} {((ix1+ix2)/2-tx)/TILE:.6f} {((iy1+iy2)/2-ty)/TILE:.6f} "
                               f"{(ix2-ix1)/TILE:.6f} {(iy2-iy1)/TILE:.6f}")
                if not out and (split == "val" or random.random() > KEEP_EMPTY):
                    if split == "train":
                        continue
                tile = img[ty:ty+TILE, tx:tx+TILE]
                if tile.shape[0] != TILE or tile.shape[1] != TILE:
                    continue
                stem = f"{name}_{tx}_{ty}"
                cv2.imwrite(f"{DST}/images/{split}/{stem}.jpg", tile)
                open(f"{DST}/labels/{split}/{stem}.txt", "w").write("\n".join(out))
                n_tiles += 1; n_boxes += len(out)
    stats[split] = (n_tiles, n_boxes)
    print(split, "tiles:", n_tiles, "boxes:", n_boxes)

open("data/pcb.yaml", "w").write(
"""path: data/tiled
train: images/train
val: images/val
nc: 6
names: ['missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper']
""")
print("wrote data/pcb.yaml")