"""TRACE-10: a controlled long-video benchmark in which the label is carried only by a brief
out-and-back micro-event of one object among look-alike distractors, under camera jitter and
optional camera bumps. Speed classes (fast vs slow jab) differ only in elapsed time."""
import numpy as np, cv2

R, WORLD, VIEW, OFF = 192, 160, 112, 24
CLASSES = ["jab-right-fast", "jab-left-fast", "jab-up-fast", "jab-down-fast", "tilt-cw-fast",
           "tilt-ccw-fast", "pulse-grow-fast", "pulse-shrink-fast", "jab-right-slow", "jab-left-slow"]
SHAPES = [np.array(p, np.float32) for p in [
    [(-1, -.6), (1, 0), (-1, .6), (-.4, 0)],
    [(-1, -1), (-.3, -1), (-.3, .4), (1, .4), (1, 1), (-1, 1)],
    [(-1, -.7), (1, -.7), (1, .7), (.2, .7), (.2, .1), (-.2, .1), (-.2, .7), (-1, .7)],
    [(-1, -1), (-.6, -1), (-.6, -.1), (1, .3), (-.6, .7), (-.6, 1), (-1, 1)]]]


def _bg(rng):
    n = rng.normal(0, 1, (10, 10, 3)).astype(np.float32)
    b = cv2.resize(n, (WORLD, WORLD), interpolation=cv2.INTER_CUBIC)
    b += 0.35 * cv2.resize(rng.normal(0, 1, (40, 40, 3)).astype(np.float32), (WORLD, WORLD), interpolation=cv2.INTER_CUBIC)
    b = (b - b.min()) / (b.max() - b.min() + 1e-6)
    return (70 + 110 * b).astype(np.uint8)


def render(vid, label, seed=12345):
    rng = np.random.default_rng(seed * 100003 + vid)
    bg = _bg(rng)
    nobj = int(rng.integers(2, 4))  # actor + 1..2 distractors
    objs = []
    for k in range(nobj):
        objs.append(dict(shape=SHAPES[int(rng.integers(0, 4))], size=float(rng.uniform(12, 15)),
                         col=tuple(int(c) for c in rng.integers(20, 255, 3)),
                         p=rng.uniform(52, 108, 2), v=rng.uniform(-.25, .25, 2),
                         th=float(rng.uniform(0, 360)), w=float(rng.uniform(-.4, .4))))
    fast = label < 8
    d = int(rng.integers(10, 17)) if fast else int(rng.integers(64, 97))
    t0 = int(rng.integers(12, R - 12 - d))
    # camera: slow random walk + optional bump (global motion burst carrying no label information)
    cam = np.clip(np.cumsum(rng.normal(0, .25, (R, 2)), 0), -6, 6)
    shake = rng.random() < 0.5
    ts = int(rng.integers(4, R - 10)) if shake else -1
    if shake:
        dirv = rng.normal(0, 1, 2); dirv /= np.linalg.norm(dirv) + 1e-6
        for t in range(ts, ts + 6):
            cam[t] += 10 * np.sin(np.pi * (t - ts) / 6) * dirv
    frames = np.empty((R, VIEW, VIEW, 3), np.uint8)
    D = 34.0
    for t in range(R):
        can = bg.copy()
        for k, o in enumerate(objs):
            in_evt = k == 0 and t0 <= t < t0 + d
            if not in_evt:
                o["p"] = o["p"] + o["v"]
                for a in range(2):
                    if o["p"][a] < 52 or o["p"][a] > 108:
                        o["v"][a] *= -1; o["p"][a] = np.clip(o["p"][a], 52, 108)
                o["th"] += o["w"]
            p, th, sc = o["p"].copy(), o["th"], 1.0
            if in_evt:
                u = (t - t0 + 1) / d
                bump = 0.5 * (1 - np.cos(2 * np.pi * u))
                if label in (0, 8): p[0] += D * bump
                elif label in (1, 9): p[0] -= D * bump
                elif label == 2: p[1] -= D * bump
                elif label == 3: p[1] += D * bump
                elif label in (4, 5):
                    th += 90 * bump if label == 4 else -90 * bump
                elif label == 6: sc = 1 + 1.0 * bump
                elif label == 7: sc = 1 - .6 * bump
            c, s_ = np.cos(np.radians(th)), np.sin(np.radians(th))
            pts = o["shape"] @ np.array([[c, s_], [-s_, c]], np.float32) * o["size"] * sc + p
            poly = [np.round(pts * 16).astype(np.int32)]
            cv2.fillPoly(can, poly, o["col"], lineType=cv2.LINE_AA, shift=4)
            if k == 0:   # actor outline
                cv2.polylines(can, poly, True, (15, 15, 15), 2, lineType=cv2.LINE_AA, shift=4)
        oy, ox = int(round(OFF + cam[t, 1])), int(round(OFF + cam[t, 0]))
        frames[t] = can[oy:oy + VIEW, ox:ox + VIEW]
    return frames, dict(t0=t0, t1=t0 + d, shake=ts)
