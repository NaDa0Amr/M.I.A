

import os
import cv2
import glob
import argparse
import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CLASS_BLUE = 0
CLASS_RED = 1

# HSV colour ranges. Red wraps around hue 0/180, so it needs two ranges.
# S (saturation) and V (value) lower bounds are kept fairly high to reject
# washed-out shadows, dull backgrounds and pale reflections.
COLOR_RANGES = {
    "red": {
        "class_id": CLASS_RED,
        "ranges": [
            (np.array([0, 90, 60]), np.array([9, 255, 255])),
            (np.array([171, 90, 60]), np.array([180, 255, 255])),
        ],
    },
    "blue": {
        "class_id": CLASS_BLUE,
        "ranges": [
            (np.array([95, 80, 40]), np.array([130, 255, 255])),
        ],
    },
}

MIN_AREA_FRACTION = 0.0008   # a detected blob must cover at least this
                              # fraction of the image area (rejects tiny
                              # colour speckles / noise)
MIN_RADIUS_PX = 10           # absolute floor in pixels, as a safety net for
                              # very small or very large images
MAX_AREA_FRACTION = 0.20     # rejects absurdly large blobs (e.g. sky/wall
                              # merged with the ball, or an entire coloured
                              # wall being picked up). Calibrated so it's
                              # comfortably above the biggest genuine close-up
                              # ball (~13% of frame) but well below a blob
                              # that has fused with a same-coloured background.

# Real soccer-ball textures (pentagons/stars printed in a second colour)
# punch dark notches into the raw contour, which tanks a "perimeter-based"
# circularity score even though the ball itself is perfectly round. To stay
# robust to that we measure roundness on the *convex hull* of the blob
# (which smooths those notches back out) and separately check "solidity"
# (raw_area / hull_area) to make sure the blob is still mostly filled in
# -- this rejects non-round shapes (limbs, clothing, tiles) that happen to
# have a roundish convex hull but are mostly empty/concave.
MIN_HULL_CIRCULARITY = 0.82  # 1.0 = perfect circle, measured on convex hull
MIN_SOLIDITY = 0.72          # raw contour area / convex-hull area

# Small blobs are inherently more ambiguous (a few dozen pixels of noise can
# look "roughly round" by chance), while a big blob that is even moderately
# round is very unlikely to be anything other than a real ball. So small
# candidates are held to a stricter combined score before being accepted.
SMALL_RADIUS_PX = 25
SMALL_RADIUS_MIN_SCORE = 0.88

# A blob whose hull circularity falls in this "near miss" band is treated as
# a possible partial fusion with a same-coloured neighbour (e.g. a ball right
# in front of a similarly red piece of clothing). Rather than throwing the
# whole blob away, we look for the largest circle that fits entirely inside
# it (via the distance transform) -- for a ball-plus-appendage shape this
# reliably lands on the ball itself, since it's the "fattest" round part of
# the blob, while a genuinely non-ball shape (e.g. a flat corner of a bin)
# won't have a large inscribed circle relative to its own area.
NEAR_MISS_CIRCULARITY = 0.65
MIN_INSCRIBED_AREA_RATIO = 0.35  # inscribed-circle area / blob area
MIN_FALLBACK_SCORE = 0.58        # circularity * solidity threshold for
                                  # fallback (fused-blob) detections


# --------------------------------------------------------------------------
# Core detection logic
# --------------------------------------------------------------------------

def _clean_mask(mask):
    """Remove speckle noise and close small gaps in the colour mask."""
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big, iterations=2)
    return mask


def _build_color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, lower, upper)
    return mask


def _candidate_circles(mask, img_area):
    """Find round blobs in a binary mask and return (cx, cy, r, score)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue

        area_frac = area / img_area
        if area_frac < MIN_AREA_FRACTION or area_frac > MAX_AREA_FRACTION:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        if hull_area <= 0 or hull_perimeter <= 0:
            continue

        hull_circularity = 4 * np.pi * hull_area / (hull_perimeter ** 2)
        solidity = area / hull_area
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h else 0

        if hull_circularity < MIN_HULL_CIRCULARITY:
            # Not round enough on its own -- check whether it's a ball
            # fused with a same-coloured neighbour (see NEAR_MISS_CIRCULARITY
            # comment above) before giving up on this blob entirely.
            if hull_circularity >= NEAR_MISS_CIRCULARITY and solidity >= MIN_SOLIDITY:
                fitted = _fit_inscribed_circle(mask, cnt, area)
                if fitted is not None:
                    cx, cy, r = fitted
                    score = hull_circularity * solidity
                    small_ok = not (r < SMALL_RADIUS_PX and score < SMALL_RADIUS_MIN_SCORE)
                    if small_ok and score >= MIN_FALLBACK_SCORE:
                        candidates.append({
                            "cx": cx, "cy": cy, "r": r,
                            "bbox": (int(cx - r), int(cy - r), int(2 * r), int(2 * r)),
                            "area": np.pi * r * r,
                            "score": score,
                        })
            continue
        if solidity < MIN_SOLIDITY:
            continue
        if not (0.55 <= aspect <= 1.8):
            continue

        (cx, cy), r = cv2.minEnclosingCircle(hull)
        if r < MIN_RADIUS_PX:
            continue

        score = hull_circularity * solidity
        if r < SMALL_RADIUS_PX and score < SMALL_RADIUS_MIN_SCORE:
            continue

        candidates.append({
            "cx": cx, "cy": cy, "r": r,
            "bbox": (x, y, w, h),
            "area": area,
            "score": score,
        })
    return candidates


def _fit_inscribed_circle(mask, cnt, blob_area):
    """Find the largest circle that fits entirely inside a single contour,
    via the distance transform. Returns (cx, cy, r) or None if the circle
    doesn't plausibly account for most of the blob (i.e. it's just a small
    round pocket inside an otherwise irregular shape, not a fused ball)."""
    blob_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(blob_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    _, r, _, (cx, cy) = cv2.minMaxLoc(dist)

    if r < MIN_RADIUS_PX:
        return None

    circle_area = np.pi * r * r
    if circle_area / blob_area < MIN_INSCRIBED_AREA_RATIO:
        return None

    return float(cx), float(cy), float(r)


def _remove_overlaps(candidates, iou_thresh=0.3):
    """Non-max suppression across all colour candidates so the same ball
    isn't reported twice."""
    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept = []
    for c in candidates:
        cx, cy, r = c["cx"], c["cy"], c["r"]
        overlap = False
        for k in kept:
            dist = np.hypot(cx - k["cx"], cy - k["cy"])
            if dist < 0.6 * (r + k["r"]):
                overlap = True
                break
        if not overlap:
            kept.append(c)
    return kept


def detect_balls(image):
    """Detect red and blue balls in a BGR image.

    Returns a list of dicts: {class_id, cx, cy, r, bbox}
    (cx, cy, r and bbox are in pixel coordinates)
    """
    h, w = image.shape[:2]
    img_area = float(h * w)

    # Mild blur to suppress JPEG noise/texture before colour thresholding.
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    all_candidates = []
    for color_name, cfg in COLOR_RANGES.items():
        mask = _build_color_mask(hsv, cfg["ranges"])
        mask = _clean_mask(mask)
        for c in _candidate_circles(mask, img_area):
            c["class_id"] = cfg["class_id"]
            c["color"] = color_name
            all_candidates.append(c)

    return _remove_overlaps(all_candidates)


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def detections_to_yolo_lines(detections, img_w, img_h):
    lines = []
    for d in detections:
        x, y, w, h = d["bbox"]
        x_center = (x + w / 2.0) / img_w
        y_center = (y + h / 2.0) / img_h
        norm_w = w / float(img_w)
        norm_h = h / float(img_h)
        lines.append(
            f"{d['class_id']} {x_center:.6f} {y_center:.6f} "
            f"{norm_w:.6f} {norm_h:.6f}"
        )
    return lines


def draw_debug(image, detections):
    out = image.copy()
    for d in detections:
        color = (255, 0, 0) if d["class_id"] == CLASS_BLUE else (0, 0, 255)
        cx, cy, r = int(d["cx"]), int(d["cy"]), int(d["r"])
        cv2.circle(out, (cx, cy), r, color, 3)
        label = "Blue" if d["class_id"] == CLASS_BLUE else "Red"
        cv2.putText(out, label, (cx - r, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return out


def process_folder(input_dir, output_dir, visualize=False, debug_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    if visualize and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"  [skip] could not read {path}")
            continue

        h, w = image.shape[:2]
        detections = detect_balls(image)
        lines = detections_to_yolo_lines(detections, w, h)

        stem = os.path.splitext(os.path.basename(path))[0]
        label_path = os.path.join(output_dir, stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        print(f"{os.path.basename(path):20s} -> {len(detections)} ball(s) "
              f"detected -> {os.path.basename(label_path)}")

        if visualize:
            debug_img = draw_debug(image, detections)
            out_path = os.path.join(debug_dir or output_dir,
                                     stem + "_debug.jpg")
            cv2.imwrite(out_path, debug_img)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect red/blue balls in images (classical CV, OpenCV)."
    )
    parser.add_argument("--input", required=True,
                         help="Folder containing input images.")
    parser.add_argument("--output", required=True,
                         help="Folder to write the YOLO-format .txt labels.")
    parser.add_argument("--visualize", action="store_true",
                         help="Also save debug images with drawn circles.")
    parser.add_argument("--debug-dir", default=None,
                         help="Folder for debug images (defaults to --output).")
    args = parser.parse_args()

    process_folder(args.input, args.output,
                    visualize=args.visualize, debug_dir=args.debug_dir)


if __name__ == "__main__":
    main()
