# Task 1.2: Detect the Pattern — Documentation

## 1. How the red and blue balls are detected

The pipeline is pure classical computer vision (OpenCV, no deep learning):

1. The image is mildly Gaussian-blurred and converted from BGR to **HSV**.
   HSV separates color (hue) from lighting (value), which makes it far more
   robust to sun/shadow than thresholding in RGB.
2. Two HSV masks are built: one for **red** (two ranges, since red wraps
   around hue 0°/180°) and one for **blue**.
3. Each mask is cleaned with a median blur + morphological opening/closing
   to remove speckle noise and fill small gaps.
4. Contours are extracted from each mask and filtered by shape (see below).
5. Surviving red/blue candidates are merged and passed through a simple
   non-max-suppression step so overlapping detections from both masks don't
   double-count the same ball.

## 2. How lighting conditions and shadows are handled

* Thresholding in **HSV instead of RGB** means a ball keeps roughly the same
  hue whether it's in full sun or shade — only its V (brightness) channel
  shifts, and V is intentionally given a wide, permissive range in the
  thresholds.
* The **saturation (S) lower bound** is kept fairly high (80–90), which
  excludes pale, washed-out shadow regions and desaturated ground/wall
  colors that could otherwise be mistaken for a dull red or blue.
* A Gaussian blur before masking, plus a median blur + morphological
  open/close on the mask, remove the small speckled noise that hard shadows
  and sun glare tend to create on textured pavement.

## 3. How noise and false detections are reduced

Raw color masks alone produce lots of junk (clothing, tiles, wood grain,
wires, etc.), so every candidate blob must pass several shape checks before
being accepted as a ball:

* **Convex-hull circularity** — real soccer balls are covered in a
  contrasting pentagon/star pattern, and that pattern's dark patches cut
  deep notches into a raw contour. Measuring circularity on the *convex
  hull* of the blob smooths those notches back out, so a genuinely round
  ball still scores near-perfect roundness while irregular objects (limbs,
  furniture, wiring) do not.
* **Solidity** (raw area ÷ hull area) — makes sure the shape is actually
  filled in, not just something whose outer envelope happens to be round.
* **Aspect ratio** of the bounding box — rejects long/thin blobs; kept
  permissive enough to still accept balls that are partially cropped by the
  edge of the frame.
* **Minimum/maximum area** — rejects tiny colour speckles and (at the other
  end) implausibly huge regions such as an entire colored wall.
* **Stricter score for small blobs** — a small round-ish blob is far more
  likely to be coincidental noise than a small round-ish *large* blob is, so
  small candidates (under ~25 px radius) are held to a higher combined
  circularity × solidity score than large ones.
* **Fused-blob recovery, with its own stricter checks** — a ball photographed
  right next to a similarly-coloured object (e.g. a red ball in front of a
  red hoodie) can merge into one blob in the color mask and fail the normal
  circularity test. Rather than dropping such blobs outright, the code looks
  for the largest circle that fits entirely inside them (via the distance
  transform); this reliably lands on the ball itself, since it's the
  roundest, "fattest" part of the fused shape. To avoid this turning into a
  loophole for false positives, a fused-blob candidate must still pass all
  the normal filters *and* a higher minimum score than a normal detection.

## 4. How the location and size of each ball are determined

For every accepted contour, `cv2.minEnclosingCircle()` on its convex hull
gives a center point and radius, and `cv2.boundingRect()` gives a bounding
box. The bounding box is converted directly into the submission format.

## 5. How the final label files are generated

For each input image `X.jpg`, a file `X.txt` is written with one line per
detected ball:

```
<class_id> <x_center> <y_center> <width> <height>
```

* `class_id`: `0` = Blue, `1` = Red
* All four coordinates are normalized to the image's width/height (0–1),
  computed from each detection's bounding box.
* Images with zero detected balls get an empty `.txt` file (still one file
  per image, as required).

## Known limitations

Being a purely color/shape-based method (no semantic understanding), it has
predictable blind spots, observed directly on the provided sample images:

* **Same-colored background** — a blue ball photographed against a clear
  blue sky, or sitting inside a blue plastic bin, can merge with the
  background in the color mask and go undetected, since the merged blob no
  longer looks round *and* doesn't have a large-enough inscribed circle
  relative to the whole fused shape for the fallback recovery to trust it.
* **Scenes with many same-colored round objects** — e.g. a robot covered in
  red wiring, knobs, and buttons: some of those really are small, round, and
  red, so a pure color+shape method can mistake them for a ball. This is the
  main source of false positives observed on the sample set, concentrated in
  one particularly cluttered image.

These are inherent trade-offs of classical color segmentation; a
learning-based approach would be needed to resolve them with certainty, but
that is outside the scope of this task's constraints (OpenCV only, no deep
learning).
