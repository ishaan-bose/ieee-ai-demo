# Rasterizer spec (stroke JSON -> 28x28 image)

One definition, two implementations that must agree: `backend/app/data/rasterizer.py` (Python/numpy)
and `frontend/src/lib/rasterizer.ts` (TypeScript). Golden fixtures: `shared/golden/rasterizer.json`.
Both must reproduce them with **max absolute pixel difference <= 1** (Python reproduces its own
fixture exactly; the tolerance covers float rounding differences in JS).

## Input

`strokes`: a list of strokes, each stroke `[xs, ys]` with equal-length number lists (SPEC 4.1 format).
Coordinates may be integers (Quick, Draw! "simplified", roughly 0..255) or arbitrary floats (a browser
canvas, e.g. 0..600). The input scale never matters, because of step 1.

## Algorithm

1. **Normalize** (preserving aspect ratio; NOT centered, to match the top-left alignment of the data):
   - `minx, miny` = minimum over all points of all strokes; `maxx, maxy` likewise.
   - `extent = max(maxx - minx, maxy - miny)`; if `extent == 0` (a single dot) use `extent = 1`.
   - `x' = (x - minx) * 255 / extent`, `y' = (y - miny) * 255 / extent`. Now the larger side is exactly 255
     and the drawing touches the top and left edges.
2. **Map into the 28x28 canvas** with a 2 px margin: `u = 2 + x' * (28 - 2*2) / 255`, same for `v` from `y'`.
   So the drawing occupies `[2, 26]` in continuous canvas coordinates; pixel `(row r, col c)` has its center at `(c + 0.5, r + 0.5)`.
3. **Render** each stroke as a polyline of segments between consecutive points (a stroke with one point is a
   zero-length segment, i.e. a dot). Line width `W = 2.0` px with round caps:
   `coverage(pixel, segment) = clamp(W/2 + 0.5 - dist(pixel center, segment), 0, 1)`
   (distance to the closest point of the segment). This is a linear anti-aliasing ramp: full ink within 0.5 px of
   the line, fading to zero at 1.5 px.
4. **Combine** segments with `max` (union), then `pixel = floor(255 * coverage + 0.5)` as `uint8`.

Output: `uint8[28][28]`, row-major (row 0 = top). The model input is `pixel / 255` as float, flattened to 784.

## Why these choices

- No timing data exists, so only geometry matters.
- Normalizing in the rasterizer (not trusting the data) matters because the stored coordinates are only
  approximately normalized (see NOTES.md): some drawings span less than 255.
- An analytic distance-based renderer (no library) is trivially identical in Python and TypeScript.

## Constants

`SIZE = 28`, `MARGIN = 2`, `LINE_WIDTH = 2.0`, `NORM = 255`.
