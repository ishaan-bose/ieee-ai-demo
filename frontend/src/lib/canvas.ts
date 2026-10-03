/** Prepare a canvas for drawing at `w` x `h` CSS pixels and return its cleared 2D context.
 *  Assigning canvas.width/height reallocates the whole bitmap (and clears it) even when the value is unchanged, so only do it when the
 *  size really changed: redrawing every animation frame with the old "resize, then draw" pattern cost a bitmap allocation per frame. */
export function fitCanvas(c: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1;
  const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
  if (c.width !== pw || c.height !== ph) { c.width = pw; c.height = ph; }
  const ctx = c.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return ctx;
}
