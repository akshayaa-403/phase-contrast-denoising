/**
 * The image viewer: one canvas showing two stages split by a draggable
 * divider, or every stage in a grid, with zoom and pan kept in sync.
 *
 * Rendering rules that matter for a scientific image:
 *   - magnifying uses nearest-neighbour and whole device-pixel steps, so every
 *     image pixel is a square of identical size and nothing is invented;
 *   - minifying averages (canvas smoothing), so a large image does not alias.
 *
 * The viewer knows nothing about the pipeline. A `provider` hands it a canvas
 * per stage id, overlay paths and labels; callbacks report what the visitor
 * did. All coordinates below are CSS pixels unless named `dev`.
 */

const MINIFY_LEVELS = [0.125, 0.25, 0.5];
const DEVICE_LEVELS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64];

export class Viewer {
  constructor({ root, canvas, divider, scanHandle, tagLayer, provider, on = {} }) {
    this.root = root;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.divider = divider;
    this.scanHandle = scanHandle;
    this.tagLayer = tagLayer;
    this.provider = provider;
    this.on = on;

    this.mode = 'split';
    this.left = 'input';
    this.right = 'classical';
    this.gridStages = [];
    this.split = 0.5;
    this.zoom = null; // null = fit
    this.cx = 0;
    this.cy = 0;
    this.hold = false;
    this.scanRow = null; // image row of the line profile, or null when off
    this.scaleGuide = null; // { sigma1, sigma2 } while a σ slider is dragged
    this.crosshair = false; // keyboard read-out at the view centre
    this.pointer = null; // last hovered screen point
    this.imageSize = null;
    this.colors = {};
    this.pending = 0;

    this.pointers = new Map();
    this.gesture = null;
    this.wheelAccumulator = 0;

    this.readColors();
    this.bind();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(root);
    this.resize();
  }

  // ------------------------------------------------------------ public API
  /** Called when a new image arrives: reset the view to fit, centred. */
  setImage(width, height) {
    const changedSize = !this.imageSize || this.imageSize.width !== width || this.imageSize.height !== height;
    this.imageSize = { width, height };
    if (changedSize) {
      this.zoom = null;
      this.cx = width / 2;
      this.cy = height / 2;
      if (this.scanRow !== null) this.scanRow = Math.floor(height / 2);
    }
    this.updateScanHandle();
    this.refreshTags();
    this.request();
    this.on.viewChange?.();
  }

  setStages(left, right) {
    this.left = left;
    this.right = right;
    this.refreshTags();
    this.updateDividerAria();
    this.request();
  }

  setGridStages(ids) {
    this.gridStages = ids;
    if (this.mode === 'grid') this.refreshTags();
    this.request();
  }

  setMode(mode) {
    this.mode = mode;
    this.divider.hidden = mode !== 'split';
    this.updateScanHandle();
    this.refreshTags();
    this.request();
    this.on.viewChange?.();
  }

  setSplit(fraction, notify = true) {
    this.split = Math.min(1, Math.max(0, fraction));
    this.positionDivider();
    this.updateDividerAria();
    this.request();
    if (notify) this.on.splitChange?.(this.split);
  }

  setHold(on) {
    if (this.hold === on) return;
    this.hold = on;
    this.refreshTags();
    this.request();
    this.on.holdChange?.(on);
  }

  setScanRow(row) {
    this.scanRow = row === null || !this.imageSize ? null : Math.min(this.imageSize.height - 1, Math.max(0, Math.round(row)));
    this.updateScanHandle();
    this.request();
    this.on.scanChange?.(this.scanRow);
  }

  setScaleGuide(guide) {
    this.scaleGuide = guide;
    this.request();
  }

  /** Current magnification, CSS px per image px. */
  get scale() {
    return this.zoom ?? this.fitScale();
  }

  fit() {
    this.zoom = null;
    if (this.imageSize) {
      this.cx = this.imageSize.width / 2;
      this.cy = this.imageSize.height / 2;
    }
    this.afterZoom();
  }

  /** One image pixel per CSS pixel, rounded to whole device pixels. */
  actualSize() {
    this.zoomTo(Math.max(1, Math.round(this.dpr)) / this.dpr);
  }

  zoomStep(direction, anchor = null) {
    const levels = this.levels();
    const current = this.scale;
    let next;
    if (direction > 0) next = levels.find((l) => l > current + 1e-6) ?? levels[levels.length - 1];
    else next = [...levels].reverse().find((l) => l < current - 1e-6) ?? levels[0];
    this.zoomTo(next, anchor);
  }

  /** Zoom to `scale`, keeping the image point under `anchor` fixed. */
  zoomTo(scale, anchor = null) {
    if (!this.imageSize) return;
    const before = this.scale;
    const s = this.snap(scale);
    if (anchor) {
      const region = this.regionAt(anchor.x, anchor.y) || null;
      const p = this.toImage(anchor.x, anchor.y, region);
      this.zoom = s;
      // Keep p under the anchor: anchor = origin + (p - c) * s.
      const origin = this.regionOrigin(region);
      this.cx = p.x - (anchor.x - origin.x) / s;
      this.cy = p.y - (anchor.y - origin.y) / s;
    } else {
      this.zoom = s;
    }
    if (Math.abs(before - s) > 1e-9 || anchor) this.clampCentre();
    this.afterZoom();
  }

  panBy(dx, dy) {
    const s = this.scale;
    this.cx -= dx / s;
    this.cy -= dy / s;
    this.clampCentre();
    this.request();
    this.on.viewChange?.();
  }

  /** Re-read colour tokens (overlay colours are fixed, but read once). */
  readColors() {
    const css = getComputedStyle(document.documentElement);
    for (const key of ['input', 'dog', 'classical', 'hybrid', 'truth', 'pin']) {
      this.colors[key] = css.getPropertyValue(`--v-${key}`).trim() || '#fff';
    }
    this.colors.bg = css.getPropertyValue('--viewer-bg').trim() || '#1a1d21';
    this.colors.text = css.getPropertyValue('--viewer-text').trim() || '#eef2f6';
    this.colors.line = css.getPropertyValue('--viewer-line').trim() || '#3a4048';
  }

  request() {
    if (this.pending) return;
    this.pending = requestAnimationFrame(() => {
      this.pending = 0;
      this.render();
    });
  }

  // ------------------------------------------------------------- geometry
  get dpr() {
    return window.devicePixelRatio || 1;
  }

  levels() {
    const dpr = this.dpr;
    const magnify = DEVICE_LEVELS.map((d) => d / dpr).filter((s) => s >= 0.5);
    const minify = MINIFY_LEVELS.filter((s) => s < magnify[0] - 1e-6);
    return [...minify, ...magnify];
  }

  /** Whole device pixels when magnifying; anything when minifying. */
  snap(scale) {
    const dev = scale * this.dpr;
    if (dev >= 1) return Math.max(1, Math.round(dev)) / this.dpr;
    return Math.max(0.05, scale);
  }

  /** The largest scale that fits, snapped down to whole device pixels. */
  fitScale() {
    if (!this.imageSize) return 1;
    const { width, height } = this.imageSize;
    const area = this.mode === 'grid' ? this.gridLayout().cell : { w: this.width, h: this.height };
    const margin = 8;
    const raw = Math.min((area.w - margin) / width, (area.h - margin) / height);
    const dev = raw * this.dpr;
    if (dev >= 1) return Math.floor(dev) / this.dpr;
    return Math.max(0.05, raw);
  }

  gridLayout() {
    const n = Math.max(1, this.gridStages.length);
    const gap = 8;
    const { width: iw = 1, height: ih = 1 } = this.imageSize || {};
    let best = null;
    for (let cols = 1; cols <= n; cols++) {
      const rows = Math.ceil(n / cols);
      const w = (this.width - gap * (cols + 1)) / cols;
      const h = (this.height - gap * (rows + 1)) / rows;
      const s = Math.min(w / iw, h / ih);
      if (!best || s > best.s) best = { cols, rows, w, h, s };
    }
    const cells = [];
    for (let i = 0; i < n; i++) {
      const c = i % best.cols;
      const r = Math.floor(i / best.cols);
      cells.push({ x: gap + c * (best.w + gap), y: gap + r * (best.h + gap), w: best.w, h: best.h });
    }
    return { cells, cell: { w: best.w, h: best.h } };
  }

  /** The region (whole frame or one grid cell) under a screen point. */
  regionAt(x, y) {
    if (this.mode !== 'grid') return { x: 0, y: 0, w: this.width, h: this.height, index: -1 };
    const { cells } = this.gridLayout();
    const i = cells.findIndex((c) => x >= c.x && x <= c.x + c.w && y >= c.y && y <= c.y + c.h);
    return i >= 0 ? { ...cells[i], index: i } : null;
  }

  regionOrigin(region) {
    const r = region || { x: 0, y: 0, w: this.width, h: this.height };
    return { x: r.x + r.w / 2, y: r.y + r.h / 2 };
  }

  /** Screen offset of image pixel (0, 0) within a region, device-aligned. */
  imageOffset(region) {
    const s = this.scale;
    const o = this.regionOrigin(region);
    const dpr = this.dpr;
    return {
      x: Math.round((o.x - this.cx * s) * dpr) / dpr,
      y: Math.round((o.y - this.cy * s) * dpr) / dpr,
    };
  }

  toImage(x, y, region) {
    const off = this.imageOffset(region);
    const s = this.scale;
    return { x: (x - off.x) / s, y: (y - off.y) / s };
  }

  clampCentre() {
    if (!this.imageSize) return;
    this.cx = Math.min(this.imageSize.width, Math.max(0, this.cx));
    this.cy = Math.min(this.imageSize.height, Math.max(0, this.cy));
  }

  afterZoom() {
    this.updateScanHandle();
    this.request();
    this.on.viewChange?.();
  }

  resize() {
    const rect = this.root.getBoundingClientRect();
    this.width = Math.max(1, rect.width);
    this.height = Math.max(1, rect.height);
    const dpr = this.dpr;
    this.canvas.width = Math.round(this.width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.positionDivider();
    this.updateScanHandle();
    this.refreshTags();
    this.render();
    this.on.viewChange?.();
  }

  // -------------------------------------------------------------- drawing
  render() {
    const { ctx } = this;
    const dpr = this.dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = this.colors.bg;
    ctx.fillRect(0, 0, this.width, this.height);
    if (!this.imageSize) return;

    if (this.mode === 'grid' && !this.hold) {
      const { cells } = this.gridLayout();
      this.gridStages.forEach((id, i) => {
        const cell = cells[i];
        this.drawStage(id, cell, cell);
        this.drawOverlays(cell, cell, id, true);
        ctx.strokeStyle = this.colors.line;
        ctx.lineWidth = 1;
        ctx.strokeRect(cell.x + 0.5, cell.y + 0.5, cell.w - 1, cell.h - 1);
      });
    } else {
      const frame = { x: 0, y: 0, w: this.width, h: this.height };
      if (this.hold) {
        this.drawStage('input', frame, frame);
      } else {
        const split = Math.round(this.width * this.split);
        this.drawStage(this.left, frame, { x: 0, y: 0, w: split, h: this.height });
        this.drawStage(this.right, frame, { x: split, y: 0, w: this.width - split, h: this.height });
      }
      this.drawOverlays(frame, frame, this.right, false);
    }
  }

  drawStage(id, region, clip) {
    const { ctx } = this;
    const source = this.provider.canvasFor(id);
    if (clip.w <= 0) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(clip.x, clip.y, clip.w, clip.h);
    ctx.clip();
    if (source) {
      const s = this.scale;
      const off = this.imageOffset(region);
      const { width, height } = this.imageSize;
      ctx.imageSmoothingEnabled = s * this.dpr < 1 || source.width !== width;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(source, 0, 0, source.width, source.height, off.x, off.y, width * s, height * s);
    } else {
      ctx.fillStyle = this.colors.text;
      ctx.font = '12px "IBM Plex Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('not available for this image', clip.x + clip.w / 2, clip.y + clip.h / 2);
    }
    ctx.restore();
  }

  drawOverlays(region, clip, stageId, inGrid) {
    const { ctx } = this;
    const s = this.scale;
    const off = this.imageOffset(region);
    const dpr = this.dpr;
    ctx.save();
    ctx.beginPath();
    ctx.rect(clip.x, clip.y, clip.w, clip.h);
    ctx.clip();

    const outlines = this.provider.outlines(inGrid ? null : stageId, inGrid);
    for (const o of outlines) {
      // Paths are in image coordinates; scale them, but keep line widths and
      // dash lengths constant on screen.
      ctx.setTransform(dpr * s, 0, 0, dpr * s, off.x * dpr, off.y * dpr);
      ctx.lineJoin = 'round';
      ctx.setLineDash(o.dashed ? [6 / s, 4 / s] : []);
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.75)';
      ctx.lineWidth = 4 / s;
      ctx.stroke(o.path);
      ctx.strokeStyle = this.colors[o.color] || this.colors.text;
      ctx.lineWidth = 2 / s;
      ctx.stroke(o.path);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.setLineDash([]);

    // Line-profile row.
    if (this.scanRow !== null) {
      const y = off.y + (this.scanRow + 0.5) * s;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(clip.x, y);
      ctx.lineTo(clip.x + clip.w, y);
      ctx.stroke();
      ctx.strokeStyle = this.colors.truth;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // σ scale guide while a σ slider is dragged: circles of radius σ₁ and σ₂
    // at the centre of the view, to compare with the cells and the halo.
    if (this.scaleGuide && !inGrid) {
      const o = this.regionOrigin(region);
      ctx.font = '12px "IBM Plex Mono", monospace';
      ctx.textAlign = 'left';
      for (const [r, label, color] of [
        [this.scaleGuide.sigma1, 'σ₁', this.colors.classical],
        [this.scaleGuide.sigma2, 'σ₂', this.colors.truth],
      ]) {
        const radius = r * s;
        ctx.beginPath();
        ctx.arc(o.x, o.y, radius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.lineWidth = 4;
        ctx.stroke();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = 'rgba(16, 18, 21, 0.85)';
        ctx.fillRect(o.x + radius * 0.71 + 2, o.y - radius * 0.71 - 16, 28, 16);
        ctx.fillStyle = color;
        ctx.fillText(label, o.x + radius * 0.71 + 6, o.y - radius * 0.71 - 4);
      }
    }

    // Keyboard crosshair: the read-out reports the pixel under it.
    if (this.crosshair && !inGrid) {
      const o = this.regionOrigin(region);
      ctx.strokeStyle = this.colors.text;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(o.x - 10, o.y + 0.5);
      ctx.lineTo(o.x - 3, o.y + 0.5);
      ctx.moveTo(o.x + 3, o.y + 0.5);
      ctx.lineTo(o.x + 10, o.y + 0.5);
      ctx.moveTo(o.x + 0.5, o.y - 10);
      ctx.lineTo(o.x + 0.5, o.y - 3);
      ctx.moveTo(o.x + 0.5, o.y + 3);
      ctx.lineTo(o.x + 0.5, o.y + 10);
      ctx.stroke();
    }
    ctx.restore();
  }

  // ------------------------------------------------------------- labels
  refreshTags() {
    const layer = this.tagLayer;
    layer.replaceChildren();
    if (!this.imageSize) return;
    const make = (id, style) => {
      const info = this.provider.info(id);
      const tag = document.createElement('div');
      tag.className = 'tag';
      tag.style.setProperty('--stage-color', `var(--v-${info.color})`);
      Object.assign(tag.style, style);
      const name = document.createElement('span');
      name.textContent = info.label;
      const sub = document.createElement('span');
      sub.className = 'sub';
      sub.textContent = ` · ${info.source}`;
      tag.append(name, sub);
      layer.append(tag);
    };
    if (this.hold) {
      make('input', { left: '8px', top: '8px' });
    } else if (this.mode === 'grid') {
      const { cells } = this.gridLayout();
      this.gridStages.forEach((id, i) => make(id, {
        left: `${cells[i].x + 4}px`, top: `${cells[i].y + 4}px`, maxWidth: `${Math.max(40, cells[i].w - 8)}px`,
      }));
    } else {
      make(this.left, { left: '8px', top: '8px' });
      make(this.right, { right: '8px', top: '8px' });
    }
  }

  positionDivider() {
    this.divider.style.left = `${this.split * 100}%`;
  }

  updateDividerAria() {
    const pct = Math.round(this.split * 100);
    this.divider.setAttribute('aria-valuenow', String(pct));
    const left = this.provider.info(this.left).label;
    const right = this.provider.info(this.right).label;
    this.divider.setAttribute('aria-valuetext', `${pct}%: ${left} on the left, ${right} on the right`);
  }

  updateScanHandle() {
    const h = this.scanHandle;
    if (this.scanRow === null || !this.imageSize || this.mode === 'grid') {
      h.hidden = true;
      return;
    }
    h.hidden = false;
    const off = this.imageOffset(null);
    const y = off.y + (this.scanRow + 0.5) * this.scale;
    h.style.top = `${Math.min(this.height - 14, Math.max(14, y))}px`;
    h.setAttribute('aria-valuemax', String(this.imageSize.height - 1));
    h.setAttribute('aria-valuenow', String(this.scanRow));
    h.setAttribute('aria-valuetext', `row ${this.scanRow} of ${this.imageSize.height}`);
  }

  // -------------------------------------------------------------- events
  bind() {
    const root = this.root;

    root.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    root.addEventListener('pointermove', (e) => this.onPointerMove(e));
    root.addEventListener('pointerup', (e) => this.onPointerUp(e));
    root.addEventListener('pointercancel', (e) => this.onPointerUp(e));
    root.addEventListener('pointerleave', (e) => {
      if (e.pointerType === 'mouse' && !this.pointers.size) {
        this.pointer = null;
        this.emitReadout();
      }
    });
    root.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    root.addEventListener('dblclick', (e) => {
      if (e.target !== this.canvas) return;
      const anchor = this.local(e);
      if (this.zoom === null) this.zoomTo(this.fitScale() * 2, anchor);
      else this.fit();
    });
    root.addEventListener('keydown', (e) => this.onKey(e));
    root.addEventListener('keyup', (e) => {
      if (e.key === '\\') this.setHold(false);
    });
    root.addEventListener('focus', () => {
      if (root.matches(':focus-visible')) {
        this.crosshair = true;
        this.emitReadout();
        this.request();
      }
    });
    root.addEventListener('blur', () => {
      this.crosshair = false;
      this.setHold(false);
      this.emitReadout();
      this.request();
    });

    // Divider: drag, or keys when focused.
    this.divider.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
      this.divider.setPointerCapture(e.pointerId);
      this.dividerDrag = true;
    });
    this.divider.addEventListener('pointermove', (e) => {
      if (!this.dividerDrag) return;
      this.setSplit(this.local(e).x / this.width);
    });
    const endDivider = () => {
      if (this.dividerDrag) this.on.splitCommit?.();
      this.dividerDrag = false;
    };
    this.divider.addEventListener('pointerup', endDivider);
    this.divider.addEventListener('pointercancel', endDivider);
    this.divider.addEventListener('keydown', (e) => {
      const step = e.shiftKey ? 0.1 : 0.01;
      let next = null;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') next = this.split - step;
      else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') next = this.split + step;
      else if (e.key === 'Home') next = 0;
      else if (e.key === 'End') next = 1;
      if (next === null) return;
      e.preventDefault();
      e.stopPropagation();
      this.setSplit(Math.round(next * 100) / 100);
      this.on.splitCommit?.();
    });

    // Line-profile handle.
    this.scanHandle.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
      this.scanHandle.setPointerCapture(e.pointerId);
      this.scanDrag = true;
    });
    this.scanHandle.addEventListener('pointermove', (e) => {
      if (!this.scanDrag) return;
      const p = this.toImage(0, this.local(e).y, null);
      this.setScanRow(Math.floor(p.y));
    });
    const endScan = () => {
      if (this.scanDrag) this.on.scanCommit?.();
      this.scanDrag = false;
    };
    this.scanHandle.addEventListener('pointerup', endScan);
    this.scanHandle.addEventListener('pointercancel', endScan);
    this.scanHandle.addEventListener('keydown', (e) => {
      if (this.scanRow === null) return;
      const step = e.shiftKey ? 10 : 1;
      let next = null;
      if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') next = this.scanRow - step;
      else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') next = this.scanRow + step;
      else if (e.key === 'Home') next = 0;
      else if (e.key === 'End') next = this.imageSize.height - 1;
      if (next === null) return;
      e.preventDefault();
      e.stopPropagation();
      this.setScanRow(next);
      this.on.scanCommit?.();
    });
  }

  local(e) {
    const rect = this.root.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  onPointerDown(e) {
    if (e.target !== this.canvas) return;
    if (e.button !== undefined && e.button !== 0 && e.pointerType === 'mouse') return;
    this.root.setPointerCapture(e.pointerId);
    this.pointers.set(e.pointerId, this.local(e));
    if (this.pointers.size === 1) {
      this.gesture = { type: 'pan', start: this.local(e), last: this.local(e), moved: 0 };
    } else if (this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      this.gesture = {
        type: 'pinch',
        startDist: Math.hypot(a.x - b.x, a.y - b.y) || 1,
        startScale: this.scale,
        mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
      };
    }
  }

  onPointerMove(e) {
    const p = this.local(e);
    if (e.pointerType === 'mouse' || !this.pointers.size) {
      this.pointer = p;
      this.crosshair = false;
    }
    if (!this.pointers.has(e.pointerId)) {
      this.emitReadout();
      return;
    }
    this.pointers.set(e.pointerId, p);
    const g = this.gesture;
    if (!g) return;
    if (g.type === 'pan' && this.pointers.size === 1) {
      const dx = p.x - g.last.x;
      const dy = p.y - g.last.y;
      g.moved += Math.abs(dx) + Math.abs(dy);
      g.last = p;
      if (g.moved > 3) {
        this.root.classList.add('dragging');
        this.panBy(dx, dy);
      }
    } else if (g.type === 'pinch' && this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y) || 1;
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const target = this.snap(g.startScale * (dist / g.startDist));
      if (Math.abs(target - this.scale) > 1e-9) this.zoomTo(target, mid);
      this.panBy(mid.x - g.mid.x, mid.y - g.mid.y);
      g.mid = mid;
    }
    this.emitReadout();
  }

  onPointerUp(e) {
    const g = this.gesture;
    const wasTap = g && g.type === 'pan' && g.moved <= 3 && this.pointers.size === 1;
    this.pointers.delete(e.pointerId);
    if (this.pointers.size === 0) {
      this.gesture = null;
      this.root.classList.remove('dragging');
      if (wasTap && this.mode === 'grid' && e.type === 'pointerup') {
        const region = this.regionAt(g.start.x, g.start.y);
        if (region && region.index >= 0) this.on.gridPick?.(this.gridStages[region.index]);
      }
    } else if (this.pointers.size === 1) {
      const [p] = [...this.pointers.values()];
      this.gesture = { type: 'pan', start: p, last: p, moved: 10 };
    }
  }

  onWheel(e) {
    if (!this.imageSize) return;
    e.preventDefault();
    // Mice send ~100 per notch; trackpads send many small deltas. Accumulate
    // so either steps one zoom level at a time.
    const unit = e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1;
    this.wheelAccumulator += e.deltaY * unit * (e.ctrlKey ? 3 : 1);
    const threshold = 60;
    const anchor = this.local(e);
    while (Math.abs(this.wheelAccumulator) >= threshold) {
      const direction = this.wheelAccumulator < 0 ? 1 : -1;
      this.wheelAccumulator -= -direction * threshold;
      this.zoomStep(direction, anchor);
    }
  }

  onKey(e) {
    if (e.target !== this.root) return;
    if (e.altKey || e.metaKey || e.ctrlKey) return;
    const pan = (e.shiftKey ? 128 : 32);
    let handled = true;
    switch (e.key) {
      case '+': case '=': this.zoomStep(1); break;
      case '-': case '_': this.zoomStep(-1); break;
      case '0': this.fit(); break;
      case '1': this.actualSize(); break;
      case 'ArrowLeft': this.panBy(pan, 0); break;
      case 'ArrowRight': this.panBy(-pan, 0); break;
      case 'ArrowUp': this.panBy(0, pan); break;
      case 'ArrowDown': this.panBy(0, -pan); break;
      case '\\': if (!e.repeat) this.setHold(true); break;
      default: handled = false;
    }
    if (handled) {
      e.preventDefault();
      this.crosshair = true;
      this.emitReadout();
    }
  }

  /** Report the image pixel under the pointer, or under the crosshair. */
  emitReadout() {
    if (!this.imageSize) return;
    let point = null;
    if (this.crosshair) {
      const o = this.regionOrigin(null);
      point = this.mode === 'grid' ? null : o;
    } else if (this.pointer) {
      point = this.pointer;
    }
    if (!point) {
      this.on.readout?.(null);
      return;
    }
    const region = this.regionAt(point.x, point.y);
    if (!region) {
      this.on.readout?.(null);
      return;
    }
    const p = this.toImage(point.x, point.y, region);
    const x = Math.floor(p.x);
    const y = Math.floor(p.y);
    if (x < 0 || y < 0 || x >= this.imageSize.width || y >= this.imageSize.height) {
      this.on.readout?.(null);
      return;
    }
    let stages;
    if (this.hold) stages = ['input'];
    else if (this.mode === 'grid') stages = this.gridStages;
    else stages = this.left === this.right ? [this.left] : [this.left, this.right];
    this.on.readout?.({ x, y, stages });
  }
}

/**
 * Trace the outline of a binary mask as closed loops along pixel edges, so a
 * dash pattern runs continuously around each object instead of restarting on
 * every pixel edge. Returns a Path2D in image coordinates.
 */
export function maskOutline(mask, width, height) {
  const W = width + 1;
  // Directed boundary edges, foreground on the right (y down). A corner can
  // start at most two edges (a saddle), so keep two slots.
  const next1 = new Int32Array(W * (height + 1)).fill(-1);
  const next2 = new Int32Array(W * (height + 1)).fill(-1);
  const add = (from, to) => {
    if (next1[from] < 0) next1[from] = to;
    else next2[from] = to;
  };
  const at = (x, y) => (x < 0 || y < 0 || x >= width || y >= height ? 0 : mask[y * width + x]);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask[y * width + x]) continue;
      if (!at(x, y - 1)) add(y * W + x, y * W + x + 1);
      if (!at(x + 1, y)) add(y * W + x + 1, (y + 1) * W + x + 1);
      if (!at(x, y + 1)) add((y + 1) * W + x + 1, (y + 1) * W + x);
      if (!at(x - 1, y)) add((y + 1) * W + x, y * W + x);
    }
  }
  const path = new Path2D();
  const take = (corner) => {
    let to = next1[corner];
    if (to >= 0) {
      next1[corner] = next2[corner];
      next2[corner] = -1;
    }
    return to;
  };
  for (let start = 0; start < next1.length; start++) {
    while (next1[start] >= 0) {
      let corner = start;
      path.moveTo(corner % W, Math.floor(corner / W));
      let dx = 0, dy = 0;
      let px = corner % W, py = Math.floor(corner / W);
      for (let guard = 0; guard < next1.length * 2; guard++) {
        const to = take(corner);
        if (to < 0) break;
        const tx = to % W, ty = Math.floor(to / W);
        const ndx = tx - (corner % W), ndy = ty - Math.floor(corner / W);
        if ((ndx !== dx || ndy !== dy) && (dx || dy)) path.lineTo(px, py);
        dx = ndx;
        dy = ndy;
        px = tx;
        py = ty;
        corner = to;
        if (corner === start) break;
      }
      path.lineTo(px, py);
      path.closePath();
    }
  }
  return path;
}
