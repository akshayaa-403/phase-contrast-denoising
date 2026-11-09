/**
 * Two small inline-SVG charts, no library.
 *
 *   drawProfile: intensity along one image row for several stages. A halo is
 *     an overshoot either side of each cell edge; suppression flattens it.
 *   drawDotPlot: one row per held-out frame, raw → U-Net, joined by a line.
 *
 * Stages are told apart by dash pattern and a direct label as well as colour,
 * so nothing depends on colour alone.
 */

const NS = 'http://www.w3.org/2000/svg';
const DASH = { input: '1.5 3', dog: '5 2 1.5 2', classical: '', hybrid: '7 3', truth: '10 3 2 3', pin: '3 2' };

function node(name, attrs = {}, parent = null) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (parent) parent.append(n);
  return n;
}

export function drawProfile(svg, series) {
  const width = Math.max(200, svg.clientWidth || svg.parentElement.clientWidth || 600);
  const height = Math.max(48, svg.clientHeight || 96);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.replaceChildren();
  const left = 28;
  const right = 108; // room for the direct labels
  const top = 4;
  const bottom = 12;
  const w = width - left - right;
  const h = height - top - bottom;

  for (const v of [0, 0.5, 1]) {
    const y = top + h * (1 - v);
    node('line', { x1: left, x2: left + w, y1: y, y2: y, class: 'axis' }, svg);
    const t = node('text', { x: left - 4, y: y + 3, 'text-anchor': 'end', class: 'axis-label' }, svg);
    t.textContent = v.toFixed(1);
  }

  const ends = [];
  for (const s of series) {
    const n = s.values.length;
    if (!n) continue;
    let d = '';
    for (let x = 0; x < n; x++) {
      const v = Math.min(1, Math.max(0, s.values[x]));
      const px = left + (n === 1 ? 0 : (x / (n - 1)) * w);
      const py = top + h * (1 - v);
      d += `${x ? 'L' : 'M'}${px.toFixed(1)},${py.toFixed(1)}`;
    }
    node('path', {
      d,
      class: 'series',
      stroke: `var(--v-${s.color})`,
      'stroke-dasharray': DASH[s.id] ?? DASH[s.color] ?? '',
    }, svg);
    ends.push({ s, y: top + h * (1 - Math.min(1, Math.max(0, s.values[n - 1]))) });
  }

  // Direct labels at the right end, nudged apart so they never overlap.
  ends.sort((a, b) => a.y - b.y);
  const gap = 11;
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + gap);
  const overflow = ends.length ? ends[ends.length - 1].y - (top + h) : 0;
  if (overflow > 0) for (const e of ends) e.y -= overflow;
  for (const e of ends) {
    node('line', {
      x1: left + w + 4, x2: left + w + 18, y1: e.y, y2: e.y,
      stroke: `var(--v-${e.s.color})`, 'stroke-width': 1.5,
      'stroke-dasharray': DASH[e.s.id] ?? DASH[e.s.color] ?? '',
    }, svg);
    const t = node('text', { x: left + w + 22, y: e.y + 3.5, class: 'series-label', fill: `var(--v-${e.s.color})` }, svg);
    t.textContent = e.s.label;
  }
}

/**
 * @param {SVGElement} svg
 * @param {object[]} samples manifest entries, in display order
 * @param {object} opts metric ('dice' | 'psnr'), digits, unit, current id, open(id), domain?
 */
export function drawDotPlot(svg, samples, { metric, digits, unit, current, open, domain, focusable = true }) {
  const rowH = 16;
  const top = 8;
  const bottom = 24;
  const left = 72;
  const right = 16;
  const width = Math.max(320, svg.clientWidth || svg.parentElement.clientWidth || 480);
  const height = top + bottom + rowH * samples.length;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('height', String(height));
  svg.replaceChildren();

  const values = samples.flatMap((s) => [s.metrics.raw[metric], s.metrics.hybrid[metric]]);
  let [lo, hi] = domain || [Math.min(...values), Math.max(...values)];
  if (!domain) {
    const pad = (hi - lo) * 0.08 || 1;
    lo = Math.floor(lo - pad);
    hi = Math.ceil(hi + pad);
  }
  const w = width - left - right;
  const x = (v) => left + ((v - lo) / (hi - lo)) * w;

  // "Nice" ticks (1, 2 or 5 × a power of ten), so labels are exact values.
  const rawStep = (hi - lo) / 5;
  const power = 10 ** Math.floor(Math.log10(rawStep));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= rawStep);
  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) {
    const px = x(v);
    node('line', { x1: px, x2: px, y1: top, y2: height - bottom + 4, class: 'grid-line' }, svg);
    const t = node('text', { x: px, y: height - 6, 'text-anchor': 'middle', class: 'axis-label' }, svg);
    t.textContent = v.toFixed(decimals);
  }

  samples.forEach((s, i) => {
    const raw = s.metrics.raw[metric];
    const hyb = s.metrics.hybrid[metric];
    const y = top + i * rowH + rowH / 2;
    const a = node('a', { href: `#sample=${s.id}`, class: s.id === current ? 'current' : '' }, svg);
    // A second chart of the same frames stays out of the tab order.
    if (!focusable) a.setAttribute('tabindex', '-1');
    a.setAttribute('aria-label',
      `${s.id}${s.id === current ? ' (open)' : ''}: raw ${raw.toFixed(digits)}${unit}, U-Net ${hyb.toFixed(digits)}${unit}`);
    a.addEventListener('click', (e) => {
      e.preventDefault();
      open(s.id);
    });
    node('rect', { x: 0, y: y - rowH / 2, width, height: rowH, class: 'hit', rx: 2 }, a);
    const label = node('text', { x: left - 8, y: y + 3.5, 'text-anchor': 'end', class: 'row-label' }, a);
    label.textContent = s.id.replace('img_', '#');
    node('line', { x1: x(raw), x2: x(hyb), y1: y, y2: y, class: 'link' }, a);
    if (s.id === current) node('circle', { cx: x(hyb), cy: y, r: 7, class: 'ring' }, a);
    node('circle', { cx: x(raw), cy: y, r: 4, class: 'raw' }, a);
    node('circle', { cx: x(hyb), cy: y, r: 4, class: 'hyb' }, a);
    const title = node('title', {}, a);
    title.textContent = a.getAttribute('aria-label');
  });
}
