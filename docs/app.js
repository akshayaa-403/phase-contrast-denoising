/**
 * The phase-contrast clean-up workbench.
 *
 * Loads the held-out samples that scripts/make_demo_assets.py publishes, runs
 * the classical pipeline (cv.js) in a Web Worker whenever a parameter moves,
 * and shows the result beside the precomputed U-Net output in a split viewer.
 * Quality is measured live against the ground truth for samples, and with
 * no-reference measures for the visitor's own images.
 *
 * Every number on the page is computed here or read from the manifest; none is
 * typed into the HTML.
 */
import { Viewer, maskOutline } from './viewer.js';
import { drawProfile, drawDotPlot } from './charts.js';

const { CONTROLS, BY_KEY } = window.PhaseParams;
const PhaseCV = window.PhaseCV;
const PhaseDecode = window.PhaseDecode;

const MANIFEST_URL = 'samples/manifest.json';
const README_CLAHE = 'https://github.com/akshayaa-403/phase-contrast-denoising#clahe-precision-and-the-clip-limit';
const LARGE_EDGE = 2048;
const MAX_PINS = 4;

// ------------------------------------------------------------------- stages
/**
 * Every stage the viewer can show. `live` stages are recomputed as the
 * sliders move; the M7 rule keeps one of them on screen while that happens.
 */
const STAGES = {
  input: { label: 'Input', color: 'input', kind: 'grey' },
  dog: { label: 'After DoG', color: 'dog', kind: 'grey', live: true },
  classical: { label: 'Classical', color: 'classical', kind: 'grey', live: true },
  hybrid: { label: 'Hybrid U-Net', color: 'hybrid', kind: 'grey', needs: ['hybrid'] },
  truth: { label: 'Ground truth', color: 'truth', kind: 'grey', needs: ['truth'] },
  removed: { label: 'Removed by DoG', color: 'dog', kind: 'diverging', live: true },
  errClassical: { label: 'Error · Classical', color: 'classical', kind: 'error', live: true, needs: ['truth'] },
  errHybrid: { label: 'Error · Hybrid', color: 'hybrid', kind: 'error', needs: ['truth', 'hybrid'] },
  residual: { label: 'U-Net residual', color: 'hybrid', kind: 'diverging', needs: ['residual'] },
};
const GRID_STAGES = ['input', 'dog', 'classical', 'hybrid', 'truth'];

// --------------------------------------------------------------------- state
const state = {
  manifest: null,
  defaults: null, // page params at config defaults (16-bit)
  params: null,
  image: null, // { kind, id, entry?, name, width, height, pixels, inputUnit, truth?, hybrid?, residual?, bitDepth, format }
  imageSeq: 0,
  live: null, // latest full or preview result from the worker
  liveVersion: 0,
  staticMetrics: null,
  pins: [],
  pinSeq: 0,
  view: { mode: 'split', left: 'input', right: 'hybrid', split: 0.5 },
  overlays: { mask: false, truth: false, profile: false },
  lut: 'grey',
  scanRow: null,
  sortBy: 'name',
};

const el = {};
const cache = new Map(); // stage canvases, masks and outlines keyed by version
let viewer = null;
let worker = null;
const jobs = { seq: 0, inFlight: null, queued: null, progressTimer: 0, announce: false, startedAt: 0 };

// ------------------------------------------------------------------ helpers
const $ = (id) => document.getElementById(id);
const fmt = (v, digits) => (Number.isFinite(v) ? v.toFixed(digits) : '—');

function announce(message) {
  // Cleared first, so the same sentence twice is still read.
  el.announcer.textContent = '';
  requestAnimationFrame(() => {
    el.announcer.textContent = message;
  });
}

let noteTimer = 0;
function viewerNote(message, ms = 4000) {
  el.viewerNote.textContent = message;
  el.viewerNote.hidden = !message;
  clearTimeout(noteTimer);
  if (message && ms) noteTimer = setTimeout(() => (el.viewerNote.hidden = true), ms);
}

function stageAvailable(id) {
  if (id.startsWith('pin-')) return state.pins.some((p) => p.id === id);
  const st = STAGES[id];
  if (!st || !state.image) return false;
  return (st.needs || []).every((key) => !!state.image[key]);
}

function stageIsLive(id) {
  return !!STAGES[id]?.live;
}

/** Label and provenance shown on the viewer tags and pickers. */
function stageInfo(id) {
  if (id.startsWith('pin-')) {
    const pin = state.pins.find((p) => p.id === id);
    return { label: pin ? pin.label : 'Pinned', source: 'frozen copy of a live result', color: 'pin' };
  }
  const st = STAGES[id] || STAGES.input;
  const image = state.image;
  const upload = image?.kind === 'upload';
  const preview = state.live?.preview && st.live ? ', preview' : '';
  const sources = {
    input: upload ? 'your image, no ground truth' : 'held-out input',
    dog: `live${preview}`,
    classical: `live${preview}`,
    hybrid: state.params?.bits === 8 ? 'precomputed · trained on 16-bit CLAHE output' : 'precomputed',
    truth: 'clean target',
    removed: `live, ±${fmt((state.live?.removedMax ?? 0) / 255, 3)}`,
    errClassical: 'live, |classical − truth|',
    errHybrid: '|hybrid − truth|',
    residual: `precomputed at config defaults, ±${fmt(state.manifest?.residual_scale, 2)}`,
  };
  return { label: st.label, source: sources[id] || '', color: st.color };
}

// ---------------------------------------------------------- colour tables
function lerpTable(stops) {
  const table = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k];
    const [t1, c1] = stops[k + 1];
    const f = (t - t0) / (t1 - t0 || 1);
    for (let c = 0; c < 3; c++) table[i * 3 + c] = Math.round(c0[c] + (c1[c] - c0[c]) * f);
  }
  return table;
}
const GREY = lerpTable([[0, [0, 0, 0]], [1, [255, 255, 255]]]);
// matplotlib's viridis, sampled at nine points: perceptually uniform.
const VIRIDIS = lerpTable([
  [0, [68, 1, 84]], [0.125, [71, 44, 122]], [0.25, [59, 81, 139]], [0.375, [44, 113, 142]],
  [0.5, [33, 144, 141]], [0.625, [39, 173, 129]], [0.75, [92, 200, 99]], [0.875, [170, 220, 50]], [1, [253, 231, 37]],
]);
// Diverging, mid-grey at zero: blue for negative, orange for positive.
const DIVERGING = lerpTable([[0, [33, 102, 172]], [0.5, [128, 128, 128]], [1, [224, 130, 20]]]);

// ------------------------------------------------------------- stage data
/**
 * The pixels behind a stage: { data, width, height, signed?, scale? }.
 * `signed` data runs -1..1 (diverging map); everything else 0..1.
 */
function stageData(id) {
  const image = state.image;
  if (!image || !stageAvailable(id)) return null;
  const live = state.live;
  const full = { width: image.width, height: image.height };
  if (id.startsWith('pin-')) {
    const pin = state.pins.find((p) => p.id === id);
    return { data: pin.data, ...full };
  }
  switch (id) {
    case 'input': return { data: image.inputUnit, ...full };
    case 'truth': return { data: image.truth, ...full };
    case 'hybrid': return { data: image.hybrid, ...full };
    case 'residual': return { data: image.residual, ...full, signed: true, scale: state.manifest.residual_scale };
    case 'errHybrid': {
      if (!image.errHybrid) image.errHybrid = absDiff(image.hybrid, image.truth);
      return { data: image.errHybrid, ...full };
    }
    default: break;
  }
  if (!live) return null;
  const dims = { width: live.width, height: live.height };
  switch (id) {
    case 'dog': return { data: live.dog, ...dims };
    case 'classical': return { data: live.result, ...dims };
    case 'removed': {
      const key = `removedUnit:${state.liveVersion}`;
      if (!cache.has(key)) {
        const m = live.removedMax || 1;
        const out = new Float32Array(live.removed.length);
        for (let i = 0; i < out.length; i++) out[i] = live.removed[i] / m;
        cache.set(key, out);
      }
      return { data: cache.get(key), ...dims, signed: true, scale: live.removedMax / 255 };
    }
    case 'errClassical': {
      if (live.preview) return null;
      const key = `errClassical:${state.liveVersion}`;
      if (!cache.has(key)) cache.set(key, absDiff(live.result, image.truth));
      return { data: cache.get(key), ...dims };
    }
    default: return null;
  }
}

function absDiff(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.abs(a[i] - b[i]);
  return out;
}

function stageVersion(id) {
  if (stageIsLive(id)) return `${state.imageSeq}.${state.liveVersion}`;
  return `${state.imageSeq}`;
}

/** A canvas holding the stage at its native resolution, through the colour map. */
function stageCanvas(id) {
  const key = `canvas:${id}:${stageVersion(id)}:${state.lut}`;
  if (cache.has(key)) return cache.get(key);
  const stage = stageData(id);
  if (!stage) return null;
  const { data, width, height, signed } = stage;
  const table = signed ? DIVERGING : state.lut === 'viridis' ? VIRIDIS : GREY;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(width, height);
  const px = img.data;
  for (let i = 0; i < data.length; i++) {
    let v = signed ? (data[i] + 1) / 2 : data[i];
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    const t = Math.round(v * 255) * 3;
    px[i * 4] = table[t];
    px[i * 4 + 1] = table[t + 1];
    px[i * 4 + 2] = table[t + 2];
    px[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  pruneCache(`canvas:${id}:`);
  cache.set(key, canvas);
  return canvas;
}

/** Drop older cached entries for one prefix, so memory stays flat. */
function pruneCache(prefix) {
  for (const key of cache.keys()) if (key.startsWith(prefix)) cache.delete(key);
}

/** Value at an image pixel, for the read-out and line profile. */
function stageValue(id, x, y) {
  const stage = stageData(id);
  if (!stage) return null;
  const sx = Math.min(stage.width - 1, Math.floor((x * stage.width) / state.image.width));
  const sy = Math.min(stage.height - 1, Math.floor((y * stage.height) / state.image.height));
  const v = stage.data[sy * stage.width + sx];
  return stage.signed ? v * stage.scale : v;
}

/** Otsu mask of a stage (grey stages only), cached per version. */
function stageMask(id) {
  if (STAGES[id] && STAGES[id].kind !== 'grey') return null;
  const key = `mask:${id}:${stageVersion(id)}`;
  if (cache.has(key)) return cache.get(key);
  let mask = null;
  if (id === 'classical' && state.live?.mask) mask = state.live.mask;
  else {
    const stage = stageData(id);
    if (!stage || stage.width !== state.image.width) return null;
    mask = PhaseCV.otsuMask(stage.data).mask;
  }
  pruneCache(`mask:${id}:`);
  cache.set(key, mask);
  return mask;
}

function outlinePath(id) {
  const key = `outline:${id}:${stageVersion(id)}`;
  if (cache.has(key)) return cache.get(key);
  const mask = stageMask(id);
  if (!mask) return null;
  const path = maskOutline(mask, state.image.width, state.image.height);
  pruneCache(`outline:${id}:`);
  cache.set(key, path);
  return path;
}

// ------------------------------------------------------------ provider
const provider = {
  canvasFor: (id) => stageCanvas(id),
  info: (id) => stageInfo(id),
  outlines(rightId, inGrid) {
    const list = [];
    if (!state.image) return list;
    if (state.overlays.mask && rightId && !inGrid) {
      const path = outlinePath(rightId);
      const color = rightId.startsWith('pin-') ? 'pin' : STAGES[rightId]?.color;
      if (path) list.push({ path, color, dashed: false });
    }
    if (state.overlays.truth && state.image.truth) {
      const path = outlinePath('truth');
      if (path) list.push({ path, color: 'truth', dashed: true });
    }
    return list;
  },
};

// ===================================================================== init
async function init() {
  for (const node of document.querySelectorAll('[id]')) el[node.id] = node;
  el.bitsRadios = [...document.querySelectorAll('input[name="bits"]')];

  let manifest = null;
  try {
    const response = await fetch(MANIFEST_URL, { cache: 'no-cache' });
    if (response.ok) manifest = await response.json();
  } catch {
    manifest = null;
  }
  if (!manifest || !manifest.samples?.length) {
    el.noSamples.hidden = false;
    return;
  }
  state.manifest = manifest;
  state.defaults = window.PhaseParams.fromManifest(manifest.defaults);
  state.params = { ...state.defaults };
  el.app.hidden = false;
  el.evidence.hidden = false;

  renderHeadline();
  buildControls();
  buildSamples();
  buildEvidence();
  initViewer();
  initWorker();
  initTabs();
  initInputs();
  initExport();
  initOverlays();

  // The view in the URL, if any; otherwise the worst raw frame.
  const fromHash = readHash();
  const worst = worstRawSample();
  const sampleId = fromHash.sample && manifest.samples.some((s) => s.id === fromHash.sample) ? fromHash.sample : worst.id;
  applyHashView(fromHash);
  syncControls();
  await openSample(sampleId, { keepView: true });
  window.addEventListener('hashchange', async () => {
    const h = readHash();
    applyHashView(h);
    syncControls();
    if (h.sample && h.sample !== state.image?.id) await openSample(h.sample, { keepView: true });
    else requestRun({ preview: false });
  });
}

function worstRawSample() {
  return [...state.manifest.samples].sort((a, b) => a.metrics.raw.dice - b.metrics.raw.dice)[0];
}

// ------------------------------------------------------------- headline
function renderHeadline() {
  const h = state.manifest.held_out;
  const raw = h?.paths?.raw;
  const hybrid = h?.paths?.hybrid;
  if (!raw || !hybrid) return;
  el.headline.replaceChildren();
  const tag = document.createElement('span');
  tag.className = 'hl-tag';
  tag.textContent = `Held-out frames (${h.frames} the U-Net never saw): `;
  const psnr = document.createElement('span');
  psnr.innerHTML = `PSNR ${fmt(raw.psnr.mean, 2)} → <strong>${fmt(hybrid.psnr.mean, 2)} dB</strong> with the U-Net`;
  const dice = document.createElement('span');
  dice.innerHTML = `; lowest Otsu Dice ${fmt(raw.dice.min, 3)} → <strong>${fmt(hybrid.dice.min, 3)}</strong>`;
  el.headline.append(tag, psnr, dice);
  el.headline.hidden = false;
}

// ------------------------------------------------------------- controls
function buildControls() {
  for (const c of CONTROLS) {
    const wrap = document.createElement('div');
    wrap.className = 'control';
    wrap.dataset.key = c.key;
    const id = `ctl-${c.key}`;
    const noteId = `${id}-note`;
    const def = state.defaults[c.key];
    const pct = (def - c.min) / (c.max - c.min);
    wrap.innerHTML = `
      <label class="name" for="${id}"><span class="sym">${c.symbol}</span> · ${c.name}, <span class="unit">${c.unit}</span></label>
      <div class="track">
        <input type="range" id="${id}" min="${c.min}" max="${c.max}" step="${c.step}" aria-describedby="${noteId}">
        <span class="default-mark" style="left: calc(8px + (100% - 16px) * ${pct})" title="Config default ${fmt(def, c.digits)}"></span>
      </div>
      <input type="number" class="number" id="${id}-num" min="${c.min}" max="${c.max}" step="${c.step}"
             aria-label="${c.symbol} ${c.name}, ${c.unit}, exact value">
      <button type="button" class="reset" id="${id}-reset" aria-label="Reset ${c.symbol} to ${fmt(def, c.digits)}" title="Reset to ${fmt(def, c.digits)}">↺</button>
      <p class="note" id="${noteId}">Default ${fmt(def, c.digits)}.</p>`;
    (c.group === 'dog' ? el.dogControls : el.claheControls).append(wrap);

    const range = wrap.querySelector('input[type="range"]');
    const number = wrap.querySelector('input[type="number"]');
    const reset = wrap.querySelector('.reset');
    c.el = { wrap, range, number, reset, note: wrap.querySelector('.note') };

    const set = (value, commit) => setParam(c.key, value, { commit, source: 'control' });
    range.addEventListener('input', () => set(parseFloat(range.value), false));
    range.addEventListener('change', () => set(parseFloat(range.value), true));
    number.addEventListener('change', () => {
      const v = parseFloat(number.value);
      if (Number.isFinite(v)) set(v, true);
      else syncControls();
    });
    // Shift+arrow moves ten steps on both the slider and the number box.
    for (const input of [range, number]) {
      input.addEventListener('keydown', (e) => {
        if (!e.shiftKey) return;
        const dir = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
        if (!dir) return;
        e.preventDefault();
        set(state.params[c.key] + dir * 10 * c.step, true);
      });
    }
    reset.addEventListener('click', () => set(state.defaults[c.key], true));

    // σ scale guide while a σ slider is in use.
    if (c.key === 'sigma1' || c.key === 'sigma2') {
      const show = () => viewer?.setScaleGuide({ sigma1: state.params.sigma1, sigma2: state.params.sigma2 });
      const hide = () => viewer?.setScaleGuide(null);
      range.addEventListener('pointerdown', show);
      range.addEventListener('pointerup', hide);
      range.addEventListener('pointercancel', hide);
      range.addEventListener('keydown', (e) => {
        if (e.key.startsWith('Arrow') || e.key === 'Home' || e.key === 'End' || e.key.startsWith('Page')) show();
      });
      range.addEventListener('keyup', () => setTimeout(hide, 600));
      range.addEventListener('blur', hide);
    }
  }

  for (const radio of el.bitsRadios) {
    radio.addEventListener('change', () => {
      if (radio.checked) setParam('bits', Number(radio.value), { commit: true, source: 'control' });
    });
  }
  el.resetAll.addEventListener('click', () => {
    state.params = { ...state.defaults };
    syncControls();
    paramsChanged({ commit: true, source: 'control' });
    announce('All parameters reset to the values in configs/default_config.yaml.');
  });
  el.pinBtn.addEventListener('click', pinCurrent);
}

/** Change one parameter, keeping σ₂ > σ₁. */
function setParam(key, value, { commit, source }) {
  const p = { ...state.params };
  let sigmaNote = '';
  if (key === 'bits') {
    p.bits = value === 8 ? 8 : 16;
  } else {
    const c = BY_KEY[key];
    let v = window.PhaseParams.snap(c, value);
    if (key === 'sigma2') v = Math.max(v, window.PhaseParams.sigma2Min(p.sigma1));
    p[key] = v;
    if (key === 'sigma1' && !(p.sigma2 > v)) {
      p.sigma2 = window.PhaseParams.sigma2Min(v);
      sigmaNote = `σ₂ raised to ${fmt(p.sigma2, 1)} to stay above σ₁.`;
    }
  }
  const changed = Object.keys(p).some((k) => p[k] !== state.params[k]);
  state.params = p;
  syncControls(sigmaNote);
  if (key === 'sigma1' || key === 'sigma2') {
    if (viewer?.scaleGuide) viewer.setScaleGuide({ sigma1: p.sigma1, sigma2: p.sigma2 });
  }
  if (changed || commit) paramsChanged({ commit, source });
}

let sigmaNoteText = '';
/** Reflect state.params in every control, note and the export panel. */
function syncControls(sigmaNote = null) {
  if (sigmaNote !== null) sigmaNoteText = sigmaNote;
  const p = state.params;
  for (const c of CONTROLS) {
    const { range, number, reset, note, wrap } = c.el;
    if (c.key === 'sigma2') {
      // The slider's minimum follows σ₁, so σ₂ ≤ σ₁ cannot be reached.
      const min = window.PhaseParams.sigma2Min(p.sigma1);
      range.min = String(min);
      number.min = String(min);
    }
    range.value = String(p[c.key]);
    if (document.activeElement !== number) number.value = fmt(p[c.key], c.digits);
    reset.disabled = Math.abs(p[c.key] - state.defaults[c.key]) < 1e-9;
    const def = `Default ${fmt(state.defaults[c.key], c.digits)}.`;
    if (c.key === 'sigma2') {
      note.textContent = sigmaNoteText ? `${sigmaNoteText} ${def}` : `Must stay above σ₁. ${def}`;
      note.classList.toggle('warn', !!sigmaNoteText);
    } else if (c.key === 'clipLimit') {
      updateClipNote(note, wrap, def);
    } else if (c.key === 'tiles') {
      note.textContent = `${p.tiles} × ${p.tiles} grid. ${def}`;
    } else {
      note.textContent = def;
    }
  }
  for (const radio of el.bitsRadios) radio.checked = Number(radio.value) === p.bits;
  updateExportText();
  refreshPickers();
  scheduleHash();
}

/**
 * The clip control's honesty note: the ceiling OpenCV derives from the clip
 * limit, live, and whether the control can change anything at this tile size,
 * image size and precision. Calculated, never hard-coded.
 */
function updateClipNote(note, wrap, def) {
  const image = state.image;
  const p = state.params;
  if (!image) {
    note.textContent = def;
    return;
  }
  const c = BY_KEY.clipLimit;
  const at = (clip) => PhaseCV.claheCeiling(clip, p.tiles, image.width, image.height, p.bits);
  const now = at(p.clipLimit);
  const lowest = at(c.min).ceiling;
  const highest = at(c.max).ceiling;
  const levels = p.bits === 8 ? '256' : '65,536';
  const tile = `${now.tileWidth} × ${now.tileHeight}`;
  note.replaceChildren();
  const line = document.createElement('span');
  line.textContent = `Ceiling: ${now.ceiling} px per grey level (${levels} levels, ${tile} px tiles). `;
  note.append(line);
  const inactive = lowest === highest;
  wrap.classList.toggle('inactive', inactive);
  note.classList.toggle('warn', inactive);
  if (inactive) {
    const why = document.createElement('span');
    why.textContent = `Inactive here: every clip value gives a ceiling of ${highest}, so the output cannot change. Try 8-bit, fewer tiles or a larger image. `;
    const link = document.createElement('a');
    link.href = README_CLAHE;
    link.textContent = 'Why?';
    note.append(why, link, document.createTextNode(' '));
  } else if (lowest === 1) {
    const firstActive = window.PhaseParams.sweep(c).find((v) => at(v).ceiling > 1);
    const partial = document.createElement('span');
    partial.textContent = `Takes effect from ${fmt(firstActive, 1)} upwards. `;
    note.append(partial);
  }
  const d = document.createElement('span');
  d.textContent = def;
  note.append(d);
}

/** After any parameter change: keep a live stage visible, recompute. */
function paramsChanged({ commit, source }) {
  if (source === 'control') ensureLiveVisible();
  requestRun({ preview: !commit });
  if (commit) jobs.announce = true;
}

/** M7: the image being changed must be on screen. */
function ensureLiveVisible() {
  const v = state.view;
  if (v.mode === 'grid') return;
  if (stageIsLive(v.left) || stageIsLive(v.right)) return;
  v.right = 'classical';
  applyView();
  const msg = 'Right side switched to Classical (live) so you can see the change.';
  viewerNote(msg);
  announce(msg);
}

// --------------------------------------------------------------- samples
function buildSamples() {
  initSampleKeys();
  el.sampleSort.addEventListener('change', () => {
    state.sortBy = el.sampleSort.value;
    renderSampleTiles();
  });
  renderSampleTiles();
}

function renderSampleTiles() {
  const worst = worstRawSample();
  const samples = [...state.manifest.samples];
  if (state.sortBy === 'dice') samples.sort((a, b) => a.metrics.raw.dice - b.metrics.raw.dice);
  el.sampleGrid.replaceChildren();
  for (const entry of samples) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'sample';
    button.dataset.id = entry.id;
    button.setAttribute('aria-pressed', String(state.image?.id === entry.id));
    const raw = entry.metrics.raw.dice;
    const hyb = entry.metrics.hybrid?.dice;
    const worstText = entry.id === worst.id ? ', worst raw frame' : '';
    button.setAttribute('aria-label',
      `${entry.id}${worstText}: Otsu Dice raw ${fmt(raw, 3)}${hyb !== undefined ? `, U-Net ${fmt(hyb, 3)}` : ''}`);
    const img = new Image(48, 48);
    img.src = `samples/${entry.thumb || entry.noisy}`;
    img.alt = '';
    img.decoding = 'async';
    const dice = document.createElement('span');
    dice.className = 'dice';
    dice.textContent = hyb !== undefined ? `${fmt(raw, 2)}→${fmt(hyb, 2)}` : fmt(raw, 2);
    button.append(img, dice);
    if (entry.id === worst.id) {
      const flag = document.createElement('span');
      flag.className = 'flag';
      flag.textContent = 'worst raw';
      button.append(flag);
    }
    button.addEventListener('click', () => openSample(entry.id));
    el.sampleGrid.append(button);
  }
  rovingFocus();
}

/** One tab stop for the whole strip; arrow keys move between frames. */
function rovingFocus() {
  const buttons = [...el.sampleGrid.querySelectorAll('.sample')];
  const current = buttons.find((b) => b.getAttribute('aria-pressed') === 'true') || buttons[0];
  for (const b of buttons) b.tabIndex = b === current ? 0 : -1;
}

function initSampleKeys() {
  el.sampleGrid.addEventListener('keydown', (e) => {
    const buttons = [...el.sampleGrid.querySelectorAll('.sample')];
    const i = buttons.indexOf(document.activeElement);
    if (i < 0) return;
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    let j = null;
    if (step) j = Math.min(buttons.length - 1, Math.max(0, i + step));
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = buttons.length - 1;
    if (j === null) return;
    e.preventDefault();
    for (const b of buttons) b.tabIndex = -1;
    buttons[j].tabIndex = 0;
    buttons[j].focus();
    buttons[j].scrollIntoView({ block: 'nearest', inline: 'nearest' });
  });
}

async function fetchPng(name) {
  const response = await fetch(`samples/${name}`);
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  return PhaseDecode.decodePng(new Uint8Array(await response.arrayBuffer()));
}

function unitFrom(decoded) {
  const out = new Float32Array(decoded.pixels.length);
  for (let i = 0; i < out.length; i++) out[i] = decoded.pixels[i] / 255;
  return out;
}

async function openSample(id, { keepView = false } = {}) {
  const entry = state.manifest.samples.find((s) => s.id === id);
  if (!entry) return;
  el.imageSource.textContent = `Loading ${id}…`;
  try {
    const [noisy, truth, hybrid, residual] = await Promise.all([
      fetchPng(entry.noisy),
      entry.truth ? fetchPng(entry.truth) : null,
      entry.hybrid ? fetchPng(entry.hybrid) : null,
      entry.residual ? fetchPng(entry.residual) : null,
    ]);
    let residualSigned = null;
    if (residual) {
      residualSigned = new Float32Array(residual.pixels.length);
      for (let i = 0; i < residualSigned.length; i++) residualSigned[i] = (residual.pixels[i] - 128) / 127;
    }
    setImage({
      kind: 'sample',
      id,
      entry,
      name: id,
      width: noisy.width,
      height: noisy.height,
      pixels: noisy.pixels,
      truth: truth ? unitFrom(truth) : null,
      hybrid: hybrid ? unitFrom(hybrid) : null,
      residual: residualSigned,
      bitDepth: 8,
      format: 'PNG',
    }, { keepView });
  } catch (err) {
    el.imageSource.textContent = `Could not load ${id}: ${err.message}`;
  }
}

function setImage(image, { keepView = false } = {}) {
  image.inputUnit = PhaseCV.normalize01(image.pixels);
  state.image = image;
  state.imageSeq++;
  state.live = null;
  state.staticMetrics = null;
  state.pins = [];
  renderPins();
  cache.clear();

  // Stages this image cannot show fall back to sensible ones.
  const v = state.view;
  if (!keepView || !stageAvailable(v.left)) v.left = 'input';
  if (!keepView || !stageAvailable(v.right)) v.right = image.hybrid ? 'hybrid' : 'classical';
  if (state.scanRow !== null) state.scanRow = Math.min(state.scanRow, image.height - 1);

  for (const b of el.sampleGrid.querySelectorAll('.sample')) {
    const on = b.dataset.id === image.id && image.kind === 'sample';
    b.setAttribute('aria-pressed', String(on));
    if (on) b.scrollIntoView({ block: 'nearest', inline: 'center' });
  }
  if (document.activeElement?.closest?.('#sampleGrid') == null) rovingFocus();
  const worst = worstRawSample();
  if (image.kind === 'sample') {
    el.imageSource.textContent = `${image.id} · held-out${image.id === worst.id ? ' · worst raw frame' : ''}`;
  } else {
    el.imageSource.textContent = `${image.name} · your image, no ground truth · ${image.bitDepth}-bit ${image.format}`;
  }
  el.measureRef.hidden = image.kind !== 'sample' || !image.truth;
  el.measureNoRef.hidden = !el.measureRef.hidden;
  el.explainNoRef.hidden = image.kind === 'sample';
  el.explainPsnr.hidden = image.kind !== 'sample';
  el.ovTruth.disabled = !image.truth;

  viewer.setImage(image.width, image.height);
  if (state.overlays.profile && state.scanRow === null) state.scanRow = Math.floor(image.height / 2);
  viewer.setScanRow(state.overlays.profile ? state.scanRow : null);
  applyView();
  syncControls();

  worker.postMessage({
    type: 'image',
    imageId: state.imageSeq,
    width: image.width,
    height: image.height,
    pixels: Float32Array.from(image.pixels),
    truth: image.truth ? Float32Array.from(image.truth) : null,
    hybrid: image.hybrid ? Float32Array.from(image.hybrid) : null,
  });
  jobs.announce = false;
  requestRun({ preview: false });
  renderMeasured();
  renderDots();
}

// --------------------------------------------------------------- uploads
function initInputs() {
  el.fileInput.addEventListener('change', () => {
    const [file] = el.fileInput.files;
    if (file) openFile(file);
    el.fileInput.value = '';
  });
  const viewerEl = el.viewer;
  let depth = 0;
  viewerEl.addEventListener('dragenter', (e) => {
    if (![...(e.dataTransfer?.types || [])].includes('Files')) return;
    e.preventDefault();
    depth++;
    el.dropHint.hidden = false;
  });
  viewerEl.addEventListener('dragover', (e) => {
    if ([...(e.dataTransfer?.types || [])].includes('Files')) e.preventDefault();
  });
  viewerEl.addEventListener('dragleave', () => {
    depth = Math.max(0, depth - 1);
    if (!depth) el.dropHint.hidden = true;
  });
  viewerEl.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    el.dropHint.hidden = true;
    const [file] = e.dataTransfer.files;
    if (file) openFile(file);
  });
  document.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.kind === 'file');
    if (!item) return;
    const file = item.getAsFile();
    if (file) {
      e.preventDefault();
      openFile(file);
    }
  });
  el.largeCancel.addEventListener('click', () => {
    el.largeDialog.hidden = true;
    viewerNote('');
  });
}

function uploadError(message) {
  el.uploadError.textContent = message;
  el.uploadError.hidden = !message;
  if (message) {
    showTab('image');
    viewerNote(message, 6000);
  }
}

async function openFile(file) {
  uploadError('');
  viewerNote(`Reading ${file.name}…`, 0);
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    let decoded = await PhaseDecode.decode(bytes);
    if (!decoded) decoded = await decodeWithBrowser(file);
    viewerNote('');
    const { width, height } = decoded;
    if (Math.max(width, height) > LARGE_EDGE) {
      offerLarge(file.name, decoded);
      return;
    }
    useDecoded(file.name, decoded);
  } catch (err) {
    viewerNote('');
    const kind = file.type || file.name.split('.').pop();
    uploadError(`Could not open ${file.name} (${kind}): ${err.message} Try PNG, TIFF, JPEG or WebP.`);
  }
}

async function decodeWithBrowser(file) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  } catch {
    throw new Error('this browser cannot decode it as an image.');
  }
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  const pixels = new Float32Array(bitmap.width * bitmap.height);
  for (let i = 0; i < pixels.length; i++) {
    // ITU luma weights keep a colour upload from shifting brightness.
    pixels[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }
  const format = (file.type.split('/')[1] || 'image').toUpperCase();
  return { width: bitmap.width, height: bitmap.height, pixels, bitDepth: 8, format };
}

function offerLarge(name, decoded) {
  const { width, height } = decoded;
  const scale = LARGE_EDGE / Math.max(width, height);
  const dw = Math.round(width * scale);
  const dh = Math.round(height * scale);
  const cw = Math.min(width, LARGE_EDGE);
  const ch = Math.min(height, LARGE_EDGE);
  el.largeTitle.textContent = `${name} is ${width} × ${height} px. Larger than ${LARGE_EDGE} px on a side is slow in a browser tab:`;
  el.largeDownscale.textContent = `Downscale to ${dw} × ${dh}`;
  el.largeCrop.textContent = `Use the centre ${cw} × ${ch}`;
  el.largeDialog.hidden = false;
  el.largeDownscale.onclick = () => {
    el.largeDialog.hidden = true;
    const pixels = PhaseCV.resizeArea(decoded.pixels, width, height, dw, dh);
    useDecoded(`${name} (downscaled)`, { ...decoded, width: dw, height: dh, pixels });
  };
  el.largeCrop.onclick = () => {
    el.largeDialog.hidden = true;
    const x0 = Math.floor((width - cw) / 2);
    const y0 = Math.floor((height - ch) / 2);
    const pixels = new Float32Array(cw * ch);
    for (let y = 0; y < ch; y++) pixels.set(decoded.pixels.subarray((y0 + y) * width + x0, (y0 + y) * width + x0 + cw), y * cw);
    useDecoded(`${name} (centre crop)`, { ...decoded, width: cw, height: ch, pixels });
  };
  el.largeDownscale.focus();
}

function useDecoded(name, decoded) {
  setImage({
    kind: 'upload',
    id: null,
    name,
    width: decoded.width,
    height: decoded.height,
    pixels: decoded.pixels,
    truth: null,
    hybrid: null,
    residual: null,
    bitDepth: decoded.bitDepth,
    format: decoded.format,
  });
  if (state.view.mode === 'split' && !stageIsLive(state.view.left) && !stageIsLive(state.view.right)) {
    state.view.right = 'classical';
    applyView();
  }
  announce(`Opened ${name}, ${decoded.width} by ${decoded.height} pixels. It has no ground truth, so no-reference measures are shown.`);
}

// ---------------------------------------------------------------- worker
function initWorker() {
  worker = new Worker('worker.js');
  worker.onmessage = (event) => {
    const m = event.data;
    if (m.type === 'static') {
      if (m.imageId !== state.imageSeq) return;
      state.staticMetrics = m;
      renderMeasured();
    } else if (m.type === 'progress') {
      if (jobs.inFlight?.job === m.job) el.progressBar.value = m.fraction;
    } else if (m.type === 'result') {
      onResult(m);
    }
  };
  worker.onerror = (e) => {
    viewerNote(`The pipeline failed: ${e.message}`, 0);
  };
}

function paramsKey(p) {
  return JSON.stringify(window.PhaseParams.toOptions(p));
}

/** Ask for a recompute; keeps one job in flight and only the newest queued. */
function requestRun({ preview }) {
  if (!state.image || !worker) return;
  const request = {
    imageId: state.imageSeq,
    options: window.PhaseParams.toOptions(state.params),
    key: paramsKey(state.params),
    preview,
  };
  if (jobs.inFlight) {
    // A full-resolution request is never downgraded by a later preview.
    if (jobs.queued && !jobs.queued.preview && request.preview && jobs.queued.key === request.key) return;
    jobs.queued = request;
    return;
  }
  send(request);
}

function send(request) {
  const job = ++jobs.seq;
  jobs.inFlight = { ...request, job };
  jobs.startedAt = performance.now();
  el.progressBar.value = 0;
  clearTimeout(jobs.progressTimer);
  // Only show progress for work that takes long enough to notice.
  jobs.progressTimer = setTimeout(() => {
    el.progressLabel.textContent = request.preview ? 'Computing preview…' : 'Computing full resolution…';
    el.viewerProgress.hidden = false;
  }, 200);
  worker.postMessage({ type: 'run', job, imageId: request.imageId, options: request.options, preview: request.preview });
}

function onResult(m) {
  const flight = jobs.inFlight;
  jobs.inFlight = null;
  clearTimeout(jobs.progressTimer);
  el.viewerProgress.hidden = true;

  const current = state.image && m.imageId === state.imageSeq && flight && flight.key === paramsKey(state.params);
  const next = jobs.queued;
  jobs.queued = null;

  // Stale: the parameters moved on while this ran. Discard it.
  if (current) {
    state.live = m;
    state.liveVersion++;
    for (const prefix of ['removedUnit:', 'errClassical:']) pruneCache(prefix);
    el.timing.textContent = `Last run ${Math.round(m.elapsed)} ms in a background worker · ${m.width} × ${m.height}${m.preview ? ' preview' : ''}`;
    viewer.refreshTags();
    viewer.request();
    renderMeasured();
    renderProfile();
    emitReadoutAgain();
    if (!m.preview && jobs.announce) {
      jobs.announce = false;
      announceResult();
    }
    // A preview is followed by the full image once the slider is released.
    if (m.preview && !next) {
      /* the release ('change') event requests the full run */
    }
  }
  if (next && next.imageId === state.imageSeq) send(next);
  else if (!current && state.image && !next) requestRun({ preview: false });
}

function announceResult() {
  const live = state.live;
  if (!live) return;
  if (live.metrics) {
    const m = live.metrics;
    announce(`Classical, live: PSNR ${fmt(m.psnr, 2)} decibels, SSIM ${fmt(m.ssim, 3)}, Otsu Dice ${fmt(m.dice, 3)}.`);
  } else if (live.noRef) {
    const r = live.noRef;
    announce(`Classical, live: edge energy ${fmt(r.edgeEnergy.ratio, 2)} times the input, background variance ${fmt(r.backgroundVariance.ratio, 2)} times.`);
  }
}

// ---------------------------------------------------------------- viewer
function initViewer() {
  viewer = new Viewer({
    root: el.viewer,
    canvas: el.viewCanvas,
    divider: el.divider,
    scanHandle: el.scanHandle,
    tagLayer: el.gridTags,
    provider,
    on: {
      viewChange: () => {
        if (!viewer) return; // the constructor's first resize
        el.zoomLevel.textContent = `${Math.round(viewer.scale * 100)}%`;
        el.zoomFit.setAttribute('aria-pressed', String(viewer.zoom === null));
      },
      splitChange: (split) => {
        state.view.split = split;
      },
      splitCommit: () => scheduleHash(),
      gridPick: (id) => {
        state.view.right = id;
        state.view.mode = 'split';
        applyView();
        announce(`${stageInfo(id).label} is now on the right of the split.`);
      },
      readout: renderReadout,
      scanChange: (row) => {
        if (row !== null) state.scanRow = row;
        renderProfile();
      },
      scanCommit: () => scheduleHash(),
      holdChange: (on) => {
        el.holdInput.classList.toggle('active', on);
        el.holdInput.setAttribute('aria-pressed', String(on));
      },
    },
  });

  el.zoomIn.addEventListener('click', () => viewer.zoomStep(1));
  el.zoomOut.addEventListener('click', () => viewer.zoomStep(-1));
  el.zoomFit.addEventListener('click', () => viewer.fit());
  el.zoomActual.addEventListener('click', () => viewer.actualSize());
  el.modeSplit.addEventListener('click', () => {
    state.view.mode = 'split';
    applyView();
  });
  el.modeGrid.addEventListener('click', () => {
    state.view.mode = 'grid';
    applyView();
  });

  // Hold to compare: pointer, or Space/Enter on the button.
  const hold = (on) => viewer.setHold(on);
  el.holdInput.addEventListener('pointerdown', (e) => {
    el.holdInput.setPointerCapture(e.pointerId);
    hold(true);
  });
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) el.holdInput.addEventListener(type, () => hold(false));
  el.holdInput.addEventListener('keydown', (e) => {
    if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) {
      e.preventDefault();
      hold(true);
    }
  });
  el.holdInput.addEventListener('keyup', (e) => {
    if (e.key === ' ' || e.key === 'Enter') hold(false);
  });
  el.holdInput.addEventListener('blur', () => hold(false));
  el.holdInput.addEventListener('click', (e) => e.preventDefault());

  for (const side of ['left', 'right']) {
    const select = el[`${side}Stage`];
    select.addEventListener('change', () => {
      state.view[side] = select.value;
      applyView();
    });
  }
  window.matchMedia('(prefers-contrast: more)').addEventListener?.('change', () => {
    viewer.readColors();
    viewer.request();
  });
}

/** Push state.view into the viewer, pickers and toolbar. */
function applyView() {
  const v = state.view;
  if (!stageAvailable(v.left)) v.left = 'input';
  if (!stageAvailable(v.right)) v.right = state.image?.hybrid ? 'hybrid' : 'classical';
  el.modeSplit.setAttribute('aria-pressed', String(v.mode === 'split'));
  el.modeGrid.setAttribute('aria-pressed', String(v.mode === 'grid'));
  viewer.setGridStages([...GRID_STAGES.filter(stageAvailable), ...state.pins.map((p) => p.id)]);
  if (viewer.mode !== v.mode) viewer.setMode(v.mode);
  viewer.setSplit(v.split, false);
  viewer.setStages(v.left, v.right);
  refreshPickers();
  updateCanvasLabel();
  renderProfile();
  scheduleHash();
}

function refreshPickers() {
  if (!state.image) return;
  const ids = [...Object.keys(STAGES), ...state.pins.map((p) => p.id)].filter(stageAvailable);
  for (const side of ['left', 'right']) {
    const select = el[`${side}Stage`];
    const value = state.view[side];
    select.replaceChildren();
    for (const id of ids) {
      const info = stageInfo(id);
      const option = document.createElement('option');
      option.value = id;
      option.textContent = info.label;
      select.append(option);
    }
    select.value = value;
    const info = stageInfo(value);
    select.style.setProperty('--stage-color', `var(--v-${info.color})`);
    select.title = `${info.label} · ${info.source}`;
  }
  el.leftStage.disabled = el.rightStage.disabled = state.view.mode === 'grid';
}

function updateCanvasLabel() {
  const v = state.view;
  const label = v.mode === 'grid'
    ? `Grid of ${viewer.gridStages.map((id) => stageInfo(id).label).join(', ')}`
    : `${stageInfo(v.left).label} on the left and ${stageInfo(v.right).label} on the right, divided at ${Math.round(v.split * 100)}%`;
  el.viewCanvas.setAttribute('aria-label', `${label}. ${el.imageSource.textContent}.`);
}

// ------------------------------------------------------------- read-out
let lastReadout = null;
function renderReadout(info) {
  lastReadout = info;
  if (!info) {
    el.readout.textContent = '';
    return;
  }
  el.readout.replaceChildren();
  const pos = document.createElement('span');
  pos.textContent = `x ${info.x}  y ${info.y}`;
  el.readout.append(pos);
  for (const id of info.stages) {
    const v = stageValue(id, info.x, info.y);
    if (v === null || v === undefined) continue;
    const signed = STAGES[id]?.kind === 'diverging';
    const item = document.createElement('span');
    const key = document.createElement('span');
    key.className = 'k';
    key.style.setProperty('--stage-color', `var(--v-${stageInfo(id).color})`);
    key.textContent = ` · ${stageInfo(id).label} `;
    const val = document.createElement('b');
    val.textContent = signed ? `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}` : v.toFixed(3);
    item.append(key, val);
    el.readout.append(item);
  }
}
function emitReadoutAgain() {
  if (lastReadout) renderReadout(lastReadout);
}

// -------------------------------------------------------------- overlays
function initOverlays() {
  const bind = (input, key) => {
    input.addEventListener('change', () => {
      state.overlays[key] = input.checked;
      if (key === 'profile') {
        if (input.checked && state.scanRow === null) state.scanRow = Math.floor((state.image?.height ?? 2) / 2);
        el.profile.hidden = !input.checked;
        viewer.setScanRow(input.checked ? state.scanRow : null);
        requestAnimationFrame(() => viewer.resize());
      }
      viewer.request();
      scheduleHash();
    });
  };
  bind(el.ovMask, 'mask');
  bind(el.ovTruth, 'truth');
  bind(el.ovProfile, 'profile');
  el.lutSelect.addEventListener('change', () => {
    state.lut = el.lutSelect.value;
    viewer.request();
    scheduleHash();
  });
  new ResizeObserver(() => renderProfile()).observe(el.profile);
}

function renderProfile() {
  if (!state.overlays.profile || !state.image || state.scanRow === null) return;
  const row = state.scanRow;
  const v = state.view;
  // The live stage on screen, or Classical.
  const liveId = [v.right, v.left].find((id) => stageIsLive(id) && STAGES[id].kind === 'grey') || 'classical';
  const ids = ['input', liveId, 'hybrid', 'truth'].filter((id, i, all) => stageAvailable(id) && all.indexOf(id) === i);
  const series = [];
  for (const id of ids) {
    const values = new Float32Array(state.image.width);
    let ok = true;
    for (let x = 0; x < values.length; x++) {
      const val = stageValue(id, x, row);
      if (val === null) {
        ok = false;
        break;
      }
      values[x] = val;
    }
    if (ok) series.push({ id, label: stageInfo(id).label, color: stageInfo(id).color, values });
  }
  drawProfile(el.profileSvg, series);
  el.profileCaption.textContent =
    `Row ${row}: intensity 0–1 across the full width. A halo shows as an overshoot either side of each cell edge; suppression flattens it.`;
}

// -------------------------------------------------------------- measured
function deltaText(delta, unit, digits, threshold) {
  if (!Number.isFinite(delta)) return { text: '', cls: '' };
  if (Math.abs(delta) < threshold) return { text: 'no change', cls: '' };
  const sign = delta > 0 ? '+' : '−';
  return {
    text: `${sign}${Math.abs(delta).toFixed(digits)}${unit}, ${delta > 0 ? 'better' : 'worse'}`,
    cls: delta > 0 ? 'better' : 'worse',
  };
}

function renderMeasured() {
  const image = state.image;
  if (!image) return;
  if (image.kind === 'upload' || !image.truth) {
    renderNoReference();
    return;
  }
  const s = state.staticMetrics;
  const live = state.live;
  const rows = [
    { id: 'input', sub: 'original', m: s?.input },
    { id: 'classical', sub: live?.preview ? 'live, preview' : 'live', m: live && !live.preview ? live.metrics : null, live: true },
  ];
  if (image.hybrid) {
    rows.push({ id: 'hybrid', sub: state.params.bits === 8 ? 'precomputed · trained on 16-bit' : 'precomputed', m: s?.hybrid });
  }
  const base = s?.input;
  const body = el.measureTable.tBodies[0];
  body.replaceChildren();
  for (const r of rows) {
    const tr = document.createElement('tr');
    if (r.live) tr.className = 'live';
    const th = document.createElement('th');
    th.scope = 'row';
    const name = document.createElement('span');
    name.className = 'stage-name';
    name.style.setProperty('--stage-color', `var(--stage-${STAGES[r.id].color})`);
    name.textContent = STAGES[r.id].label;
    const sub = document.createElement('span');
    sub.className = 'stage-sub';
    sub.textContent = r.sub;
    th.append(name, sub);
    tr.append(th);
    for (const [key, unit, digits, dDigits, thr] of [
      ['psnr', ' dB', 2, 1, 0.05],
      ['ssim', '', 3, 3, 0.0005],
      ['dice', '', 3, 3, 0.0005],
    ]) {
      const td = document.createElement('td');
      const value = r.m?.[key];
      const main = document.createElement('span');
      main.textContent = Number.isFinite(value) ? `${value.toFixed(digits)}${unit}` : '…';
      td.append(main);
      if (r.id !== 'input' && base && Number.isFinite(value)) {
        const d = deltaText(value - base[key], unit, dDigits, thr);
        const span = document.createElement('span');
        span.className = `delta ${d.cls}`;
        span.textContent = d.text;
        td.append(span);
      }
      tr.append(td);
    }
    body.append(tr);
  }
  el.measureSource.textContent = 'live, against the ground truth';
  renderPythonRef();
}

/** M17: Python's numbers, but only where they still apply. */
function renderPythonRef() {
  const m = state.image?.entry?.metrics;
  if (!m) {
    el.pythonRef.textContent = '';
    return;
  }
  const bits = state.params.bits;
  const atDefaults = window.PhaseParams.atDefaults(state.params, state.defaults);
  const ref = (x) => `${fmt(x.psnr, 2)} dB · ${fmt(x.ssim, 3)} · ${fmt(x.dice, 3)}`;
  const cv = m.cv_only?.[bits === 8 ? 'uint8' : 'uint16'];
  el.pythonRef.replaceChildren();
  const title = document.createElement('b');
  title.textContent = 'Measured in Python: ';
  const parts = [`Input ${ref(m.raw)}`];
  parts.push(`Classical ${atDefaults && cv ? `${ref(cv)} (${bits}-bit defaults)` : '— sliders moved'}`);
  if (m.hybrid) parts.push(`Hybrid ${ref(m.hybrid)}`);
  const text = document.createElement('span');
  text.textContent = `${parts.join('; ')}. The live numbers are computed from the 8-bit PNGs this page loads, so they can differ in the third decimal.`;
  el.pythonRef.append(title, text);
}

function renderNoReference() {
  const live = state.live;
  const body = el.noRefTable.tBodies[0];
  body.replaceChildren();
  el.measureSource.textContent = 'live, your image, no ground truth';
  if (!live || !live.noRef) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 4;
    td.textContent = live?.preview ? 'Measured when you release the slider…' : 'Measuring…';
    tr.append(td);
    body.append(tr);
    el.pythonRef.textContent = '';
    return;
  }
  const r = live.noRef;
  const f = (v) => (v >= 100 || v < 0.01 ? v.toExponential(2) : v.toFixed(4));
  const rows = [
    ['Edge energy', 'higher = sharper edges', r.edgeEnergy.before, r.edgeEnergy.after, r.edgeEnergy.ratio, true],
    ['Background variance', 'lower = flatter background', r.backgroundVariance.before, r.backgroundVariance.after, r.backgroundVariance.ratio, false],
  ];
  for (const [name, hint, before, after, ratio, higherBetter] of rows) {
    const tr = document.createElement('tr');
    const better = higherBetter ? ratio > 1 : ratio < 1;
    tr.innerHTML = `<th scope="row"><span class="stage-name">${name}</span><span class="stage-sub">${hint}</span></th>
      <td>${f(before)}</td><td>${f(after)}</td>
      <td>${fmt(ratio, 2)}×<span class="delta ${better ? 'better' : 'worse'}">${better ? 'better' : 'worse'}</span></td>`;
    body.append(tr);
  }
  const fg = document.createElement('tr');
  fg.innerHTML = `<th scope="row"><span class="stage-name">Otsu foreground</span><span class="stage-sub">share of pixels called cells</span></th>
    <td>${fmt(r.foreground.before * 100, 1)}%</td><td>${fmt(r.foreground.after * 100, 1)}%</td>
    <td><span class="delta">${fmt((r.foreground.after - r.foreground.before) * 100, 1)} points</span></td>`;
  body.append(fg);
  el.pythonRef.textContent =
    'These say whether the halo was suppressed, not whether the result is faithful to any particular truth.';
}

// ----------------------------------------------------------------- pins
function pinCurrent() {
  const live = state.live;
  if (!live || live.preview) {
    viewerNote('Release the slider first: only a full-resolution result can be pinned.');
    return;
  }
  const changed = CONTROLS.filter((c) => Math.abs(state.params[c.key] - state.defaults[c.key]) > 1e-9)
    .map((c) => `${c.symbol} ${fmt(state.params[c.key], c.digits)}`);
  if (state.params.bits !== state.defaults.bits) changed.push(`${state.params.bits}-bit`);
  const label = `Pinned · ${changed.length ? changed.join(', ') : 'defaults'}`;
  const id = `pin-${++state.pinSeq}`;
  state.pins.push({ id, label, data: Float32Array.from(live.result), params: { ...state.params } });
  if (state.pins.length > MAX_PINS) state.pins.shift();
  renderPins();
  applyView();
  announce(`${label} added. Choose it in either side's picker.`);
}

function renderPins() {
  el.pinList.replaceChildren();
  for (const pin of state.pins) {
    const li = document.createElement('li');
    const span = document.createElement('span');
    span.textContent = pin.label;
    const show = document.createElement('button');
    show.type = 'button';
    show.className = 'button ghost small';
    show.textContent = 'show left';
    show.addEventListener('click', () => {
      state.view.left = pin.id;
      state.view.mode = 'split';
      applyView();
    });
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'button ghost small';
    remove.textContent = 'remove';
    remove.setAttribute('aria-label', `Remove ${pin.label}`);
    remove.addEventListener('click', () => {
      state.pins = state.pins.filter((p) => p !== pin);
      renderPins();
      applyView();
    });
    li.append(span, show, remove);
    el.pinList.append(li);
  }
}

// --------------------------------------------------------------- export
function yamlText() {
  const p = state.params;
  const n = (c, v) => fmt(v, Math.max(1, c.digits));
  const lines = [
    'preprocessor:',
    `  dog_sigma1: ${n(BY_KEY.sigma1, p.sigma1)}`,
    `  dog_sigma2: ${n(BY_KEY.sigma2, p.sigma2)}`,
    `  dog_alpha: ${fmt(p.alpha, 2).replace(/0$/, '')}`,
    `  clahe_clip_limit: ${n(BY_KEY.clipLimit, p.clipLimit)}`,
    `  clahe_grid_size: [${p.tiles}, ${p.tiles}]`,
  ];
  if (p.bits === 8) {
    lines.push(
      '  # 8-bit CLAHE was selected. The config and CLI do not read a precision',
      '  # key yet; in Python use Preprocessor(..., clahe_dtype="uint8").',
    );
  }
  return lines.join('\n');
}

function commandText() {
  return 'python main.py --input_dir data/raw/ --output_dir results_cv/ --mode cv_only --config configs/tuned.yaml';
}

function updateExportText() {
  if (!el.yamlOut) return;
  el.yamlOut.textContent = yamlText();
  el.cmdOut.textContent = commandText();
  const eight = state.params.bits === 8;
  el.exportNote.hidden = false;
  el.exportNote.className = eight ? 'hint warn' : 'hint';
  el.exportNote.textContent = eight
    ? 'Save the block as configs/tuned.yaml. 8-bit CLAHE is not supported by the Python config or CLI yet, so the command will run the 16-bit path; the opt-in Preprocessor(clahe_dtype="uint8") parameter does support it.'
    : 'Save the block as configs/tuned.yaml (other sections fall back to their defaults), or paste it over the preprocessor: section of configs/default_config.yaml.';
}

async function copy(text, what) {
  try {
    await navigator.clipboard.writeText(text);
    announce(`${what} copied.`);
    viewerNote(`${what} copied.`, 2000);
  } catch {
    announce(`Could not copy ${what.toLowerCase()}; select the text and copy it instead.`);
  }
}

function initExport() {
  el.copyYaml.addEventListener('click', () => copy(yamlText(), 'YAML'));
  el.copyCmd.addEventListener('click', () => copy(commandText(), 'Command'));
  el.copyLink.addEventListener('click', () => {
    writeHash();
    copy(location.href, 'Link');
  });
  el.exportStage.addEventListener('click', () => exportStage());
  el.exportFigure.addEventListener('click', () => exportFigure());
}

function fileBase() {
  const name = state.image.id || state.image.name.replace(/\.[a-z0-9]+(?=\s|$)/i, '');
  return name.replace(/[^\w-]+/g, '_').replace(/^_+|_+$/g, '');
}

function paramSlug() {
  const p = state.params;
  return `s1-${p.sigma1}_s2-${p.sigma2}_a-${p.alpha}_clip-${p.clipLimit}_t-${p.tiles}_${p.bits}bit`;
}

function download(canvas, name) {
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, 'image/png');
}

function exportStage() {
  const id = state.view.right;
  if (stageIsLive(id) && state.live?.preview) {
    viewerNote('Release the slider first: the export is full resolution.');
    return;
  }
  const canvas = stageCanvas(id);
  if (!canvas) return;
  const base = fileBase();
  download(canvas, `${base}_${id}${stageIsLive(id) ? `_${paramSlug()}` : ''}.png`);
}

function exportFigure() {
  const v = state.view;
  const left = stageCanvas(v.left);
  const right = stageCanvas(v.right);
  if (!left || !right) return;
  const { width, height } = state.image;
  const k = Math.max(1, Math.floor(512 / Math.max(width, height)));
  const pw = width * k;
  const ph = height * k;
  const pad = 16;
  const head = 32;
  const foot = 44;
  const canvas = document.createElement('canvas');
  canvas.width = pw * 2 + pad * 3;
  canvas.height = ph + head + foot + pad;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#1a1d21';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(left, pad, head, pw, ph);
  ctx.drawImage(right, pad * 2 + pw, head, pw, ph);
  ctx.font = '500 14px "IBM Plex Mono", monospace';
  ctx.textBaseline = 'alphabetic';
  const label = (id, x) => {
    const info = stageInfo(id);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(`--v-${info.color}`).trim();
    ctx.fillText(`${info.label} · ${info.source}`, x, head - 10);
  };
  label(v.left, pad);
  label(v.right, pad * 2 + pw);
  const p = state.params;
  ctx.fillStyle = '#b2bcc6';
  ctx.font = '12px "IBM Plex Mono", monospace';
  ctx.fillText(
    `σ₁ ${fmt(p.sigma1, 1)} px · σ₂ ${fmt(p.sigma2, 1)} px · α ${fmt(p.alpha, 2)} · clip ${fmt(p.clipLimit, 1)} · tiles ${p.tiles}×${p.tiles} · CLAHE ${p.bits}-bit`,
    pad, head + ph + 20,
  );
  ctx.fillText(`${el.imageSource.textContent} · Phase-Contrast Clean-Up`, pad, head + ph + 38);
  const base = fileBase();
  download(canvas, `${base}_${v.left}_vs_${v.right}_${paramSlug()}.png`);
}

// ----------------------------------------------------------------- hash
function readHash() {
  const out = {};
  const params = new URLSearchParams(location.hash.slice(1));
  for (const [k, v] of params) out[k] = v;
  return out;
}

/** Apply a parsed hash; anything invalid falls back to the defaults. */
function applyHashView(h) {
  const raw = { bits: h.bits };
  for (const c of CONTROLS) if (h[c.hash] !== undefined) raw[c.key] = Number(h[c.hash]);
  state.params = window.PhaseParams.sanitize({ ...state.defaults, ...raw }, state.defaults);
  const valid = (id) => id && (id in STAGES);
  state.view.left = valid(h.left) ? h.left : 'input';
  state.view.right = valid(h.right) ? h.right : 'hybrid';
  state.view.mode = h.view === 'grid' ? 'grid' : 'split';
  const split = Number(h.split);
  state.view.split = Number.isFinite(split) && split >= 0 && split <= 100 ? split / 100 : 0.5;
  const ov = new Set((h.ov || '').split(',').filter(Boolean));
  state.overlays.mask = ov.has('mask');
  state.overlays.truth = ov.has('truth');
  state.overlays.profile = ov.has('profile');
  const row = Number(h.row);
  state.scanRow = Number.isInteger(row) && row >= 0 ? row : null;
  state.lut = h.lut === 'viridis' ? 'viridis' : 'grey';
  el.ovMask.checked = state.overlays.mask;
  el.ovTruth.checked = state.overlays.truth;
  el.ovProfile.checked = state.overlays.profile;
  el.profile.hidden = !state.overlays.profile;
  el.lutSelect.value = state.lut;
  if (state.image) {
    if (state.overlays.profile && state.scanRow === null) state.scanRow = Math.floor(state.image.height / 2);
    viewer.setScanRow(state.overlays.profile ? state.scanRow : null);
    applyView();
  }
}

function hashString() {
  const p = state.params;
  const v = state.view;
  const h = new URLSearchParams();
  if (state.image?.kind === 'sample') h.set('sample', state.image.id);
  for (const c of CONTROLS) h.set(c.hash, String(p[c.key]));
  h.set('bits', String(p.bits));
  h.set('left', v.left.startsWith('pin-') ? 'classical' : v.left);
  h.set('right', v.right.startsWith('pin-') ? 'classical' : v.right);
  h.set('split', String(Math.round(v.split * 100)));
  if (v.mode === 'grid') h.set('view', 'grid');
  const ov = ['mask', 'truth', 'profile'].filter((k) => state.overlays[k]);
  if (ov.length) h.set('ov', ov.join(','));
  if (state.overlays.profile && state.scanRow !== null) h.set('row', String(state.scanRow));
  if (state.lut !== 'grey') h.set('lut', state.lut);
  return h.toString();
}

function writeHash() {
  if (!state.image) return;
  const next = `#${hashString()}`;
  if (location.hash !== next) history.replaceState(null, '', next);
}

let hashTimer = 0;
function scheduleHash() {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(writeHash, 300);
}

// ----------------------------------------------------------------- tabs
function initTabs() {
  const tabs = [...el.tabList.querySelectorAll('[role="tab"]')];
  const panels = tabs.map((t) => $(t.getAttribute('aria-controls')));
  const mq = window.matchMedia('(max-width: 1023.98px)');
  let active = 'tune';

  const apply = () => {
    const narrow = mq.matches;
    el.tabList.hidden = !narrow;
    tabs.forEach((tab, i) => {
      const name = tab.id.replace('tab-', '');
      const selected = name === active;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (narrow) {
        panels[i].setAttribute('role', 'tabpanel');
        panels[i].hidden = !selected;
      } else {
        panels[i].removeAttribute('role');
        panels[i].hidden = false;
      }
    });
  };
  showTab = (name) => {
    active = name;
    apply();
  };
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => showTab(tab.id.replace('tab-', '')));
    tab.addEventListener('keydown', (e) => {
      const dir = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      let j = null;
      if (dir) j = (i + dir + tabs.length) % tabs.length;
      else if (e.key === 'Home') j = 0;
      else if (e.key === 'End') j = tabs.length - 1;
      if (j === null) return;
      e.preventDefault();
      showTab(tabs[j].id.replace('tab-', ''));
      tabs[j].focus();
    });
  });
  mq.addEventListener('change', () => {
    apply();
    requestAnimationFrame(() => viewer?.resize());
  });
  apply();
}
let showTab = () => {};

// ------------------------------------------------------------- evidence
function buildEvidence() {
  const h = state.manifest.held_out;
  if (!h?.paths) {
    el.evidence.hidden = true;
    return;
  }
  const paths = h.paths;
  const rows = [
    ['raw', 'Raw input', 'input'],
    ['cv_only_uint16', 'Classical, 16-bit (as shipped)', 'classical'],
    ['cv_only_uint8', 'Classical, 8-bit (fixed)', 'classical'],
    ['hybrid', 'Hybrid U-Net', 'hybrid'],
  ];
  const body = el.summaryTable.tBodies[0];
  body.replaceChildren();
  for (const [key, label, color] of rows) {
    const p = paths[key];
    if (!p) continue;
    const tr = document.createElement('tr');
    tr.innerHTML = `<th scope="row" style="--stage-color: var(--stage-${color})">${label}</th>
      <td>${fmt(p.psnr.mean, 2)} dB</td><td>${fmt(p.psnr.min, 2)} dB</td><td>${fmt(p.ssim.mean, 3)}</td>
      <td>${fmt(p.dice.mean, 3)}</td><td>${fmt(p.dice.min, 3)}</td>`;
    body.append(tr);
  }
  const raw = paths.raw;
  const hyb = paths.hybrid;
  const meanMove = hyb ? hyb.dice.mean - raw.dice.mean : 0;
  el.evidenceLede.innerHTML = '';
  const lede = document.createElement('span');
  lede.textContent =
    `${h.frames} frames of ${h.of} that training held out (seed ${h.seed}, ${Math.round(h.val_split * 100)}%), measured in Python by scripts/make_demo_assets.py. `;
  el.evidenceLede.append(lede);
  if (hyb) {
    const s = document.createElement('strong');
    s.textContent = Math.abs(meanMove) < 0.05
      ? `The U-Net lifts mean PSNR from ${fmt(raw.psnr.mean, 2)} to ${fmt(hyb.psnr.mean, 2)} dB, yet mean Otsu Dice barely moves (${fmt(raw.dice.mean, 3)} → ${fmt(hyb.dice.mean, 3)}); what changes is the worst frame, which stops failing (${fmt(raw.dice.min, 3)} → ${fmt(hyb.dice.min, 3)}).`
      : `The U-Net lifts mean PSNR from ${fmt(raw.psnr.mean, 2)} to ${fmt(hyb.psnr.mean, 2)} dB and mean Otsu Dice from ${fmt(raw.dice.mean, 3)} to ${fmt(hyb.dice.mean, 3)}.`;
    el.evidenceLede.append(s);
  }
}

function renderDots() {
  if (!state.manifest?.held_out) return;
  const samples = state.manifest.samples.filter((s) => s.metrics?.hybrid);
  if (!samples.length) return;
  const order = [...samples].sort((a, b) => a.metrics.raw.dice - b.metrics.raw.dice);
  const current = state.image?.kind === 'sample' ? state.image.id : null;
  const open = (id) => {
    openSample(id, { keepView: true });
    el.viewer.scrollIntoView({ block: 'nearest' });
  };
  drawDotPlot(el.dotDice, order, { metric: 'dice', digits: 3, unit: '', current, open, domain: [0.5, 1] });
  drawDotPlot(el.dotPsnr, order, { metric: 'psnr', digits: 2, unit: ' dB', current, open, focusable: false });
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
