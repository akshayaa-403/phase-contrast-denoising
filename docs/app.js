/**
 * Interactive front-end for the phase-contrast clean-up demo.
 *
 * Loads the sample bundle written by scripts/make_demo_assets.py, runs the
 * classical pipeline from cv.js on every parameter change, and recomputes
 * PSNR/SSIM against the ground truth so the numbers move with the sliders.
 */
(() => {
  'use strict';

  const MANIFEST_URL = 'samples/manifest.json';

  const state = {
    manifest: null,
    /** { id, width, height, noisy: Float32Array(0-255), truth|hybrid: Float32Array(0-1) } */
    current: null,
    params: { sigma1: 1, sigma2: 8, alpha: 0.6, clipLimit: 2, tilesX: 8, tilesY: 8 },
    pending: null,
  };

  const el = {};
  const ids = [
    'app', 'noSamples', 'status', 'sampleStrip', 'fileInput', 'sourceLabel',
    'canvasInput', 'canvasDog', 'canvasCv', 'canvasHybrid', 'canvasTruth',
    'hybridStage', 'truthStage', 'metricsTable', 'metricsNote', 'timing',
    'resetBtn', 'sigmaWarning',
  ];
  const SLIDERS = [
    { id: 'sigma1', key: 'sigma1', digits: 1 },
    { id: 'sigma2', key: 'sigma2', digits: 1 },
    { id: 'alpha', key: 'alpha', digits: 2 },
    { id: 'clip', key: 'clipLimit', digits: 1 },
    { id: 'tiles', key: 'tilesX', digits: 0 },
  ];

  // ------------------------------------------------------------ image helpers
  /** Decode an image URL/blob into { pixels: Float32Array(0-255), width, height }. */
  async function loadGray(source) {
    const blob = typeof source === 'string' ? await (await fetch(source)).blob() : source;
    const bitmap = await createImageBitmap(blob);
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);

    const pixels = new Float32Array(bitmap.width * bitmap.height);
    for (let i = 0; i < pixels.length; i++) {
      // The samples are grayscale PNGs, so any channel would do; the ITU
      // luma weights keep a colour upload from shifting brightness.
      pixels[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
    }
    return { pixels, width: bitmap.width, height: bitmap.height };
  }

  /** Paint a [0, 1] float image onto a canvas. */
  function draw(canvas, pixels, width, height) {
    canvas.width = width;
    canvas.height = height;
    const image = new ImageData(width, height);
    for (let i = 0; i < pixels.length; i++) {
      const v = Math.max(0, Math.min(255, Math.round(pixels[i] * 255)));
      image.data[i * 4] = v;
      image.data[i * 4 + 1] = v;
      image.data[i * 4 + 2] = v;
      image.data[i * 4 + 3] = 255;
    }
    canvas.getContext('2d').putImageData(image, 0, 0);
  }

  // ------------------------------------------------------------------ compute
  function recompute() {
    const image = state.current;
    if (!image) return;

    const { sigma1, sigma2, alpha, clipLimit, tilesX, tilesY } = state.params;
    // σ₂ ≤ σ₁ is meaningless (and cv.js throws), so hold the last valid render
    // and say why instead of blanking the page.
    el.sigmaWarning.hidden = sigma2 > sigma1;
    if (sigma2 <= sigma1) return;

    const started = performance.now();
    const { dog, result } = PhaseCV.runCvPipeline(
      image.pixels, image.width, image.height,
      { sigma1, sigma2, alpha, clipLimit, tilesX, tilesY },
    );
    const elapsed = performance.now() - started;

    draw(el.canvasInput, PhaseCV.normalize01(image.pixels), image.width, image.height);
    draw(el.canvasDog, PhaseCV.normalize01(dog), image.width, image.height);
    draw(el.canvasCv, result, image.width, image.height);

    el.timing.textContent = `${Math.round(elapsed)} ms · ${image.width}×${image.height}`;
    updateMetrics(result);
  }

  /** Coalesce slider input into one recompute per animation frame. */
  function scheduleRecompute() {
    if (state.pending) return;
    state.pending = requestAnimationFrame(() => {
      state.pending = null;
      recompute();
    });
  }

  function updateMetrics(cvResult) {
    const image = state.current;
    const body = el.metricsTable.querySelector('tbody');
    body.innerHTML = '';

    if (!image.truth) {
      el.metricsTable.hidden = true;
      el.metricsNote.textContent =
        'Your own image has no clean counterpart, so there is nothing to measure against.';
      return;
    }
    el.metricsTable.hidden = false;

    const { width, height } = image;
    const reference = image.reference || {};
    const rows = [
      ['Input (untouched)', PhaseCV.normalize01(image.pixels), reference.raw],
      ['Classical: DoG + CLAHE', cvResult, reference.cv_only],
    ];
    if (image.hybrid) rows.push(['Hybrid U-Net (precomputed)', image.hybrid, reference.hybrid]);

    for (const [name, pixels, ref] of rows) {
      const p = PhaseCV.psnr(pixels, image.truth);
      const s = PhaseCV.ssim(pixels, image.truth, width, height);
      const row = document.createElement('tr');
      row.innerHTML =
        `<td>${name}</td><td>${p.toFixed(2)}</td><td>${s.toFixed(4)}</td>` +
        `<td class="ref">${ref ? `${ref.psnr.toFixed(2)} / ${ref.ssim.toFixed(4)}` : '—'}</td>`;
      body.appendChild(row);
    }

    el.metricsNote.textContent =
      'The last column is what the Python code measured at the config defaults — it only ' +
      'matches the live numbers while the sliders sit at those defaults.';
  }

  // ------------------------------------------------------------------ samples
  async function selectSample(entry, thumb) {
    setStatus('Loading sample…');
    for (const img of el.sampleStrip.querySelectorAll('img')) img.classList.remove('active');
    if (thumb) thumb.classList.add('active');

    const noisy = await loadGray(`samples/${entry.noisy}`);
    const image = { id: entry.id, ...noisy, reference: entry.metrics || {} };

    if (entry.truth) {
      const truth = await loadGray(`samples/${entry.truth}`);
      image.truth = scaleToUnit(truth.pixels);
    }
    if (entry.hybrid) {
      const hybrid = await loadGray(`samples/${entry.hybrid}`);
      image.hybrid = scaleToUnit(hybrid.pixels);
      draw(el.canvasHybrid, image.hybrid, hybrid.width, hybrid.height);
    }
    el.hybridStage.hidden = !entry.hybrid;
    el.truthStage.hidden = !entry.truth;
    if (image.truth) draw(el.canvasTruth, image.truth, noisy.width, noisy.height);

    state.current = image;
    el.sourceLabel.textContent = `sample ${entry.id}`;
    setStatus('');
    recompute();
  }

  function scaleToUnit(pixels) {
    const out = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i++) out[i] = pixels[i] / 255;
    return out;
  }

  async function useUpload(file) {
    setStatus('Reading image…');
    try {
      const loaded = await loadGray(file);
      state.current = { id: file.name, ...loaded, reference: {} };
      el.hybridStage.hidden = true;
      el.truthStage.hidden = true;
      for (const img of el.sampleStrip.querySelectorAll('img')) img.classList.remove('active');
      el.sourceLabel.textContent = file.name;
      setStatus('');
      recompute();
    } catch (err) {
      setStatus(`Could not read that image: ${err.message}`, true);
    }
  }

  function setStatus(message, error = false) {
    el.status.textContent = message;
    el.status.classList.toggle('error', error);
  }

  // --------------------------------------------------------------------- init
  function applyParamsToSliders() {
    for (const { id, key, digits } of SLIDERS) {
      el[id].value = state.params[key];
      document.getElementById(`${id}Out`).textContent = Number(state.params[key]).toFixed(digits);
    }
  }

  function initSliders() {
    for (const { id, key, digits } of SLIDERS) {
      el[id] = document.getElementById(id);
      el[id].addEventListener('input', () => {
        const value = parseFloat(el[id].value);
        state.params[key] = value;
        // The grid is square, so one control drives both axes.
        if (key === 'tilesX') state.params.tilesY = value;
        document.getElementById(`${id}Out`).textContent = value.toFixed(digits);
        scheduleRecompute();
      });
    }
    el.resetBtn.addEventListener('click', () => {
      state.params = { ...state.manifest.defaults, tilesY: state.manifest.defaults.tilesY };
      applyParamsToSliders();
      recompute();
    });
  }

  async function init() {
    for (const id of ids) el[id] = document.getElementById(id);

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
    state.params = { ...state.params, ...manifest.defaults };
    el.app.hidden = false;
    applyParamsToSliders();
    initSliders();

    el.fileInput.addEventListener('change', (event) => {
      const [file] = event.target.files;
      if (file) useUpload(file);
    });

    for (const entry of manifest.samples) {
      const thumb = new Image();
      thumb.src = `samples/${entry.noisy}`;
      thumb.alt = `Sample ${entry.id}`;
      thumb.title = entry.id;
      thumb.tabIndex = 0;
      thumb.addEventListener('click', () => selectSample(entry, thumb));
      thumb.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectSample(entry, thumb); }
      });
      el.sampleStrip.appendChild(thumb);
    }

    await selectSample(manifest.samples[0], el.sampleStrip.querySelector('img'));
  }

  document.addEventListener('DOMContentLoaded', init);
})();
