/**
 * Interactive front-end for the phase-contrast clean-up demo.
 *
 * Loads the sample bundle written by scripts/make_demo_assets.py, runs the
 * classical pipeline from cv.js on every parameter change, and reports quality:
 * PSNR/SSIM against the clean target for the bundled samples, and no-reference
 * halo measures for an image the visitor uploads (which has no target).
 */
(() => {
  'use strict';

  const MANIFEST_URL = 'samples/manifest.json';

  const state = {
    manifest: null,
    /** { id, width, height, pixels: Float32Array(0-255), truth?, hybrid?, reference } */
    current: null,
    params: { sigma1: 1, sigma2: 8, alpha: 0.6, clipLimit: 2, tilesX: 8, tilesY: 8 },
    pending: null,
  };

  const el = {};
  const ELEMENTS = {
    app: 'app', noSamples: 'noSamples', status: 'status', sampleStrip: 'sampleStrip',
    fileInput: 'fileInput', sourceLabel: 'sourceLabel', canvasInput: 'canvasInput',
    canvasDog: 'canvasDog', canvasCv: 'canvasCv', canvasHybrid: 'canvasHybrid',
    canvasTruth: 'canvasTruth', hybridStage: 'hybridStage', truthStage: 'truthStage',
    metricsTable: 'metricsTable', noRefTable: 'noRefTable', metricsNote: 'metricsNote',
    timing: 'timing', resetBtn: 'resetBtn', sigmaWarning: 'sigmaWarning',
    // Sliders are looked up here too, so nothing depends on call order.
    sigma1: 'sigma1', sigma2: 'sigma2', alpha: 'alpha', clip: 'clip', tiles: 'tiles',
    sigma1Out: 'sigma1Out', sigma2Out: 'sigma2Out', alphaOut: 'alphaOut',
    clipOut: 'clipOut', tilesOut: 'tilesOut',
  };
  const SLIDERS = [
    { id: 'sigma1', key: 'sigma1', digits: 1 },
    { id: 'sigma2', key: 'sigma2', digits: 1 },
    { id: 'alpha', key: 'alpha', digits: 2 },
    { id: 'clip', key: 'clipLimit', digits: 1 },
    { id: 'tiles', key: 'tilesX', digits: 0 },
  ];

  function setStatus(message, error = false) {
    el.status.textContent = message;
    el.status.classList.toggle('error', error);
  }

  // ------------------------------------------------------------ image helpers
  /** Decode an image URL or blob into { pixels: Float32Array(0-255), width, height }. */
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
      // The samples are grayscale, so any channel would do; the ITU luma
      // weights keep a colour upload from shifting brightness.
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

  function scaleToUnit(pixels) {
    const out = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i++) out[i] = pixels[i] / 255;
    return out;
  }

  // ------------------------------------------------------------------ compute
  function recompute() {
    const image = state.current;
    if (!image) return;

    const { sigma1, sigma2, alpha, clipLimit, tilesX, tilesY } = state.params;
    // sigma2 <= sigma1 is meaningless (and cv.js throws), so hold the last valid
    // render and say why rather than blanking the page.
    el.sigmaWarning.hidden = sigma2 > sigma1;
    if (sigma2 <= sigma1) return;

    const started = performance.now();
    const { dog, result } = PhaseCV.runCvPipeline(
      image.pixels, image.width, image.height,
      { sigma1, sigma2, alpha, clipLimit, tilesX, tilesY },
    );
    const elapsed = performance.now() - started;

    const normalizedInput = PhaseCV.normalize01(image.pixels);
    draw(el.canvasInput, normalizedInput, image.width, image.height);
    draw(el.canvasDog, PhaseCV.normalize01(dog), image.width, image.height);
    draw(el.canvasCv, result, image.width, image.height);

    el.timing.textContent = `${Math.round(elapsed)} ms · ${image.width}×${image.height}`;
    if (image.truth) updateReferenceMetrics(result);
    else updateNoReferenceMetrics(normalizedInput, result);
  }

  /** Coalesce slider input into one recompute per animation frame. */
  function scheduleRecompute() {
    if (state.pending) return;
    state.pending = requestAnimationFrame(() => {
      state.pending = null;
      recompute();
    });
  }

  function row(cells, className = '') {
    const tr = document.createElement('tr');
    if (className) tr.className = className;
    cells.forEach((cell, index) => {
      const td = document.createElement(index === 0 ? 'th' : 'td');
      if (index === 0) td.scope = 'row';
      if (typeof cell === 'object' && cell !== null) {
        td.textContent = cell.text;
        if (cell.className) td.className = cell.className;
      } else {
        td.textContent = cell;
      }
      tr.appendChild(td);
    });
    return tr;
  }

  function updateReferenceMetrics(cvResult) {
    const image = state.current;
    el.metricsTable.hidden = false;
    el.noRefTable.hidden = true;

    const body = el.metricsTable.querySelector('tbody');
    body.innerHTML = '';
    const reference = image.reference || {};
    const rows = [
      ['Input (untouched)', PhaseCV.normalize01(image.pixels), reference.raw, ''],
      ['Classical: DoG + CLAHE', cvResult, reference.cv_only, 'row-primary'],
    ];
    if (image.hybrid) rows.push(['Hybrid U-Net (precomputed)', image.hybrid, reference.hybrid, '']);

    for (const [name, pixels, ref, className] of rows) {
      const p = PhaseCV.psnr(pixels, image.truth);
      const s = PhaseCV.ssim(pixels, image.truth, image.width, image.height);
      body.appendChild(row([
        name,
        p.toFixed(2),
        s.toFixed(4),
        { text: ref ? `${ref.psnr.toFixed(2)} / ${ref.ssim.toFixed(4)}` : '—', className: 'ref' },
      ], className));
    }

    el.metricsNote.textContent =
      'The last column is what the Python code measured at the config defaults, so it ' +
      'only matches the live numbers while the sliders sit at those defaults.';
  }

  /**
   * For an uploaded image there is no clean target, so PSNR/SSIM are undefined.
   * These two measures need no target: more edge energy means membranes are
   * sharper, less background variance means the halo and noise flattened.
   */
  function updateNoReferenceMetrics(before, after) {
    const image = state.current;
    el.metricsTable.hidden = true;
    el.noRefTable.hidden = false;

    const report = PhaseCV.noReferenceReport(before, after, image.width, image.height);
    const body = el.noRefTable.querySelector('tbody');
    body.innerHTML = '';

    const format = (value) => (value >= 100 || value < 0.01 ? value.toExponential(2) : value.toFixed(4));
    const change = (ratio, higherIsBetter) => ({
      text: Number.isFinite(ratio) ? `${ratio.toFixed(2)}×` : '—',
      className: Number.isFinite(ratio) && (higherIsBetter ? ratio > 1 : ratio < 1) ? 'better' : '',
    });

    body.appendChild(row([
      'Edge energy (higher = sharper)',
      format(report.edgeEnergy.before),
      format(report.edgeEnergy.after),
      change(report.edgeEnergy.ratio, true),
    ], 'row-primary'));
    body.appendChild(row([
      'Background variance (lower = flatter)',
      format(report.backgroundVariance.before),
      format(report.backgroundVariance.after),
      change(report.backgroundVariance.ratio, false),
    ]));

    el.metricsNote.textContent =
      'Your image has no clean counterpart, so PSNR/SSIM cannot be computed. These two ' +
      'measures need no target — they say whether the halo was suppressed, not whether the ' +
      'result is faithful to any particular truth.';
  }

  // ------------------------------------------------------------------ sources
  async function selectSample(entry, button) {
    setStatus('Loading sample…');
    for (const other of el.sampleStrip.querySelectorAll('button')) {
      other.setAttribute('aria-pressed', String(other === button));
    }

    try {
      const noisy = await loadGray(`samples/${entry.noisy}`);
      const image = { id: entry.id, ...noisy, reference: entry.metrics || {} };

      if (entry.truth) {
        const truth = await loadGray(`samples/${entry.truth}`);
        image.truth = scaleToUnit(truth.pixels);
        draw(el.canvasTruth, image.truth, noisy.width, noisy.height);
      }
      if (entry.hybrid) {
        const hybrid = await loadGray(`samples/${entry.hybrid}`);
        image.hybrid = scaleToUnit(hybrid.pixels);
        draw(el.canvasHybrid, image.hybrid, hybrid.width, hybrid.height);
      }
      el.hybridStage.hidden = !entry.hybrid;
      el.truthStage.hidden = !entry.truth;

      state.current = image;
      el.sourceLabel.textContent = `sample ${entry.id} · synthetic, with a known clean target`;
      setStatus('');
      recompute();
    } catch (err) {
      setStatus(`Could not load that sample: ${err.message}`, true);
    }
  }

  async function useUpload(file) {
    setStatus('Reading image…');
    try {
      const loaded = await loadGray(file);
      state.current = { id: file.name, ...loaded, reference: {} };
      // No paired truth and no precomputed U-Net output for an arbitrary image.
      el.hybridStage.hidden = true;
      el.truthStage.hidden = true;
      for (const button of el.sampleStrip.querySelectorAll('button')) {
        button.setAttribute('aria-pressed', 'false');
      }
      el.sourceLabel.textContent = `${file.name} · your image, no clean target`;
      setStatus('');
      recompute();
    } catch (err) {
      setStatus(`Could not read that image: ${err.message}`, true);
    }
  }

  // --------------------------------------------------------------------- init
  function applyParamsToSliders() {
    for (const { id, key, digits } of SLIDERS) {
      el[id].value = state.params[key];
      el[`${id}Out`].textContent = Number(state.params[key]).toFixed(digits);
    }
  }

  function initSliders() {
    for (const { id, key, digits } of SLIDERS) {
      el[id].addEventListener('input', () => {
        const value = parseFloat(el[id].value);
        state.params[key] = value;
        // The CLAHE grid is square, so one control drives both axes.
        if (key === 'tilesX') state.params.tilesY = value;
        el[`${id}Out`].textContent = value.toFixed(digits);
        scheduleRecompute();
      });
    }
    el.resetBtn.addEventListener('click', () => {
      state.params = { ...state.params, ...state.manifest.defaults };
      applyParamsToSliders();
      recompute();
      setStatus('Parameters reset to the values in configs/default_config.yaml.');
    });
  }

  async function init() {
    // Every element, sliders included, resolved in one pass before anything
    // touches them — an ordering bug here silently kills the whole page.
    for (const [key, id] of Object.entries(ELEMENTS)) el[key] = document.getElementById(id);

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
      const button = document.createElement('button');
      button.type = 'button';
      button.setAttribute('aria-pressed', 'false');
      button.setAttribute('aria-label', `Sample ${entry.id}`);
      const thumb = new Image();
      thumb.src = `samples/${entry.noisy}`;
      thumb.alt = '';
      thumb.loading = 'lazy';
      button.appendChild(thumb);
      button.addEventListener('click', () => selectSample(entry, button));
      el.sampleStrip.appendChild(button);
    }

    await selectSample(manifest.samples[0], el.sampleStrip.querySelector('button'));
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
