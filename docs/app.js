/**
 * Phase-contrast clean-up demo: page wiring.
 *
 * Loads the held-out samples written by scripts/make_demo_assets.py, runs the
 * classical pipeline from cv.js (DoG halo suppression, then CLAHE) whenever a
 * slider moves, and scores the result:
 *   - samples have an ideal (ground-truth) image, so PSNR and SSIM are shown;
 *   - uploads have none, so two measures that need no ideal are shown instead.
 */
(() => {
  'use strict';

  const MANIFEST_URL = 'samples/manifest.json';

  const SLIDERS = [
    { id: 'sigma2', key: 'sigma2', digits: 1 },
    { id: 'alpha', key: 'alpha', digits: 2 },
    { id: 'sigma1', key: 'sigma1', digits: 1 },
    { id: 'tiles', key: 'tilesX', digits: 0 },
  ];

  const state = {
    defaults: { sigma1: 1, sigma2: 8, alpha: 0.6, clipLimit: 2, tilesX: 8, tilesY: 8, bits: 16 },
    params: null,
    /** { name, width, height, pixels (0-255), truth?, hybrid? (both 0-1) } */
    current: null,
    pending: 0,
  };

  const $ = (id) => document.getElementById(id);
  let el;

  function setStatus(message, error = false) {
    el.status.textContent = message;
    el.status.classList.toggle('error', error);
  }

  // ------------------------------------------------------------- images
  /**
   * Read an image into grey float32 pixels on the 0-255 scale.
   * PNG and TIFF go through decode.js so 16-bit data keeps its precision
   * (browsers would round it to 8 bits, and cannot open TIFF at all).
   */
  async function loadGray(source) {
    const blob = typeof source === 'string' ? await (await fetch(source)).blob() : source;
    const bytes = new Uint8Array(await blob.arrayBuffer());
    const decoded = await PhaseDecode.decode(bytes).catch(() => null);
    if (decoded) return decoded;

    const bitmap = await createImageBitmap(blob);
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    const pixels = new Float32Array(bitmap.width * bitmap.height);
    for (let i = 0; i < pixels.length; i++) {
      pixels[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
    }
    return { pixels, width: bitmap.width, height: bitmap.height };
  }

  const toUnit = (pixels) => pixels.map((v) => v / 255);

  /** Paint a 0-1 grey image onto a canvas. */
  function draw(canvas, pixels, width, height) {
    canvas.width = width;
    canvas.height = height;
    const image = new ImageData(width, height);
    for (let i = 0; i < pixels.length; i++) {
      const v = Math.max(0, Math.min(255, Math.round(pixels[i] * 255)));
      image.data[i * 4] = image.data[i * 4 + 1] = image.data[i * 4 + 2] = v;
      image.data[i * 4 + 3] = 255;
    }
    canvas.getContext('2d').putImageData(image, 0, 0);
  }

  // ------------------------------------------------------------ compute
  function recompute() {
    const image = state.current;
    if (!image) return;

    const p = state.params;
    // The wide blur must be wider than the narrow one, or there is no halo band.
    el.sigmaWarning.hidden = p.sigma2 > p.sigma1;
    if (p.sigma2 <= p.sigma1) return;

    const started = performance.now();
    const { result } = PhaseCV.runCvPipeline(image.pixels, image.width, image.height, p);
    const ms = Math.round(performance.now() - started);

    const input = PhaseCV.normalize01(image.pixels);
    draw(el.canvasInput, input, image.width, image.height);
    draw(el.canvasCv, result, image.width, image.height);
    el.timing.textContent = `Processed ${image.width}×${image.height} px in ${ms} ms, in your browser.`;

    if (image.truth) showScores(input, result);
    else showNoReferenceScores(input, result);
  }

  function scheduleRecompute() {
    if (state.pending) return;
    state.pending = requestAnimationFrame(() => {
      state.pending = 0;
      recompute();
    });
  }

  // ------------------------------------------------------------- scores
  function addRow(body, cells, className = '') {
    const tr = document.createElement('tr');
    if (className) tr.className = className;
    cells.forEach((cell, i) => {
      const td = document.createElement(i === 0 ? 'th' : 'td');
      if (i === 0) td.scope = 'row';
      if (cell && typeof cell === 'object') {
        td.textContent = cell.text;
        if (cell.change) {
          const span = document.createElement('span');
          span.className = `change ${cell.good ? 'better' : 'worse'}`;
          span.textContent = cell.change;
          td.append(' ', span);
        }
      } else {
        td.textContent = cell;
      }
      tr.appendChild(td);
    });
    body.appendChild(tr);
  }

  /** Samples: score each image against the ideal, and say how it moved from Before. */
  function showScores(input, cvResult) {
    const image = state.current;
    el.metricsTable.hidden = false;
    el.noRefTable.hidden = true;
    const body = el.metricsTable.querySelector('tbody');
    body.innerHTML = '';

    const score = (pixels) => ({
      psnr: PhaseCV.psnr(pixels, image.truth),
      ssim: PhaseCV.ssim(pixels, image.truth, image.width, image.height),
    });
    const base = score(input);
    const cell = (value, baseValue, digits, unit) => {
      const delta = value - baseValue;
      return {
        text: value.toFixed(digits),
        change: `${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(digits)}${unit}`,
        good: delta >= 0,
      };
    };

    addRow(body, ['Before', base.psnr.toFixed(2), base.ssim.toFixed(3)]);
    const cv = score(cvResult);
    addRow(body, ['Classical', cell(cv.psnr, base.psnr, 2, ' dB'), cell(cv.ssim, base.ssim, 3, '')], 'row-primary');
    if (image.hybrid) {
      const hy = score(image.hybrid);
      addRow(body, ['+ Neural network', cell(hy.psnr, base.psnr, 2, ' dB'), cell(hy.ssim, base.ssim, 3, '')]);
    }

    el.metricsNote.textContent =
      'PSNR measures pixel-by-pixel closeness; SSIM measures whether shapes and edges match. ' +
      'The classical step can score below Before on PSNR because contrast boosting changes ' +
      'overall brightness, even while the halo is gone. The neural network corrects that.';
  }

  /** Uploads have no ideal image, so measure halo removal directly instead. */
  function showNoReferenceScores(before, after) {
    const image = state.current;
    el.metricsTable.hidden = true;
    el.noRefTable.hidden = false;
    const body = el.noRefTable.querySelector('tbody');
    body.innerHTML = '';

    const r = PhaseCV.noReferenceReport(before, after, image.width, image.height);
    const fmt = (v) => (v >= 100 || v < 0.01 ? v.toExponential(2) : v.toFixed(4));
    const verdict = (ratio, higherIsBetter) => {
      if (!Number.isFinite(ratio)) return '—';
      const improved = higherIsBetter ? ratio > 1 : ratio < 1;
      return { text: `${ratio.toFixed(2)}×`, change: improved ? 'improved' : 'not improved', good: improved };
    };

    addRow(body, ['Edge sharpness', fmt(r.edgeEnergy.before), fmt(r.edgeEnergy.after),
      verdict(r.edgeEnergy.ratio, true)], 'row-primary');
    addRow(body, ['Background unevenness', fmt(r.backgroundVariance.before), fmt(r.backgroundVariance.after),
      verdict(r.backgroundVariance.ratio, false)]);

    el.metricsNote.textContent =
      'Your image has no ideal version to compare against, so these measure the halo directly: ' +
      'sharper cell edges and a flatter background both mean less halo.';
  }

  // ------------------------------------------------------------ sources
  function markSelected(button) {
    for (const b of el.sampleStrip.querySelectorAll('button')) {
      b.setAttribute('aria-pressed', String(b === button));
    }
  }

  async function selectSample(entry, button) {
    markSelected(button);
    setStatus('Loading…');
    try {
      const [noisy, truth, hybrid] = await Promise.all([
        loadGray(`samples/${entry.noisy}`),
        entry.truth ? loadGray(`samples/${entry.truth}`) : null,
        entry.hybrid ? loadGray(`samples/${entry.hybrid}`) : null,
      ]);
      const image = { name: entry.id, ...noisy };
      if (truth) {
        image.truth = toUnit(truth.pixels);
        draw(el.canvasTruth, image.truth, truth.width, truth.height);
      }
      if (hybrid) {
        image.hybrid = toUnit(hybrid.pixels);
        draw(el.canvasHybrid, image.hybrid, hybrid.width, hybrid.height);
      }
      el.truthStage.hidden = !truth;
      el.hybridStage.hidden = !hybrid;
      state.current = image;
      el.sourceLabel.textContent = `Sample ${entry.id.replace(/\D/g, '').replace(/^0+/, '') || 0}: synthetic test image`;
      setStatus('');
      recompute();
    } catch (err) {
      setStatus(`Could not load that sample: ${err.message}`, true);
    }
  }

  async function useUpload(file) {
    markSelected(null);
    setStatus('Reading image…');
    try {
      const loaded = await loadGray(file);
      state.current = { name: file.name, ...loaded };
      el.truthStage.hidden = true;
      el.hybridStage.hidden = true;
      el.sourceLabel.textContent = `Your image: ${file.name}`;
      setStatus('');
      recompute();
    } catch (err) {
      setStatus(`Could not read that image: ${err.message}`, true);
    }
  }

  // --------------------------------------------------------------- init
  function showParams() {
    for (const { id, key, digits } of SLIDERS) {
      el[id].value = state.params[key];
      el[`${id}Out`].textContent = Number(state.params[key]).toFixed(digits);
    }
  }

  function initControls() {
    for (const { id, key, digits } of SLIDERS) {
      el[id].addEventListener('input', () => {
        const value = parseFloat(el[id].value);
        state.params[key] = value;
        if (key === 'tilesX') state.params.tilesY = value; // square grid
        el[`${id}Out`].textContent = value.toFixed(digits);
        scheduleRecompute();
      });
    }
    el.resetBtn.addEventListener('click', () => {
      state.params = { ...state.defaults };
      showParams();
      recompute();
    });
    el.fileInput.addEventListener('change', (event) => {
      const [file] = event.target.files;
      if (file) useUpload(file);
      event.target.value = '';
    });
  }

  async function init() {
    el = new Proxy({}, { get: (cache, id) => cache[id] || (cache[id] = $(id)) });

    let manifest = null;
    try {
      const response = await fetch(MANIFEST_URL, { cache: 'no-cache' });
      if (response.ok) manifest = await response.json();
    } catch { /* shown below */ }
    if (!manifest?.samples?.length) {
      el.noSamples.hidden = false;
      return;
    }

    state.defaults = { ...state.defaults, ...manifest.defaults };
    state.params = { ...state.defaults };
    el.app.hidden = false;
    showParams();
    initControls();

    for (const entry of manifest.samples) {
      const button = document.createElement('button');
      button.type = 'button';
      button.setAttribute('aria-pressed', 'false');
      button.setAttribute('aria-label', `Sample ${entry.id}`);
      const thumb = new Image();
      thumb.src = `samples/${entry.thumb || entry.noisy}`;
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
