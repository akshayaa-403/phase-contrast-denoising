/**
 * Phase-contrast clean-up demo: page wiring.
 *
 * Loads the held-out samples written by scripts/make_demo_assets.py, runs the
 * classical pipeline from cv.js (polarity, background flattening, DoG halo
 * suppression, CLAHE, normalisation) whenever a slider moves, and scores it:
 *   - samples have an ideal (ground-truth) image, so affine-matched PSNR, SSIM
 *     and Otsu Dice are shown, as in the Python report;
 *   - uploads have none, so two measures that need no ideal are shown instead.
 *
 * The residual U-Net's output is pre-computed for samples. If the manifest
 * names an ONNX export of the model (and the file is reachable), uploads get a
 * button that runs it in the browser with onnxruntime-web, loaded on demand.
 */
(() => {
  'use strict';

  const MANIFEST_URL = 'samples/manifest.json';
  const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  const NN_MAX_SIDE = 1024;

  const SLIDERS = [
    { id: 'sigma2', key: 'sigma2', digits: 1 },
    { id: 'alpha', key: 'alpha', digits: 2 },
    { id: 'sigma1', key: 'sigma1', digits: 1 },
    { id: 'tiles', key: 'tilesX', digits: 0 },
    { id: 'clipLimit', key: 'clipLimit', digits: 1 },
  ];

  const state = {
    defaults: { ...PhaseCV.DEFAULTS },
    params: null,
    /** { name, width, height, pixels, truth?, hybrid?, upload? }, all in [0, 1] */
    current: null,
    /** Latest classical result for the current image and params (0-1). */
    cvResult: null,
    pending: 0,
    /** { onnx, multiple } from the manifest, once the file is known to exist. */
    model: null,
    session: null,
  };

  const $ = (id) => document.getElementById(id);
  let el;

  function setStatus(message, error = false) {
    el.status.textContent = message;
    el.status.classList.toggle('error', error);
  }

  // ------------------------------------------------------------- images
  /**
   * Read an image into grey float32 pixels in [0, 1], scaled by bit depth as
   * Python's load_gray01 does. PNG and TIFF go through decode.js so 16-bit data
   * keeps its precision (browsers would round it to 8 bits, and cannot open
   * TIFF at all).
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
      const grey = Math.floor((data[i * 4] * 4899 + data[i * 4 + 1] * 9617 + data[i * 4 + 2] * 1868 + 8192) / 16384);
      pixels[i] = Math.fround(grey / 255);
    }
    return { pixels, width: bitmap.width, height: bitmap.height };
  }

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
  /** runCvPipeline options: the sliders plus the fixed settings from the manifest. */
  const options = () => ({ ...state.params, tilesY: state.params.tilesX });

  function recompute() {
    const image = state.current;
    if (!image) return;

    const p = state.params;
    // The wide blur must be wider than the narrow one, or there is no halo band.
    el.sigmaWarning.hidden = p.sigma2 > p.sigma1;
    if (p.sigma2 <= p.sigma1) return;

    const started = performance.now();
    const stages = PhaseCV.runCvPipeline(image.pixels, image.width, image.height, options());
    const ms = Math.round(performance.now() - started);
    state.cvResult = stages.result;

    // Show the input as it is; score it polarity-matched, the fair baseline
    // against an ideal image with bright cells.
    draw(el.canvasInput, PhaseCV.normalize01(image.pixels), image.width, image.height);
    draw(el.canvasCv, stages.result, image.width, image.height);
    el.timing.textContent = `Processed ${image.width}×${image.height} px in ${ms} ms, in your browser.`;

    // A network result computed in this tab belongs to the slider values it was
    // run with; once they move it is stale.
    if (image.upload && image.hybrid && image.hybridKey !== JSON.stringify(p)) {
      image.hybrid = null;
      el.hybridStage.hidden = true;
    }
    updateNnButton();

    const scored = PhaseCV.normalize01(stages.polarity);
    if (image.truth) showScores(scored, stages.result);
    else showNoReferenceScores(scored, stages.result);
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

    const score = (pixels) => {
      const fitted = PhaseCV.affineMatch(pixels, image.truth);
      return {
        psnr: PhaseCV.psnr(fitted, image.truth),
        ssim: PhaseCV.ssim(fitted, image.truth, image.width, image.height),
        dice: PhaseCV.otsuDice(pixels, image.truth),
      };
    };
    const base = score(input);
    const cell = (value, baseValue, digits, unit) => {
      const delta = value - baseValue;
      return {
        text: value.toFixed(digits),
        change: `${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(digits)}${unit}`,
        good: delta >= 0,
      };
    };
    const row = (label, s, cls) => addRow(body, [label,
      cell(s.psnr, base.psnr, 2, ' dB'), cell(s.ssim, base.ssim, 3, ''), cell(s.dice, base.dice, 3, '')], cls);

    addRow(body, ['Before', base.psnr.toFixed(2), base.ssim.toFixed(3), base.dice.toFixed(3)]);
    row('Classical', score(cvResult), 'row-primary');
    if (image.hybrid) row('+ Neural network', score(image.hybrid));

    el.metricsNote.textContent =
      'Before is the input with its polarity flipped so cells are bright, like the ideal. ' +
      'PSNR and SSIM are measured after fitting a·x + b of each image to the ideal (as the ' +
      'Python report does), so a method is not penalised for grey levels only a trained model ' +
      'could know. Dice is the overlap between an automatic (Otsu) segmentation of the image ' +
      'and of the ideal: it is what a cell-counting step downstream would see.' +
      (image.hybrid ? ' The neural-network row uses the default settings it was trained with.' : '');
  }

  /** Uploads have no ideal image, so measure halo removal directly instead. */
  function showNoReferenceScores(before, after) {
    const image = state.current;
    el.metricsTable.hidden = true;
    el.noRefTable.hidden = false;
    const body = el.noRefTable.querySelector('tbody');
    body.innerHTML = '';

    const fmt = (v) => (v >= 100 || v < 0.01 ? v.toExponential(2) : v.toFixed(4));
    const verdict = (ratio, higherIsBetter) => {
      if (!Number.isFinite(ratio)) return '—';
      const improved = higherIsBetter ? ratio > 1 : ratio < 1;
      return { text: `${ratio.toFixed(2)}×`, change: improved ? 'improved' : 'not improved', good: improved };
    };
    const rows = [['Classical', after]];
    if (image.hybrid) rows.push(['+ Neural network', image.hybrid]);
    for (const [label, result] of rows) {
      const r = PhaseCV.noReferenceReport(before, result, image.width, image.height);
      addRow(body, [`Edge sharpness (${label})`, fmt(r.edgeEnergy.before), fmt(r.edgeEnergy.after),
        verdict(r.edgeEnergy.ratio, true)], label === 'Classical' ? 'row-primary' : '');
      addRow(body, [`Background unevenness (${label})`, fmt(r.backgroundVariance.before),
        fmt(r.backgroundVariance.after), verdict(r.backgroundVariance.ratio, false)]);
    }

    el.metricsNote.textContent =
      'Your image has no ideal version to compare against, so these measure the artifacts ' +
      'directly: sharper cell edges and a flatter background both mean less halo and less ' +
      'uneven illumination.';
  }

  // ------------------------------------------------------ neural network
  /**
   * Is the ONNX model named in the manifest actually deployed? Its path is
   * relative to the page (e.g. "models/phase_unet.onnx" -> docs/models/).
   */
  async function probeModel(model) {
    if (!model?.onnx) return null;
    try {
      const url = new URL(model.onnx, document.baseURI).href;
      const response = await fetch(url, { method: 'HEAD', cache: 'no-cache' });
      return response.ok ? { multiple: 16, ...model, url } : null;
    } catch {
      return null;
    }
  }

  function updateNnButton() {
    const image = state.current;
    const show = Boolean(state.model && image?.upload && !image.hybrid);
    el.nnRow.hidden = !show;
    if (show) {
      const tooBig = Math.max(image.width, image.height) > NN_MAX_SIDE;
      el.nnBtn.disabled = tooBig;
      el.nnStatus.textContent = tooBig
        ? `The network runs on images up to ${NN_MAX_SIDE} px on a side in the browser; ` +
          `this one is ${image.width}×${image.height}. Use main.py for large images.`
        : '';
    }
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = src;
      script.crossOrigin = 'anonymous';
      script.onload = resolve;
      script.onerror = () => reject(new Error(`could not load ${src}`));
      document.head.appendChild(script);
    });
  }

  async function getSession() {
    if (state.session) return state.session;
    if (!window.ort) await loadScript(`${ORT_DIST}ort.min.js`);
    ort.env.wasm.wasmPaths = ORT_DIST;
    // Several threads need cross-origin isolation, which GitHub Pages cannot enable.
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    state.session = await ort.InferenceSession.create(state.model.url, { executionProviders: ['wasm'] });
    return state.session;
  }

  /** Reflect-pad (no edge repeat, like torch's 'reflect') bottom/right up to a multiple. */
  function padToMultiple(src, width, height, multiple) {
    const W = Math.ceil(width / multiple) * multiple;
    const H = Math.ceil(height / multiple) * multiple;
    const reflect = (i, n) => {
      if (n === 1) return 0;
      while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i;
      return i;
    };
    const out = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      const sy = reflect(y, height) * width;
      for (let x = 0; x < W; x++) out[y * W + x] = src[sy + reflect(x, width)];
    }
    return { data: out, W, H };
  }

  async function runNetwork() {
    const image = state.current;
    const cv = state.cvResult;
    if (!image || !cv) return;
    el.nnBtn.disabled = true;
    try {
      el.nnStatus.textContent = state.session ? 'Running…' : 'Downloading the network (once) and the ONNX runtime…';
      const session = await getSession();
      el.nnStatus.textContent = 'Running the network in your browser…';
      await new Promise((r) => setTimeout(r, 30)); // let the status paint
      const started = performance.now();
      const { data, W, H } = padToMultiple(cv, image.width, image.height, state.model.multiple);
      const input = new ort.Tensor('float32', data, [1, 1, H, W]);
      const output = await session.run({ input });
      const residual = output.residual.data;
      const cleaned = new Float32Array(image.width * image.height);
      for (let y = 0; y < image.height; y++) {
        for (let x = 0; x < image.width; x++) {
          const v = cv[y * image.width + x] - residual[y * W + x];
          cleaned[y * image.width + x] = v < 0 ? 0 : v > 1 ? 1 : v;
        }
      }
      const ms = Math.round(performance.now() - started);
      if (state.current !== image) return; // the visitor moved on meanwhile
      image.hybrid = cleaned;
      image.hybridKey = JSON.stringify(state.params);
      draw(el.canvasHybrid, cleaned, image.width, image.height);
      el.hybridSub.textContent = `run in your browser in ${(ms / 1000).toFixed(1)} s`;
      el.hybridStage.hidden = false;
      el.nnStatus.textContent = '';
      recompute();
    } catch (err) {
      el.nnStatus.textContent = `The network could not run: ${err.message}`;
    } finally {
      el.nnBtn.disabled = false;
    }
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
        image.truth = truth.pixels;
        draw(el.canvasTruth, image.truth, truth.width, truth.height);
      }
      if (hybrid) {
        image.hybrid = hybrid.pixels;
        draw(el.canvasHybrid, image.hybrid, hybrid.width, hybrid.height);
        el.hybridSub.textContent = 'pre-computed, default settings';
      }
      el.truthStage.hidden = !truth;
      el.hybridStage.hidden = !hybrid;
      state.current = image;
      el.sourceLabel.textContent =
        `Sample ${entry.id.replace(/\D/g, '').replace(/^0+/, '') || 0}: synthetic image the network never trained on`;
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
      state.current = { name: file.name, upload: true, ...loaded };
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
    el.nnBtn.addEventListener('click', runNetwork);
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

    for (const [key, value] of Object.entries(manifest.defaults || {})) {
      if (value !== undefined) state.defaults[key] = value;
    }
    state.params = { ...state.defaults };
    state.model = await probeModel(manifest.model);
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
