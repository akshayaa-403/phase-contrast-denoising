/**
 * The demo's tunable parameters, in one place.
 *
 * The page builds its controls from this list, the URL hash is validated
 * against it, the YAML export uses its `yaml` keys (the exact keys under
 * `preprocessor:` in configs/default_config.yaml), and scripts/check_controls.js
 * sweeps every control across the same range the visitor can reach. Defaults
 * are not here: they come from docs/samples/manifest.json, which
 * scripts/make_demo_assets.py writes from the config.
 *
 * UMD-wrapped like cv.js: `self.PhaseParams` in a page or worker, `require` in Node.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.PhaseParams = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const CONTROLS = [
    {
      key: 'sigma1', yaml: 'dog_sigma1', group: 'dog', hash: 's1',
      symbol: 'σ₁', name: 'cell-body scale', unit: 'px',
      min: 0.2, max: 6, step: 0.1, digits: 1,
    },
    {
      key: 'sigma2', yaml: 'dog_sigma2', group: 'dog', hash: 's2',
      symbol: 'σ₂', name: 'halo width', unit: 'px',
      min: 1, max: 24, step: 0.5, digits: 1,
    },
    {
      key: 'alpha', yaml: 'dog_alpha', group: 'dog', hash: 'a',
      symbol: 'α', name: 'subtraction strength', unit: '× DoG',
      min: 0, max: 2, step: 0.05, digits: 2,
    },
    {
      key: 'clipLimit', yaml: 'clahe_clip_limit', group: 'clahe', hash: 'clip',
      symbol: 'clip', name: 'contrast limit', unit: '× a flat histogram',
      min: 0.5, max: 8, step: 0.5, digits: 1,
    },
    {
      key: 'tiles', yaml: 'clahe_grid_size', group: 'clahe', hash: 'tiles',
      symbol: 'tiles', name: 'grid size', unit: 'per side',
      min: 2, max: 16, step: 1, digits: 0,
    },
  ];

  const BY_KEY = Object.fromEntries(CONTROLS.map((c) => [c.key, c]));

  /** Snap to a control's step grid, counted from its minimum, and clamp. */
  function snap(control, value) {
    const steps = Math.round((value - control.min) / control.step);
    const snapped = control.min + steps * control.step;
    const clamped = Math.min(control.max, Math.max(control.min, snapped));
    return Number(clamped.toFixed(control.digits));
  }

  /** The smallest σ₂ on its own step grid that is strictly above σ₁. */
  function sigma2Min(sigma1) {
    const c = BY_KEY.sigma2;
    const steps = Math.floor((sigma1 - c.min) / c.step + 1e-9) + 1;
    return Math.max(c.min, Number((c.min + steps * c.step).toFixed(c.digits)));
  }

  /** Every value a control can take, in order. */
  function sweep(control, from = control.min) {
    const values = [];
    for (let v = from; v <= control.max + 1e-9; v += control.step) values.push(snap(control, v));
    return values;
  }

  /**
   * Settings that are not on a slider but must still reach cv.js: polarity,
   * background-flattening radius and the normalisation percentiles. They come
   * from the manifest (i.e. the YAML config) and pass through unchanged.
   */
  const FIXED = ['invert', 'backgroundRadius', 'pLow', 'pHigh'];

  /** Manifest defaults ({sigma1, …, tilesX, tilesY, bits, invert, …}) -> page params. */
  function fromManifest(defaults) {
    const params = {
      sigma1: defaults.sigma1,
      sigma2: defaults.sigma2,
      alpha: defaults.alpha,
      clipLimit: defaults.clipLimit,
      tiles: defaults.tilesX,
      bits: defaults.bits || 8,
    };
    for (const key of FIXED) if (key in defaults) params[key] = defaults[key];
    return params;
  }

  /** Page params -> cv.js runCvPipeline options. */
  function toOptions(params) {
    const options = {
      sigma1: params.sigma1,
      sigma2: params.sigma2,
      alpha: params.alpha,
      clipLimit: params.clipLimit,
      tilesX: params.tiles,
      tilesY: params.tiles,
      bits: params.bits,
    };
    for (const key of FIXED) if (key in params) options[key] = params[key];
    return options;
  }

  /**
   * Make a params object valid: every value on its grid and in range, σ₂
   * strictly above σ₁, bits 8 or 16. Returns a new object.
   */
  function sanitize(params, fallback) {
    const out = {};
    for (const c of CONTROLS) {
      const v = Number(params[c.key]);
      // Out of range or not a number: the default, not the nearest end.
      const ok = Number.isFinite(v) && v >= c.min - 1e-9 && v <= c.max + 1e-9;
      out[c.key] = ok ? snap(c, v) : fallback[c.key];
    }
    out.bits = params.bits === 8 || params.bits === '8' ? 8 : params.bits === 16 || params.bits === '16' ? 16 : fallback.bits;
    for (const key of FIXED) if (key in fallback) out[key] = fallback[key];
    if (!(out.sigma2 > out.sigma1)) {
      // An invalid pair: keep σ₁ and fall back to the default σ₂ if that is
      // valid, otherwise the smallest σ₂ above σ₁.
      out.sigma2 = fallback.sigma2 > out.sigma1 ? fallback.sigma2 : sigma2Min(out.sigma1);
    }
    return out;
  }

  /** True when every parameter equals its default. */
  function atDefaults(params, defaults) {
    return CONTROLS.every((c) => Math.abs(params[c.key] - defaults[c.key]) < 1e-9);
  }

  return { CONTROLS, BY_KEY, FIXED, snap, sigma2Min, sweep, fromManifest, toOptions, sanitize, atDefaults };
});
