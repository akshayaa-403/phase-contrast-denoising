#!/usr/bin/env node
/**
 * Test that every control on the demo page moves something -- and that the one
 * known not to says so.
 *
 *     node scripts/check_controls.js
 *
 * Sweeps every parameter in docs/params.js across the full range a visitor can
 * reach, on two held-out samples, through docs/cv.js (the code the page runs),
 * with the shipped non-slider settings (polarity, background flattening,
 * percentile normalisation) taken from the manifest:
 *
 *   - 8-bit CLAHE (the default): every parameter, including the clip limit,
 *     must change the output at every value other than its default.
 *   - 16-bit CLAHE (opt-in, `clahe_dtype: uint16`): σ₁, σ₂, α and tiles must
 *     change the output. The clip limit must have NO effect at 8x8 tiles on a
 *     256 px sample, because OpenCV's per-bin ceiling max(int(clip * tileArea /
 *     65536), 1) is 1 for every clip value -- the reason 8-bit is the default.
 *   - PhaseCV.claheCeiling must predict whether the clip control can change
 *     anything: wherever it says the ceiling is constant, the output must not
 *     change either, and vice versa -- including on a large image, where
 *     16-bit clip does work.
 *
 * Node only, no dependencies. Exits non-zero on any failure.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const ROOT = path.resolve(__dirname, '..');
const PhaseCV = require(path.join(ROOT, 'docs', 'cv.js'));
const PhaseParams = require(path.join(ROOT, 'docs', 'params.js'));
const PhaseDecode = require(path.join(ROOT, 'docs', 'decode.js'));

const inflate = async (bytes) => new Uint8Array(zlib.inflateSync(bytes));
const failures = [];

function fail(message) {
  failures.push(message);
  console.log(`  FAIL  ${message}`);
}

function same(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function run(image, params) {
  return PhaseCV.runCvPipeline(image.pixels, image.width, image.height, PhaseParams.toOptions(params)).result;
}

/** Nearest-neighbour upscale, to make a large test image from a sample. */
function upscale(image, factor) {
  const width = image.width * factor;
  const height = image.height * factor;
  const pixels = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      pixels[y * width + x] = image.pixels[Math.floor(y / factor) * image.width + Math.floor(x / factor)];
    }
  }
  return { width, height, pixels };
}

async function loadSample(entry) {
  const bytes = new Uint8Array(fs.readFileSync(path.join(ROOT, 'docs', 'samples', entry.noisy)));
  return PhaseDecode.decodePng(bytes, inflate);
}

/** Sweep one control; report how many values left the output unchanged. */
function sweepControl(image, defaults, control, bits) {
  const base = { ...defaults, bits };
  const reference = run(image, base);
  const from = control.key === 'sigma2' ? PhaseParams.sigma2Min(base.sigma1) : control.min;
  const values = PhaseParams.sweep(control, from);
  const unchanged = [];
  for (const value of values) {
    if (Math.abs(value - defaults[control.key]) < 1e-9) continue;
    const params = { ...base, [control.key]: value };
    if (control.key === 'sigma1' && !(params.sigma2 > value)) params.sigma2 = PhaseParams.sigma2Min(value);
    if (same(run(image, params), reference)) unchanged.push(value);
  }
  return { tested: values.length - 1, unchanged };
}

async function main() {
  const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'docs', 'samples', 'manifest.json'), 'utf8'));
  const defaults = PhaseParams.fromManifest(manifest.defaults);
  // Two samples: the worst and best raw frame by Dice (first and last if the
  // manifest has no per-sample metrics).
  const rawDice = (e) => e.metrics?.raw?.dice ?? 0;
  const byDice = [...manifest.samples].sort((a, b) => rawDice(a) - rawDice(b));
  const entries = [byDice[0], byDice[byDice.length - 1]];

  for (const entry of entries) {
    const image = await loadSample(entry);
    console.log(`\n${entry.id} (${image.width}x${image.height})`);

    for (const bits of [8, 16]) {
      for (const control of PhaseParams.CONTROLS) {
        const { tested, unchanged } = sweepControl(image, defaults, control, bits);
        const label = `${bits}-bit  ${control.key.padEnd(9)}`;
        const knownDead = bits === 16 && control.key === 'clipLimit' && defaults.tiles === 8;
        if (knownDead) {
          if (unchanged.length === tested) {
            console.log(`  ok    ${label} no effect at any of ${tested} values (expected for opt-in 16-bit CLAHE; the default is 8-bit)`);
          } else {
            fail(`${label} now changes the output at ${tested - unchanged.length} value(s): 16-bit clip is no longer inert; update the CLAHE notes in the README and config`);
          }
        } else if (unchanged.length) {
          fail(`${label} left the output unchanged at ${unchanged.join(', ')}`);
        } else {
          console.log(`  ok    ${label} changes the output at all ${tested} non-default values`);
        }
      }
    }
  }

  // The page's "inactive" note comes from claheCeiling; check it tells the truth.
  console.log('\nclip-control state vs actual effect (claheCeiling)');
  const sample = await loadSample(entries[0]);
  const cases = [
    { name: '256 px, 8x8 tiles, 16-bit', image: sample, tiles: 8, bits: 16 },
    { name: '256 px, 2x2 tiles, 16-bit', image: sample, tiles: 2, bits: 16 },
    { name: '256 px, 8x8 tiles, 8-bit', image: sample, tiles: 8, bits: 8 },
    { name: '1024 px, 8x8 tiles, 16-bit', image: upscale(sample, 4), tiles: 8, bits: 16 },
  ];
  const clip = PhaseParams.BY_KEY.clipLimit;
  for (const c of cases) {
    const ceilings = PhaseParams.sweep(clip).map((v) => PhaseCV.claheCeiling(v, c.tiles, c.image.width, c.image.height, c.bits).ceiling);
    const claimsInactive = ceilings.every((v) => v === ceilings[0]);
    const outputs = PhaseParams.sweep(clip).map((v) => run(c.image, { ...defaults, clipLimit: v, tiles: c.tiles, bits: c.bits }));
    const actuallyInactive = outputs.every((o) => same(o, outputs[0]));
    // Equal ceilings must mean equal outputs; distinct ceilings must give distinct outputs.
    let consistent = claimsInactive === actuallyInactive;
    for (let i = 1; i < outputs.length && consistent; i++) {
      if ((ceilings[i] === ceilings[i - 1]) !== same(outputs[i], outputs[i - 1])) consistent = false;
    }
    const range = `${ceilings[0]}-${ceilings[ceilings.length - 1]} px per grey level`;
    if (consistent) {
      console.log(`  ok    ${c.name.padEnd(28)} ceiling ${range}; page says ${claimsInactive ? 'inactive' : 'active'}, output agrees`);
    } else {
      fail(`${c.name}: ceiling ${range} but the output ${actuallyInactive ? 'never' : 'does'} change(s) -- claheCeiling no longer predicts the clip control`);
    }
  }

  console.log(failures.length ? `\n${failures.length} failure(s).` : '\nPASS: every control moves the output, and claheCeiling predicts when clip can act.');
  process.exitCode = failures.length ? 1 : 0;
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
