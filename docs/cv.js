/**
 * The project's classical front-end (Difference-of-Gaussians halo suppression
 * followed by CLAHE), ported to plain JavaScript so the demo page can run it
 * live with no server and no OpenCV.js download.
 *
 * This is a deliberate re-implementation of `src/preprocessor.py`, matching
 * OpenCV's behaviour where it is observable:
 *
 *   - `GaussianBlur(src, (0,0), sigma)` derives its kernel size the way OpenCV
 *     does for float images (`ksize = round(sigma * 8 + 1) | 1`) and reflects at
 *     the border (`BORDER_REFLECT_101`).
 *   - `createCLAHE(clipLimit, tileGridSize)` follows OpenCV's clahe.cpp:
 *     per-tile histograms (65536 bins for uint16, 256 for uint8), the clip
 *     limit turned into a per-bin pixel ceiling `max(int(clip * tileArea /
 *     bins), 1)` -- truncated, as the C++ `static_cast<int>` does -- excess
 *     redistributed, then bilinear interpolation between neighbouring tile LUTs.
 *   - Float arithmetic around the quantisation step is done in float32
 *     (`Math.fround`) because that is what NumPy does in the Python code, and
 *     the truncation to an integer is sensitive to the last bit.
 *
 * Agreement with the Python path is measured, not assumed -- see
 * `scripts/verify_web_pipeline.py` and `scripts/check_controls.js`.
 *
 * Images are handled as Float32Array in the 0-255 range (what the Python code
 * gets from `cv2.imread`), with a final `normalize01` to [0, 1].
 *
 * UMD-wrapped: a <script> tag or `importScripts` sets `self.PhaseCV`, and
 * Node's `require` gets the same object.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.PhaseCV = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const f32 = Math.fround;

  // ------------------------------------------------------------- Gaussian
  /** OpenCV's kernel size for GaussianBlur with ksize=(0,0) on a float image. */
  function kernelSizeFor(sigma) {
    return (roundHalfEven(sigma * 4 * 2 + 1) | 1);
  }

  /**
   * Normalised 1-D Gaussian as OpenCV builds it for a float image:
   * getGaussianKernelBitExact (one reciprocal of the sum, multiplied in),
   * then cast to float32.
   */
  function gaussianKernel(sigma) {
    const n = kernelSizeFor(sigma);
    const half = (n - 1) / 2;
    const scale2X = -0.125 / (sigma * sigma);
    const values = new Float64Array(half);
    let sum = 0;
    for (let i = 0, x = 1 - n; i < half; i++, x += 2) {
      values[i] = Math.exp((x * x) * scale2X);
      sum += values[i];
    }
    sum = sum * 2 + 1;
    const mul = 1 / sum;
    const kernel = new Float32Array(n);
    for (let i = 0; i < half; i++) {
      kernel[i] = kernel[n - 1 - i] = values[i] * mul;
    }
    kernel[half] = mul;
    return kernel;
  }

  /**
   * Reflect an out-of-range index without repeating the edge pixel
   * (gfedcb|abcdefgh|gfedcba), which is OpenCV's BORDER_REFLECT_101 default.
   */
  function reflect101(index, length) {
    if (length === 1) return 0;
    while (index < 0 || index >= length) {
      if (index < 0) index = -index;
      if (index >= length) index = 2 * (length - 1) - index;
    }
    return index;
  }

  /** Index table for one axis: padded position -> source position. */
  function reflectTable(length, radius) {
    const table = new Int32Array(length + 2 * radius);
    for (let i = 0; i < table.length; i++) table[i] = reflect101(i - radius, length);
    return table;
  }

  /**
   * Separable Gaussian blur of a single-channel float32 image, reproducing
   * OpenCV's float filter engine operation for operation so the result is
   * bit-identical, not merely close. That matters downstream: CLAHE truncates
   * to integers and Otsu thresholds, and both turn last-bit differences into
   * whole flipped pixels.
   *
   *   rows:    a float32 multiply-add from zero over every tap, in order
   *            (RowVec_32f); kernels of 3 or 5 taps fold symmetrically
   *            (SymmRowSmallVec_32f)
   *   columns: centre tap, then a multiply-add of each symmetric pair
   *            (SymmColumnVec_32f)
   *
   * A fused multiply-add is emulated as fround(a * b + c): the float32 product
   * is exact in a double, so only the addition rounds.
   */
  function gaussianBlur(src, width, height, sigma) {
    if (sigma <= 0) return Float32Array.from(src);
    const kernel = gaussianKernel(sigma);
    const n = kernel.length;
    const radius = (n - 1) / 2;
    const horizontal = new Float32Array(width * height);
    const output = new Float32Array(width * height);

    // Horizontal pass, one reflect-padded row at a time.
    const xTable = reflectTable(width, radius);
    const padded = new Float32Array(width + 2 * radius);
    for (let y = 0; y < height; y++) {
      const row = y * width;
      for (let i = 0; i < padded.length; i++) padded[i] = src[row + xTable[i]];
      if (n === 3) {
        const k0 = kernel[1], k1 = kernel[2];
        for (let x = 0; x < width; x++) {
          const c = x + 1;
          horizontal[row + x] = f32(f32(padded[c - 1] + padded[c + 1]) * k1 + f32(padded[c] * k0));
        }
      } else if (n === 5) {
        const k0 = kernel[2], k1 = kernel[3], k2 = kernel[4];
        for (let x = 0; x < width; x++) {
          const c = x + 2;
          const inner = f32(f32(padded[c - 1] + padded[c + 1]) * k1 + f32(padded[c] * k0));
          horizontal[row + x] = f32(f32(padded[c + 2] + padded[c - 2]) * k2 + inner);
        }
      } else {
        // Four outputs at once: each accumulator still sums its taps in
        // order, but the four chains are independent, so the CPU can overlap
        // them instead of waiting on one long dependency chain.
        let x = 0;
        for (; x + 3 < width; x += 4) {
          let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
          for (let k = 0; k < n; k++) {
            const w = kernel[k];
            const i = x + k;
            a0 = f32(padded[i] * w + a0);
            a1 = f32(padded[i + 1] * w + a1);
            a2 = f32(padded[i + 2] * w + a2);
            a3 = f32(padded[i + 3] * w + a3);
          }
          horizontal[row + x] = a0;
          horizontal[row + x + 1] = a1;
          horizontal[row + x + 2] = a2;
          horizontal[row + x + 3] = a3;
        }
        for (; x < width; x++) {
          let acc = 0;
          for (let k = 0; k < n; k++) acc = f32(padded[x + k] * kernel[k] + acc);
          horizontal[row + x] = acc;
        }
      }
    }

    // Vertical pass, row by row so memory is read in order.
    const yTable = reflectTable(height, radius);
    const acc = new Float32Array(width);
    const k0 = kernel[radius];
    for (let y = 0; y < height; y++) {
      const centre = yTable[y + radius] * width;
      for (let x = 0; x < width; x++) acc[x] = horizontal[centre + x] * k0;
      for (let k = 1; k <= radius; k++) {
        const w = kernel[radius + k];
        const below = yTable[y + radius + k] * width;
        const above = yTable[y + radius - k] * width;
        for (let x = 0; x < width; x++) {
          acc[x] = f32(f32(horizontal[below + x] + horizontal[above + x]) * w + acc[x]);
        }
      }
      output.set(acc, y * width);
    }
    return output;
  }

  /**
   * Halo suppression: `enhanced = image - alpha * (blur(sigma1) - blur(sigma2))`.
   *
   * The halo is broad and low-frequency, so the wide blur captures it while the
   * narrow one keeps the cell body; subtracting their difference removes the
   * bright ring without hollowing out the cell.
   */
  function haloSuppressionDoG(src, width, height, sigma1, sigma2, alpha) {
    if (!(sigma2 > sigma1)) throw new Error(`sigma2 (${sigma2}) must be greater than sigma1 (${sigma1})`);
    const small = gaussianBlur(src, width, height, sigma1);
    const large = gaussianBlur(src, width, height, sigma2);
    const a = f32(alpha);
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) out[i] = src[i] - f32(a * f32(small[i] - large[i]));
    return out;
  }

  // ------------------------------------------------------------- helpers
  /** Min and max of an array. */
  function extent(data) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return [lo, hi];
  }

  /** Min-max scale to [0, 1] in float32, like Preprocessor.normalize_01. */
  function normalize01(data) {
    const [lo, hi] = extent(data);
    const range = f32(hi - lo);
    const out = new Float32Array(data.length);
    if (range < 1e-8) return out;
    for (let i = 0; i < data.length; i++) out[i] = f32(f32(data[i] - lo) / range);
    return out;
  }

  // ------------------------------------------------------------- CLAHE
  /** Histogram size OpenCV uses for each CLAHE input depth. */
  function histSizeFor(bits) {
    return bits === 8 ? 256 : 65536;
  }

  /**
   * The per-grey-level pixel ceiling OpenCV derives from the clip limit.
   *
   * `clipLimit` is relative to a flat histogram, so OpenCV multiplies it by the
   * tile area and divides by the number of bins. With 65536 bins and a small
   * tile that is far below one pixel, the floor of 1 wins, and every clip value
   * gives the same output. The page uses this to tell the visitor when the clip
   * control cannot do anything.
   */
  function claheCeiling(clipLimit, tiles, width, height, bits = 16) {
    const histSize = histSizeFor(bits);
    const tileWidth = Math.ceil(width / tiles);
    const tileHeight = Math.ceil(height / tiles);
    const tileArea = tileWidth * tileHeight;
    const ceiling = clipLimit > 0 ? Math.max(Math.trunc((clipLimit * tileArea) / histSize), 1) : Infinity;
    return { ceiling, histSize, tileArea, tileWidth, tileHeight };
  }

  /**
   * CLAHE on a float image, via the integer quantisation the Python code uses
   * (`cv2.createCLAHE` needs an integer image). `bits` is 16 (the shipped
   * behaviour) or 8 (`Preprocessor(clahe_dtype="uint8")`).
   */
  function applyClahe(src, width, height, clipLimit, tilesX, tilesY, bits = 16) {
    const [lo, hi] = extent(src);
    const range = hi - lo;
    if (range < 1e-8) return Float32Array.from(src);

    const maxValue = histSizeFor(bits) - 1;
    const maxF = f32(maxValue);
    const rangeF = f32(range);
    const quantized = bits === 8 ? new Uint8Array(src.length) : new Uint16Array(src.length);
    for (let i = 0; i < src.length; i++) {
      // ((img - lo) / rng * max).astype(uint), all float32, truncating.
      const v = Math.trunc(f32(f32(f32(src[i] - lo) / rangeF) * maxF));
      quantized[i] = v > maxValue ? maxValue : v < 0 ? 0 : v;
    }

    const equalized = claheInteger(quantized, width, height, clipLimit, tilesX, tilesY, maxValue + 1);

    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) out[i] = f32(f32(f32(equalized[i] / maxF) * rangeF) + lo);
    return out;
  }

  // Scratch buffers survive between calls: the 16-bit LUT table alone is 8 MB
  // at 8x8 tiles, and allocating it on every slider movement costs more than
  // filling it.
  let lutCache = null;
  let histCache = null;

  /** The tile-histogram core, for 8- or 16-bit data. */
  function claheInteger(src, width, height, clipLimit, tilesX, tilesY, histSize) {
    // Same arithmetic as claheCeiling, kept general for non-square grids.
    const tw = Math.ceil(width / tilesX);
    const th = Math.ceil(height / tilesY);
    const area = tw * th;
    const clipCount = clipLimit > 0 ? Math.max(Math.trunc((clipLimit * area) / histSize), 1) : 0;

    const lutScale = f32((histSize - 1) / area);
    const lutLength = tilesX * tilesY * histSize;
    if (!lutCache || lutCache.length < lutLength) lutCache = new Uint16Array(lutLength);
    if (!histCache || histCache.length < histSize) histCache = new Int32Array(histSize);
    const luts = lutCache;
    const hist = histCache;
    hist.fill(0);

    const tilePixels = new Int32Array(area);

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        // Histogram, counting the excess over the ceiling as it accumulates.
        let n = 0;
        let clipped = 0;
        for (let y = 0; y < th; y++) {
          // OpenCV pads the image with BORDER_REFLECT_101 so the grid divides it.
          const row = reflect101(ty * th + y, height) * width;
          for (let x = 0; x < tw; x++) {
            const v = src[row + reflect101(tx * tw + x, width)];
            tilePixels[n++] = v;
            if (++hist[v] > clipCount && clipCount > 0) clipped++;
          }
        }

        // Clipped mass goes back uniformly ("batch"), and the remainder is
        // sprinkled every `step` bins starting at 0, as clahe.cpp does.
        const batch = clipCount > 0 ? Math.floor(clipped / histSize) : 0;
        const residual = clipCount > 0 ? clipped - batch * histSize : 0;
        const step = residual > 0 ? Math.max(Math.floor(histSize / residual), 1) : histSize + 1;
        const limit = clipCount > 0 ? clipCount : 0x7fffffff;

        // Cumulative sum -> LUT. The sprinkled bins are 0, step, 2*step, ...
        // for the first `residual` of them; track the next one instead of
        // testing i % step on all 65536 bins.
        const base = (ty * tilesX + tx) * histSize;
        if (batch === 0 && histSize > 4 * n) {
          // Sparse case (every 16-bit tile under 65536 px): the cumulative sum
          // only changes at occupied bins and at the few sprinkled ones, so
          // the LUT is a step function. Walk those events in order and fill
          // the runs between them. Same values as the dense loop below.
          const occupied = tilePixels.subarray(0, n).sort();
          let sum = 0;
          let sprinkled = 0;
          let nextSprinkle = residual > 0 ? 0 : histSize;
          let k = 0;
          let start = 0;
          while (start < histSize) {
            const nextOccupied = k < n ? occupied[k] : histSize;
            const at = nextOccupied < nextSprinkle ? nextOccupied : nextSprinkle;
            if (at > start) {
              const value = roundHalfEven(f32(sum * lutScale));
              luts.fill(value < 65535 ? value : 65535, base + start, base + at);
            }
            if (at >= histSize) break;
            if (at === nextOccupied) {
              const h = hist[at];
              sum += h > limit ? limit : h;
              while (k < n && occupied[k] === at) k++;
            }
            if (at === nextSprinkle) {
              sum++;
              nextSprinkle = ++sprinkled < residual ? nextSprinkle + step : histSize;
            }
            const value = roundHalfEven(f32(sum * lutScale));
            luts[base + at] = value < 65535 ? value : 65535;
            start = at + 1;
          }
          for (let i = 0; i < n; i++) hist[tilePixels[i]] = 0;
          continue;
        }

        let sum = 0;
        let nextSprinkle = residual > 0 ? 0 : histSize;
        let sprinkled = 0;
        for (let i = 0; i < histSize; i++) {
          const h = hist[i];
          sum += (h > limit ? limit : h) + batch;
          if (i === nextSprinkle) {
            sum++;
            nextSprinkle = ++sprinkled < residual ? nextSprinkle + step : histSize;
          }
          // sum * lutScale never exceeds histSize - 1 by more than rounding,
          // and Uint16Array storage saturates nothing, so clamp explicitly.
          const value = roundHalfEven(f32(sum * lutScale));
          luts[base + i] = value < 65535 ? value : 65535;
        }

        // Reset only the bins this tile touched.
        for (let i = 0; i < n; i++) hist[tilePixels[i]] = 0;
      }
    }

    // Bilinear blend between the four nearest tile LUTs, so tile boundaries do
    // not show up as seams. This is CLAHE_Interpolation_Body from clahe.cpp
    // operation for operation, in float32: the weights are multiples of
    // 1/tileSize, so exact .5 ties are common, and OpenCV breaks them to even.
    const out = histSize === 256 ? new Uint8Array(src.length) : new Uint16Array(src.length);
    const maxOut = histSize - 1;
    const invTw = f32(1 / tw);
    const invTh = f32(1 / th);
    const ind1 = new Int32Array(width);
    const ind2 = new Int32Array(width);
    const xa = new Float32Array(width);
    const xa1 = new Float32Array(width);
    for (let x = 0; x < width; x++) {
      const txf = f32(f32(x * invTw) - 0.5);
      const tx1 = Math.floor(txf);
      xa[x] = f32(txf - tx1);
      xa1[x] = f32(1 - xa[x]);
      ind1[x] = Math.max(tx1, 0) * histSize;
      ind2[x] = Math.min(tx1 + 1, tilesX - 1) * histSize;
    }
    for (let y = 0; y < height; y++) {
      const tyf = f32(f32(y * invTh) - 0.5);
      const ty1 = Math.floor(tyf);
      const ya = f32(tyf - ty1);
      const ya1 = f32(1 - ya);
      const plane1 = Math.max(ty1, 0) * tilesX * histSize;
      const plane2 = Math.min(ty1 + 1, tilesY - 1) * tilesX * histSize;
      const row = y * width;
      for (let x = 0; x < width; x++) {
        const value = src[row + x];
        const a = ind1[x] + value;
        const b = ind2[x] + value;
        const res = f32(
          f32(f32(f32(luts[plane1 + a] * xa1[x]) + f32(luts[plane1 + b] * xa[x])) * ya1) +
          f32(f32(f32(luts[plane2 + a] * xa1[x]) + f32(luts[plane2 + b] * xa[x])) * ya),
        );
        const v = roundHalfEven(res);
        out[row + x] = v > maxOut ? maxOut : v < 0 ? 0 : v;
      }
    }
    return out;
  }

  /** cvRound: round to nearest, ties to even. */
  function roundHalfEven(x) {
    const r = Math.round(x);
    return r - x === 0.5 && (r & 1) === 1 ? r - 1 : r;
  }

  /**
   * The full CV-only pipeline, mirroring `CleanupPipeline._cv_baseline`.
   *
   * @param {Float32Array} src 0-255 grayscale
   * @param {object} options sigma1, sigma2, alpha, clipLimit, tilesX, tilesY, bits
   * @returns {{dog: Float32Array, clahe: Float32Array, result: Float32Array}}
   *   `result` is the [0, 1] output; the intermediates are returned for the
   *   stage-by-stage view.
   */
  function runCvPipeline(src, width, height, options) {
    const {
      sigma1 = 1.0,
      sigma2 = 8.0,
      alpha = 0.6,
      clipLimit = 2.0,
      tilesX = 8,
      tilesY = 8,
      bits = 16,
    } = options || {};

    const dog = haloSuppressionDoG(src, width, height, sigma1, sigma2, alpha);
    const clahe = applyClahe(dog, width, height, clipLimit, tilesX, tilesY, bits);
    return { dog, clahe, result: normalize01(clahe) };
  }

  /** Area-average downscale, for previews of large images and thumbnails. */
  function resizeArea(src, width, height, newWidth, newHeight) {
    const out = new Float32Array(newWidth * newHeight);
    const sx = width / newWidth;
    const sy = height / newHeight;
    for (let y = 0; y < newHeight; y++) {
      const y0 = Math.floor(y * sy);
      const y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
      for (let x = 0; x < newWidth; x++) {
        const x0 = Math.floor(x * sx);
        const x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
        let sum = 0;
        for (let yy = y0; yy < y1; yy++) {
          const row = yy * width;
          for (let xx = x0; xx < x1; xx++) sum += src[row + xx];
        }
        out[y * newWidth + x] = sum / ((y1 - y0) * (x1 - x0));
      }
    }
    return out;
  }

  // ------------------------------------------------------------- metrics
  function clamp01(v) {
    return v < 0 ? 0 : v > 1 ? 1 : v;
  }

  /** PSNR in dB between two [0, 1] images. */
  function psnr(a, b) {
    let mse = 0;
    for (let i = 0; i < a.length; i++) {
      const d = clamp01(a[i]) - clamp01(b[i]);
      mse += d * d;
    }
    mse /= a.length;
    if (mse === 0) return Infinity;
    return 10 * Math.log10(1 / mse);
  }

  /** Sum over every windowSize x windowSize window, via a summed-area table. */
  function windowSums(values, width, height, windowSize) {
    const W = width + 1;
    const table = new Float64Array(W * (height + 1));
    for (let y = 0; y < height; y++) {
      let rowSum = 0;
      for (let x = 0; x < width; x++) {
        rowSum += values[y * width + x];
        table[(y + 1) * W + x + 1] = table[y * W + x + 1] + rowSum;
      }
    }
    const outW = width - windowSize + 1;
    const outH = height - windowSize + 1;
    const out = new Float64Array(Math.max(outW, 0) * Math.max(outH, 0));
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        out[y * outW + x] =
          table[(y + windowSize) * W + x + windowSize] - table[y * W + x + windowSize] -
          table[(y + windowSize) * W + x] + table[y * W + x];
      }
    }
    return out;
  }

  /**
   * Mean SSIM over 7x7 windows, the same window skimage uses by default.
   *
   * skimage applies a uniform filter and crops the border by half a window;
   * this does the same, so the numbers line up with the report's. Window sums
   * come from summed-area tables, which is the same arithmetic in O(N).
   */
  function ssim(a, b, width, height, windowSize = 7) {
    const C1 = 0.01 * 0.01;
    const C2 = 0.03 * 0.03;
    const n = windowSize * windowSize;
    // skimage's default uses the sample covariance (N-1 normalisation).
    const norm = n / (n - 1);

    const N = width * height;
    const A = new Float64Array(N);
    const B = new Float64Array(N);
    const AA = new Float64Array(N);
    const BB = new Float64Array(N);
    const AB = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const va = clamp01(a[i]);
      const vb = clamp01(b[i]);
      A[i] = va; B[i] = vb; AA[i] = va * va; BB[i] = vb * vb; AB[i] = va * vb;
    }
    const sA = windowSums(A, width, height, windowSize);
    const sB = windowSums(B, width, height, windowSize);
    const sAA = windowSums(AA, width, height, windowSize);
    const sBB = windowSums(BB, width, height, windowSize);
    const sAB = windowSums(AB, width, height, windowSize);
    if (!sA.length) return NaN;

    let total = 0;
    for (let i = 0; i < sA.length; i++) {
      const meanA = sA[i] / n;
      const meanB = sB[i] / n;
      const varA = (sAA[i] / n - meanA * meanA) * norm;
      const varB = (sBB[i] / n - meanB * meanB) * norm;
      const covAB = (sAB[i] / n - meanA * meanB) * norm;
      total +=
        ((2 * meanA * meanB + C1) * (2 * covAB + C2)) /
        ((meanA * meanA + meanB * meanB + C1) * (varA + varB + C2));
    }
    return total / sA.length;
  }

  // -------------------------------------------------- Otsu segmentation
  /**
   * `(np.clip(img, 0, 1) * 255).astype(np.uint8)` on a float32 image: the
   * product is rounded to float32 and then truncated, exactly as NumPy does.
   */
  function toUint8(unit) {
    const out = new Uint8Array(unit.length);
    for (let i = 0; i < unit.length; i++) {
      const v = Math.trunc(f32(f32(clamp01(f32(unit[i]))) * 255));
      out[i] = v > 255 ? 255 : v;
    }
    return out;
  }

  /** OpenCV's getThreshVal_Otsu_8u, line for line. */
  function otsuThreshold(u8) {
    const N = 256;
    const hist = new Float64Array(N);
    for (let i = 0; i < u8.length; i++) hist[u8[i]]++;
    const scale = 1 / u8.length;
    let mu = 0;
    for (let i = 0; i < N; i++) mu += i * hist[i];
    mu *= scale;

    const FLT_EPSILON = 1.1920928955078125e-7;
    let mu1 = 0;
    let q1 = 0;
    let maxSigma = 0;
    let maxVal = 0;
    for (let i = 0; i < N; i++) {
      const p = hist[i] * scale;
      mu1 *= q1;
      q1 += p;
      const q2 = 1 - q1;
      if (Math.min(q1, q2) < FLT_EPSILON || Math.max(q1, q2) > 1 - FLT_EPSILON) continue;
      mu1 = (mu1 + i * p) / q1;
      const mu2 = (mu - q1 * mu1) / q2;
      const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
      if (sigma > maxSigma) {
        maxSigma = sigma;
        maxVal = i;
      }
    }
    return maxVal;
  }

  /**
   * Foreground mask of a [0, 1] image by a global Otsu threshold, as
   * `cv2.threshold(u8, 0, 255, THRESH_BINARY + THRESH_OTSU)` makes it.
   * Returns the mask (1 = foreground) and the threshold on the 0-255 scale.
   */
  function otsuMask(unit) {
    const u8 = toUint8(unit);
    const threshold = otsuThreshold(u8);
    const mask = new Uint8Array(u8.length);
    let count = 0;
    for (let i = 0; i < u8.length; i++) {
      if (u8[i] > threshold) {
        mask[i] = 1;
        count++;
      }
    }
    return { mask, threshold, fraction: count / u8.length };
  }

  /** Dice overlap of two binary masks; two empty masks agree perfectly. */
  function maskDice(a, b) {
    let inter = 0;
    let total = 0;
    for (let i = 0; i < a.length; i++) {
      inter += a[i] & b[i];
      total += a[i] + b[i];
    }
    return total > 0 ? (2 * inter) / total : 1;
  }

  /**
   * Port of `src/report.py::_otsu_dice`: segment both images with Otsu and
   * report how well the cell masks overlap. 1 is a perfect match.
   */
  function otsuDice(cleaned, truth) {
    return maskDice(otsuMask(cleaned).mask, otsuMask(truth).mask);
  }

  // ------------------------------------------------- no-reference measures
  /**
   * Mean squared gradient magnitude (Sobel), a stand-in for "how much edge
   * detail is present". Cell membranes are edges, so more is better here --
   * but only alongside the background measure below, since amplifying noise
   * also raises it.
   */
  function edgeEnergy(data, width, height) {
    let total = 0;
    let count = 0;
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const i = y * width + x;
        const gx =
          (data[i - width + 1] + 2 * data[i + 1] + data[i + width + 1]) -
          (data[i - width - 1] + 2 * data[i - 1] + data[i + width - 1]);
        const gy =
          (data[i + width - 1] + 2 * data[i + width] + data[i + width + 1]) -
          (data[i - width - 1] + 2 * data[i - width] + data[i - width + 1]);
        total += gx * gx + gy * gy;
        count++;
      }
    }
    return count ? total / count : 0;
  }

  /**
   * Variance of the flattest tiles, as a proxy for background noise.
   *
   * The flattest quarter of an 8x8 tiling is almost certainly background
   * rather than cells, so its variance tracks what the halo suppression is
   * doing to empty space. Lower is better.
   */
  function backgroundVariance(data, width, height, tiles = 8) {
    const variances = [];
    const tileW = Math.ceil(width / tiles);
    const tileH = Math.ceil(height / tiles);

    for (let ty = 0; ty < tiles; ty++) {
      for (let tx = 0; tx < tiles; tx++) {
        let sum = 0;
        let sumSq = 0;
        let n = 0;
        for (let y = ty * tileH; y < Math.min((ty + 1) * tileH, height); y++) {
          for (let x = tx * tileW; x < Math.min((tx + 1) * tileW, width); x++) {
            const v = data[y * width + x];
            sum += v;
            sumSq += v * v;
            n++;
          }
        }
        if (n) variances.push(sumSq / n - (sum / n) ** 2);
      }
    }
    variances.sort((a, b) => a - b);
    const flattest = variances.slice(0, Math.max(1, Math.round(variances.length / 4)));
    return flattest.reduce((a, b) => a + b, 0) / flattest.length;
  }

  /**
   * Before/after measures that need no ground truth.
   *
   * Edge energy and background variance for both images plus their ratios, and
   * the Otsu foreground fraction, so an uploaded image can still be judged on
   * whether the halo was suppressed -- which is not the same claim as fidelity
   * to a clean target.
   */
  function noReferenceReport(before, after, width, height) {
    const beforeEdge = edgeEnergy(before, width, height);
    const afterEdge = edgeEnergy(after, width, height);
    const beforeBackground = backgroundVariance(before, width, height);
    const afterBackground = backgroundVariance(after, width, height);
    return {
      edgeEnergy: { before: beforeEdge, after: afterEdge, ratio: beforeEdge ? afterEdge / beforeEdge : NaN },
      backgroundVariance: {
        before: beforeBackground,
        after: afterBackground,
        ratio: beforeBackground ? afterBackground / beforeBackground : NaN,
      },
      foreground: { before: otsuMask(before).fraction, after: otsuMask(after).fraction },
    };
  }

  return {
    gaussianKernel,
    gaussianBlur,
    haloSuppressionDoG,
    applyClahe,
    claheCeiling,
    normalize01,
    runCvPipeline,
    resizeArea,
    psnr,
    ssim,
    toUint8,
    otsuThreshold,
    otsuMask,
    maskDice,
    otsuDice,
    edgeEnergy,
    backgroundVariance,
    noReferenceReport,
  };
});
