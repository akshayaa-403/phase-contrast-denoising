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
 *   - `createCLAHE(clipLimit, tileGridSize)` follows the algorithm in OpenCV's
 *     clahe.cpp: per-tile histograms over a 65536-bin range, the clip limit
 *     rescaled by tile area, excess redistributed, then bilinear interpolation
 *     between neighbouring tile LUTs.
 *
 * Agreement with the Python path is measured, not assumed -- see
 * `scripts/verify_web_pipeline.py`.
 *
 * Images are handled as Float32Array in the 0-255 range (what the Python code
 * gets from `cv2.imread`), with a final `normalize01` to [0, 1].
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.PhaseCV = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const HIST_SIZE = 65536;

  /** OpenCV's kernel size for GaussianBlur with ksize=(0,0) on a float image. */
  function kernelSizeFor(sigma) {
    return (Math.round(sigma * 4 * 2 + 1) | 1);
  }

  /** Normalised 1-D Gaussian, matching cv2.getGaussianKernel(ksize, sigma). */
  function gaussianKernel(sigma) {
    const size = kernelSizeFor(sigma);
    const center = (size - 1) / 2;
    const scale = -0.5 / (sigma * sigma);
    const kernel = new Float64Array(size);
    let sum = 0;
    for (let i = 0; i < size; i++) {
      const x = i - center;
      kernel[i] = Math.exp(scale * x * x);
      sum += kernel[i];
    }
    for (let i = 0; i < size; i++) kernel[i] /= sum;
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

  /** Separable Gaussian blur of a single-channel float image. */
  function gaussianBlur(src, width, height, sigma) {
    if (sigma <= 0) return Float32Array.from(src);
    const kernel = gaussianKernel(sigma);
    const radius = (kernel.length - 1) / 2;
    const horizontal = new Float32Array(width * height);
    const output = new Float32Array(width * height);

    for (let y = 0; y < height; y++) {
      const row = y * width;
      for (let x = 0; x < width; x++) {
        let acc = 0;
        for (let k = -radius; k <= radius; k++) {
          acc += kernel[k + radius] * src[row + reflect101(x + k, width)];
        }
        horizontal[row + x] = acc;
      }
    }
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let acc = 0;
        for (let k = -radius; k <= radius; k++) {
          acc += kernel[k + radius] * horizontal[reflect101(y + k, height) * width + x];
        }
        output[y * width + x] = acc;
      }
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
    if (sigma2 <= sigma1) throw new Error(`sigma2 (${sigma2}) must be greater than sigma1 (${sigma1})`);
    const small = gaussianBlur(src, width, height, sigma1);
    const large = gaussianBlur(src, width, height, sigma2);
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) out[i] = src[i] - alpha * (small[i] - large[i]);
    return out;
  }

  /** Min and max of a Float32Array. */
  function extent(data) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < lo) lo = data[i];
      if (data[i] > hi) hi = data[i];
    }
    return [lo, hi];
  }

  /** Min-max scale to [0, 1]; a constant image becomes all zeros. */
  function normalize01(data) {
    const [lo, hi] = extent(data);
    const range = hi - lo;
    const out = new Float32Array(data.length);
    if (range < 1e-8) return out;
    for (let i = 0; i < data.length; i++) out[i] = (data[i] - lo) / range;
    return out;
  }

  /**
   * CLAHE on a float image, via the same uint16 quantisation the Python code
   * uses (`cv2.createCLAHE` needs an integer image, and uint16 keeps far more
   * precision than uint8 through the DoG stage's float output).
   */
  function applyClahe(src, width, height, clipLimit, tilesX, tilesY) {
    const [lo, hi] = extent(src);
    const range = hi - lo;
    if (range < 1e-8) return Float32Array.from(src);

    const quantized = new Uint16Array(src.length);
    for (let i = 0; i < src.length; i++) {
      quantized[i] = Math.round(((src[i] - lo) / range) * 65535);
    }

    const equalized = claheUint16(quantized, width, height, clipLimit, tilesX, tilesY);

    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) out[i] = (equalized[i] / 65535) * range + lo;
    return out;
  }

  /** The tile-histogram core, operating on 16-bit data. */
  function claheUint16(src, width, height, clipLimit, tilesX, tilesY) {
    // OpenCV pads the image so the tile grid divides it evenly.
    const tileWidth = Math.ceil(width / tilesX);
    const tileHeight = Math.ceil(height / tilesY);
    const tileArea = tileWidth * tileHeight;
    const lutScale = (HIST_SIZE - 1) / tileArea;

    let clipCount = 0;
    if (clipLimit > 0) {
      clipCount = Math.max(Math.round((clipLimit * tileArea) / HIST_SIZE), 1);
    }

    // One LUT per tile, laid out tile-major.
    const luts = new Uint16Array(tilesX * tilesY * HIST_SIZE);
    const hist = new Int32Array(HIST_SIZE);

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        hist.fill(0);
        for (let y = 0; y < tileHeight; y++) {
          // Reflect rather than clamp, so padded tiles are not biased by a
          // repeated edge row.
          const sy = reflect101(ty * tileHeight + y, height);
          const row = sy * width;
          for (let x = 0; x < tileWidth; x++) {
            hist[src[row + reflect101(tx * tileWidth + x, width)]]++;
          }
        }

        if (clipCount > 0) {
          let clipped = 0;
          for (let i = 0; i < HIST_SIZE; i++) {
            if (hist[i] > clipCount) {
              clipped += hist[i] - clipCount;
              hist[i] = clipCount;
            }
          }
          // Spread the clipped mass back uniformly, then sprinkle the remainder.
          const batch = Math.floor(clipped / HIST_SIZE);
          let residual = clipped - batch * HIST_SIZE;
          if (batch > 0) for (let i = 0; i < HIST_SIZE; i++) hist[i] += batch;
          if (residual > 0) {
            const step = Math.max(Math.floor(HIST_SIZE / residual), 1);
            for (let i = 0; i < HIST_SIZE && residual > 0; i += step) {
              hist[i]++;
              residual--;
            }
          }
        }

        const base = (ty * tilesX + tx) * HIST_SIZE;
        let sum = 0;
        for (let i = 0; i < HIST_SIZE; i++) {
          sum += hist[i];
          const value = Math.round(sum * lutScale);
          luts[base + i] = value > 65535 ? 65535 : value;
        }
      }
    }

    // Bilinear blend between the four nearest tile LUTs, so tile boundaries do
    // not show up as seams.
    const out = new Uint16Array(src.length);
    for (let y = 0; y < height; y++) {
      const fy = y / tileHeight - 0.5;
      let y0 = Math.floor(fy);
      const wy = fy - y0;
      const ya = Math.min(Math.max(y0, 0), tilesY - 1);
      const yb = Math.min(Math.max(y0 + 1, 0), tilesY - 1);

      for (let x = 0; x < width; x++) {
        const fx = x / tileWidth - 0.5;
        const x0 = Math.floor(fx);
        const wx = fx - x0;
        const xa = Math.min(Math.max(x0, 0), tilesX - 1);
        const xb = Math.min(Math.max(x0 + 1, 0), tilesX - 1);

        const value = src[y * width + x];
        const tl = luts[(ya * tilesX + xa) * HIST_SIZE + value];
        const tr = luts[(ya * tilesX + xb) * HIST_SIZE + value];
        const bl = luts[(yb * tilesX + xa) * HIST_SIZE + value];
        const br = luts[(yb * tilesX + xb) * HIST_SIZE + value];

        const top = tl + (tr - tl) * wx;
        const bottom = bl + (br - bl) * wx;
        out[y * width + x] = Math.round(top + (bottom - top) * wy);
      }
    }
    return out;
  }

  /**
   * The full CV-only pipeline, mirroring `CleanupPipeline._cv_baseline`.
   *
   * @param {Float32Array} src 0-255 grayscale
   * @param {object} options sigma1, sigma2, alpha, clipLimit, tilesX, tilesY
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
    } = options || {};

    const dog = haloSuppressionDoG(src, width, height, sigma1, sigma2, alpha);
    const clahe = applyClahe(dog, width, height, clipLimit, tilesX, tilesY);
    return { dog, clahe, result: normalize01(clahe) };
  }

  // ----------------------------------------------------------------- metrics
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

  function clamp01(v) {
    return v < 0 ? 0 : v > 1 ? 1 : v;
  }

  /**
   * Mean SSIM over 7x7 windows, the same window skimage uses by default.
   *
   * skimage applies a uniform filter and crops the border by half a window;
   * this does the same, so the numbers line up with the report's.
   */
  function ssim(a, b, width, height, windowSize = 7) {
    const C1 = 0.01 * 0.01;
    const C2 = 0.03 * 0.03;
    const radius = Math.floor(windowSize / 2);
    const n = windowSize * windowSize;
    // skimage's default uses the sample covariance (N-1 normalisation).
    const norm = n / (n - 1);

    let total = 0;
    let count = 0;
    for (let y = radius; y < height - radius; y++) {
      for (let x = radius; x < width - radius; x++) {
        let sumA = 0, sumB = 0, sumAA = 0, sumBB = 0, sumAB = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          const row = (y + dy) * width;
          for (let dx = -radius; dx <= radius; dx++) {
            const va = clamp01(a[row + x + dx]);
            const vb = clamp01(b[row + x + dx]);
            sumA += va; sumB += vb;
            sumAA += va * va; sumBB += vb * vb; sumAB += va * vb;
          }
        }
        const meanA = sumA / n;
        const meanB = sumB / n;
        const varA = (sumAA / n - meanA * meanA) * norm;
        const varB = (sumBB / n - meanB * meanB) * norm;
        const covAB = (sumAB / n - meanA * meanB) * norm;

        total +=
          ((2 * meanA * meanB + C1) * (2 * covAB + C2)) /
          ((meanA * meanA + meanB * meanB + C1) * (varA + varB + C2));
        count++;
      }
    }
    return count ? total / count : NaN;
  }

  return {
    gaussianKernel,
    gaussianBlur,
    haloSuppressionDoG,
    applyClahe,
    normalize01,
    runCvPipeline,
    psnr,
    ssim,
  };
});
