/**
 * Runs the classical pipeline off the main thread, so dragging a slider never
 * blocks the page.
 *
 * Protocol (all buffers are transferred, not copied):
 *
 *   -> { type: 'image', imageId, width, height, pixels, truth?, hybrid? }
 *        pixels: Float32Array 0-255; truth, hybrid: Float32Array [0, 1].
 *        Replies { type: 'static', imageId, input, hybrid } with PSNR/SSIM/Dice
 *        of the stages that do not depend on the sliders.
 *   -> { type: 'run', job, imageId, options, preview }
 *        preview: compute on a copy no larger than 512 px (for large images
 *        while a slider moves); the full image is computed on release.
 *        Replies { type: 'progress', job, fraction } for long jobs, then
 *        { type: 'result', job, imageId, preview, width, height, dog, result,
 *          removed, removedMax, mask, metrics | noRef, elapsed }.
 *
 * The main thread keeps at most one job in flight and drops any result whose
 * parameters have since changed, so stale work is never shown.
 */
/* global PhaseCV */
importScripts('cv.js');

const PREVIEW_SIZE = 512;
let image = null;

function previewOf(img) {
  if (!img.preview) {
    const scale = PREVIEW_SIZE / Math.max(img.width, img.height);
    const width = Math.max(1, Math.round(img.width * scale));
    const height = Math.max(1, Math.round(img.height * scale));
    img.preview = {
      width,
      height,
      pixels: PhaseCV.resizeArea(img.pixels, img.width, img.height, width, height),
    };
  }
  return img.preview;
}

function scores(stage, truth, width, height) {
  return {
    psnr: PhaseCV.psnr(stage, truth),
    ssim: PhaseCV.ssim(stage, truth, width, height),
    dice: PhaseCV.otsuDice(stage, truth),
  };
}

self.onmessage = (event) => {
  const message = event.data;

  if (message.type === 'image') {
    image = {
      id: message.imageId,
      width: message.width,
      height: message.height,
      pixels: message.pixels,
      truth: message.truth || null,
      inputUnit: PhaseCV.normalize01(message.pixels),
      preview: null,
    };
    const reply = { type: 'static', imageId: image.id, input: null, hybrid: null };
    if (image.truth) {
      reply.input = scores(image.inputUnit, image.truth, image.width, image.height);
      if (message.hybrid) reply.hybrid = scores(message.hybrid, image.truth, image.width, image.height);
    }
    self.postMessage(reply);
    return;
  }

  if (message.type === 'run') {
    if (!image || image.id !== message.imageId) return;
    const started = performance.now();
    const usePreview = message.preview && Math.max(image.width, image.height) > PREVIEW_SIZE;
    const src = usePreview ? previewOf(image) : image;
    const { width, height, pixels } = src;
    const progress = (fraction) => self.postMessage({ type: 'progress', job: message.job, fraction });

    const o = message.options;
    progress(0.05);
    const dog = PhaseCV.haloSuppressionDoG(pixels, width, height, o.sigma1, o.sigma2, o.alpha);
    progress(0.45);
    const clahe = PhaseCV.applyClahe(dog, width, height, o.clipLimit, o.tilesX, o.tilesY, o.bits);
    const result = PhaseCV.normalize01(clahe);
    progress(0.75);

    // What the DoG step took away, in grey levels of the input.
    const removed = new Float32Array(pixels.length);
    let removedMax = 0;
    for (let i = 0; i < pixels.length; i++) {
      const r = pixels[i] - dog[i];
      removed[i] = r;
      const a = r < 0 ? -r : r;
      if (a > removedMax) removedMax = a;
    }

    const reply = {
      type: 'result',
      job: message.job,
      imageId: image.id,
      preview: usePreview,
      width,
      height,
      dog: PhaseCV.normalize01(dog),
      result,
      removed,
      removedMax,
      mask: null,
      metrics: null,
      noRef: null,
    };

    if (!usePreview) {
      const segmented = PhaseCV.otsuMask(result);
      reply.mask = segmented.mask;
      if (image.truth) {
        reply.metrics = scores(result, image.truth, width, height);
      } else {
        reply.noRef = PhaseCV.noReferenceReport(image.inputUnit, result, width, height);
      }
    }
    reply.elapsed = performance.now() - started;

    const transfer = [reply.dog.buffer, reply.result.buffer, reply.removed.buffer];
    if (reply.mask) transfer.push(reply.mask.buffer);
    self.postMessage(reply, transfer);
  }
};
