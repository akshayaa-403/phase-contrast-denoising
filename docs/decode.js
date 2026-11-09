/**
 * Decoders for the image formats microscopes actually write, without a
 * library: PNG at 8 or 16 bits per sample, and baseline TIFF (uncompressed,
 * LZW or PackBits, with or without the horizontal-differencing predictor) at 8
 * or 16 bits, grey or RGB.
 *
 * Browsers decode a 16-bit PNG to 8 bits and refuse TIFF entirely, which would
 * throw away the precision a 16-bit micrograph carries. This keeps every bit:
 * the output is float32 on the 0-255 scale the pipeline uses, computed as
 * `value / 65535 * 255` for 16-bit data, exactly as `scripts/make_demo_assets.py`
 * and `main.py` scale it. Decoding the 8-bit samples here too, rather than
 * through a canvas, means no colour management or anti-fingerprinting noise can
 * nudge a pixel before the pipeline sees it.
 *
 * `inflate` is injected: DecompressionStream in a browser, zlib in Node.
 *
 * UMD-wrapped: `self.PhaseDecode` in a page, `require` in Node.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.PhaseDecode = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const f32 = Math.fround;

  class DecodeError extends Error {}

  /** Browser inflate for zlib-wrapped data (PNG IDAT, TIFF Deflate). */
  async function browserInflate(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  /** Collapse interleaved samples to one float32 0-255 grey channel. */
  function toGrey(samples, width, height, channels, maxValue, invert = false) {
    const pixels = new Float32Array(width * height);
    const top = f32(maxValue);
    const scale = (v) => (maxValue === 255 ? v : f32(f32(v / top) * 255));
    for (let i = 0; i < pixels.length; i++) {
      let v;
      if (channels >= 3) {
        const o = i * channels;
        // ITU-R BT.601 luma, the weights cv2.cvtColor(BGR2GRAY) uses.
        v = 0.299 * samples[o] + 0.587 * samples[o + 1] + 0.114 * samples[o + 2];
      } else {
        v = samples[i * channels];
      }
      if (invert) v = maxValue - v;
      pixels[i] = scale(v);
    }
    return pixels;
  }

  // ---------------------------------------------------------------- PNG
  function isPng(bytes) {
    const sig = [137, 80, 78, 71, 13, 10, 26, 10];
    return bytes.length > 33 && sig.every((b, i) => bytes[i] === b);
  }

  /** Bit depth from the IHDR chunk, without decoding (0 if not a PNG). */
  function pngBitDepth(bytes) {
    return isPng(bytes) ? bytes[24] : 0;
  }

  async function decodePng(bytes, inflate = browserInflate) {
    if (!isPng(bytes)) throw new DecodeError('Not a PNG file.');
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let offset = 8;
    let width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
    let palette = null;
    const idat = [];
    while (offset + 8 <= bytes.length) {
      const length = view.getUint32(offset);
      const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
      const data = bytes.subarray(offset + 8, offset + 8 + length);
      if (type === 'IHDR') {
        width = view.getUint32(offset + 8);
        height = view.getUint32(offset + 12);
        bitDepth = data[8];
        colorType = data[9];
        interlace = data[12];
      } else if (type === 'PLTE') {
        palette = data;
      } else if (type === 'IDAT') {
        idat.push(data);
      } else if (type === 'IEND') {
        break;
      }
      offset += 12 + length;
    }
    if (!width || !height) throw new DecodeError('PNG has no image header.');
    if (interlace) throw new DecodeError('Interlaced PNG is not supported.');
    if (bitDepth !== 8 && bitDepth !== 16) throw new DecodeError(`PNG at ${bitDepth} bits per sample is not supported.`);
    const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[colorType];
    if (!channels) throw new DecodeError('Unknown PNG colour type.');

    let total = 0;
    for (const part of idat) total += part.length;
    const compressed = new Uint8Array(total);
    let at = 0;
    for (const part of idat) {
      compressed.set(part, at);
      at += part.length;
    }
    const raw = await inflate(compressed);

    const bpp = Math.max(1, (channels * bitDepth) / 8);
    const stride = width * bpp;
    if (raw.length < height * (stride + 1)) throw new DecodeError('PNG data is truncated.');
    const out = new Uint8Array(height * stride);
    for (let y = 0; y < height; y++) {
      const filter = raw[y * (stride + 1)];
      const src = y * (stride + 1) + 1;
      const dst = y * stride;
      const prev = dst - stride;
      for (let x = 0; x < stride; x++) {
        const a = x >= bpp ? out[dst + x - bpp] : 0;
        const b = y > 0 ? out[prev + x] : 0;
        const c = x >= bpp && y > 0 ? out[prev + x - bpp] : 0;
        let v = raw[src + x];
        switch (filter) {
          case 0: break;
          case 1: v += a; break;
          case 2: v += b; break;
          case 3: v += (a + b) >> 1; break;
          case 4: {
            const p = a + b - c;
            const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
            v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
            break;
          }
          default: throw new DecodeError('Corrupt PNG filter byte.');
        }
        out[dst + x] = v;
      }
    }

    let samples;
    let sampleChannels = channels;
    if (colorType === 3) {
      if (!palette) throw new DecodeError('Palette PNG without a palette.');
      samples = new Uint8Array(width * height * 3);
      for (let i = 0; i < width * height; i++) {
        const p = out[i] * 3;
        samples[i * 3] = palette[p];
        samples[i * 3 + 1] = palette[p + 1];
        samples[i * 3 + 2] = palette[p + 2];
      }
      sampleChannels = 3;
    } else if (bitDepth === 16) {
      samples = new Uint16Array(width * height * channels);
      for (let i = 0; i < samples.length; i++) samples[i] = (out[i * 2] << 8) | out[i * 2 + 1];
    } else {
      samples = out;
    }
    // Grey+alpha and RGBA: alpha is ignored, as cv2.imread does.
    const maxValue = bitDepth === 16 && colorType !== 3 ? 65535 : 255;
    const pixels = toGrey(samples, width, height, sampleChannels, maxValue);
    return { width, height, pixels, bitDepth, format: 'PNG' };
  }

  // --------------------------------------------------------------- TIFF
  function isTiff(bytes) {
    return bytes.length > 8 &&
      ((bytes[0] === 0x49 && bytes[1] === 0x49 && bytes[2] === 42 && bytes[3] === 0) ||
       (bytes[0] === 0x4d && bytes[1] === 0x4d && bytes[2] === 0 && bytes[3] === 42));
  }

  /** TIFF LZW (MSB-first codes, "early change"), as written by libtiff. */
  function lzwDecode(input, expected) {
    const out = new Uint8Array(expected);
    let outPos = 0;
    // The table holds strings as (prefix code, last byte, length).
    const prefix = new Int32Array(4096);
    const suffix = new Uint8Array(4096);
    const lengths = new Int32Array(4096);
    for (let i = 0; i < 256; i++) {
      prefix[i] = -1;
      suffix[i] = i;
      lengths[i] = 1;
    }
    let next = 258;
    let width = 9;
    let bitPos = 0;
    let previous = -1;
    const totalBits = input.length * 8;

    const readCode = () => {
      if (bitPos + width > totalBits) return 257;
      let code = 0;
      for (let i = 0; i < width; i++) {
        const byte = input[(bitPos + i) >> 3];
        code = (code << 1) | ((byte >> (7 - ((bitPos + i) & 7))) & 1);
      }
      bitPos += width;
      return code;
    };

    const emit = (code) => {
      const len = lengths[code];
      let pos = outPos + len - 1;
      let c = code;
      while (c >= 0) {
        if (pos < expected) out[pos] = suffix[c];
        pos--;
        c = prefix[c];
      }
      outPos += len;
    };
    const firstByte = (code) => {
      let c = code;
      while (prefix[c] >= 0) c = prefix[c];
      return suffix[c];
    };

    while (outPos < expected) {
      const code = readCode();
      if (code === 257) break;
      if (code === 256) {
        next = 258;
        width = 9;
        previous = -1;
        continue;
      }
      if (previous < 0) {
        if (code > 255) throw new DecodeError('Corrupt LZW data.');
        emit(code);
        previous = code;
        continue;
      }
      if (code < next) {
        emit(code);
        if (next < 4096) {
          prefix[next] = previous;
          suffix[next] = firstByte(code);
          lengths[next] = lengths[previous] + 1;
          next++;
        }
      } else if (code === next) {
        const first = firstByte(previous);
        prefix[next] = previous;
        suffix[next] = first;
        lengths[next] = lengths[previous] + 1;
        next++;
        emit(code);
      } else {
        throw new DecodeError('Corrupt LZW data.');
      }
      previous = code;
      // Early change: widen one code before the table fills the current width.
      if (next + 1 >= 1 << width && width < 12) width++;
    }
    return out;
  }

  function packBitsDecode(input, expected) {
    const out = new Uint8Array(expected);
    let i = 0, o = 0;
    while (i < input.length && o < expected) {
      const n = (input[i++] << 24) >> 24;
      if (n >= 0) {
        for (let k = 0; k <= n && o < expected; k++) out[o++] = input[i++];
      } else if (n !== -128) {
        const v = input[i++];
        for (let k = 0; k < 1 - n && o < expected; k++) out[o++] = v;
      }
    }
    return out;
  }

  async function decodeTiff(bytes, inflate = browserInflate) {
    if (!isTiff(bytes)) throw new DecodeError('Not a TIFF file.');
    const little = bytes[0] === 0x49;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const u16 = (o) => view.getUint16(o, little);
    const u32 = (o) => view.getUint32(o, little);

    const ifd = u32(4);
    const count = u16(ifd);
    const tags = {};
    for (let e = 0; e < count; e++) {
      const entry = ifd + 2 + e * 12;
      const tag = u16(entry);
      const type = u16(entry + 2);
      const n = u32(entry + 4);
      const size = { 1: 1, 2: 1, 3: 2, 4: 4, 6: 1, 7: 1, 8: 2, 9: 4 }[type] || 0;
      if (!size) continue;
      const inline = size * n <= 4;
      const base = inline ? entry + 8 : u32(entry + 8);
      const values = [];
      for (let k = 0; k < n; k++) {
        const o = base + k * size;
        values.push(size === 1 ? view.getUint8(o) : size === 2 ? u16(o) : u32(o));
      }
      tags[tag] = values;
    }

    const get = (tag, fallback) => (tags[tag] ? tags[tag] : fallback);
    const width = get(256, [0])[0];
    const height = get(257, [0])[0];
    const samplesPerPixel = get(277, [1])[0];
    const bits = get(258, [1])[0];
    const compression = get(259, [1])[0];
    const photometric = get(262, [1])[0];
    const predictor = get(317, [1])[0];
    const planar = get(284, [1])[0];
    const sampleFormat = get(339, [1])[0];

    if (!width || !height) throw new DecodeError('TIFF has no image size.');
    if (tags[322]) throw new DecodeError('Tiled TIFF is not supported; save it with strips.');
    if (bits !== 8 && bits !== 16) throw new DecodeError(`TIFF at ${bits} bits per sample is not supported.`);
    if (sampleFormat !== 1) throw new DecodeError('Only unsigned-integer TIFF samples are supported.');
    if (samplesPerPixel > 1 && planar !== 1) throw new DecodeError('Planar-separated TIFF is not supported.');
    if (![1, 5, 8, 32946, 32773].includes(compression)) {
      throw new DecodeError(`TIFF compression ${compression} is not supported (use none, LZW or Deflate).`);
    }

    const offsets = get(273, null);
    const byteCounts = get(279, null);
    if (!offsets || !byteCounts) throw new DecodeError('TIFF has no strips.');
    const rowsPerStrip = Math.min(get(278, [height])[0], height);
    const bytesPerSample = bits / 8;
    const rowBytes = width * samplesPerPixel * bytesPerSample;
    const data = new Uint8Array(rowBytes * height);

    for (let s = 0; s < offsets.length; s++) {
      const rows = Math.min(rowsPerStrip, height - s * rowsPerStrip);
      if (rows <= 0) break;
      const expected = rows * rowBytes;
      const chunk = bytes.subarray(offsets[s], offsets[s] + byteCounts[s]);
      let strip;
      if (compression === 1) strip = chunk;
      else if (compression === 5) strip = lzwDecode(chunk, expected);
      else if (compression === 32773) strip = packBitsDecode(chunk, expected);
      else strip = await inflate(chunk);
      data.set(strip.subarray(0, expected), s * rowsPerStrip * rowBytes);
    }

    const total = width * height * samplesPerPixel;
    let samples;
    if (bits === 16) {
      samples = new Uint16Array(total);
      const dv = new DataView(data.buffer);
      for (let i = 0; i < total; i++) samples[i] = dv.getUint16(i * 2, little);
    } else {
      samples = data;
    }
    if (predictor === 2) {
      const mask = bits === 16 ? 0xffff : 0xff;
      for (let y = 0; y < height; y++) {
        const row = y * width * samplesPerPixel;
        for (let x = samplesPerPixel; x < width * samplesPerPixel; x++) {
          samples[row + x] = (samples[row + x] + samples[row + x - samplesPerPixel]) & mask;
        }
      }
    }

    const pixels = toGrey(samples, width, height, samplesPerPixel, bits === 16 ? 65535 : 255, photometric === 0);
    return { width, height, pixels, bitDepth: bits, format: 'TIFF' };
  }

  /**
   * Decode PNG or TIFF bytes. Returns null for anything else, so the caller
   * can fall back to the browser's own decoder (JPEG, WebP, …).
   */
  async function decode(bytes, inflate = browserInflate) {
    if (isPng(bytes)) return decodePng(bytes, inflate);
    if (isTiff(bytes)) return decodeTiff(bytes, inflate);
    return null;
  }

  return { decode, decodePng, decodeTiff, pngBitDepth, isPng, isTiff, DecodeError };
});
