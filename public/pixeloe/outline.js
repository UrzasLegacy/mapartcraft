// Browser adaptation of KohakuBlueleaf/PixelOE's torch outline expansion.
// Apache-2.0, see LICENSE.txt. Configurable outline and sampling settings.
// Image boundaries are extended and fully transparent neighbours are ignored.
function lightness(data) {
  const linear = Float32Array.from({ length: 256 }, (_, i) => {
    const s = i / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  const result = new Float32Array(data.length / 4);
  for (let i = 0; i < result.length; i++) {
    const p = 4 * i;
    const y = 0.2126729 * linear[data[p]] + 0.7151522 * linear[data[p+1]] + 0.072175 * linear[data[p+2]];
    result[i] = y <= 0.008856 ? 9.033 * y : 1.16 * Math.cbrt(y) - 0.16;
  }
  return result;
}

// Average each overlapping patch statistic back onto its source pixels.
function localStats(image, luminance, size, median) {
  const { width, height, data } = image;
  const sums = new Float32Array(width * height);
  const counts = new Uint8Array(sums.length);
  const values = new Float32Array(size * size);
  const stride = size / (median ? 4 : 2);
  for (let cy = 0; cy <= height; cy += stride) {
    for (let cx = 0; cx <= width; cx += stride) {
      const left = Math.max(0, cx - size / 2), right = Math.min(width, cx + size / 2);
      const top = Math.max(0, cy - size / 2), bottom = Math.min(height, cy + size / 2);
      let n = 0, min = 1, max = 0;
      for (let y = top; y < bottom; y++) {
        for (let x = left; x < right; x++) {
          const i = y * width + x;
          if (!data[i * 4 + 3]) continue;
          const v = luminance[i];
          values[n++] = v;
          min = Math.min(min, v);
          max = Math.max(max, v);
        }
      }
      if (!n) continue;
      const stat = median ? values.subarray(0, n).sort()[(n - 1) >> 1] : min + max;
      for (let y = top; y < bottom; y++) {
        for (let x = left; x < right; x++) {
          const i = y * width + x;
          sums[i] += stat;
          counts[i]++;
        }
      }
    }
  }
  for (let i = 0; i < sums.length; i++) sums[i] /= counts[i] || 1;
  return sums;
}

function morphology(image, src, radius, dilate) {
  const { width, height, data } = image;
  const out = new Uint8ClampedArray(src);
  const extent = Math.max(1, Math.floor(radius - 0.5));
  for (let dy = -extent; dy <= extent; dy++) {
    for (let dx = -extent; dx <= extent; dx++) {
      if (!dx && !dy) continue;
      const x = Math.abs(dx), y = Math.abs(dy);
      const near = Math.min(Math.hypot(x - 0.5, y - 0.5), Math.hypot(x - 0.5, y), Math.hypot(x, y - 0.5));
      const far = Math.hypot(x + 0.5, y + 0.5);
      const penalty = 255 * (1 - Math.max(0, Math.min(1, (radius - near) / (far - near))));
      for (let row = Math.max(0,-dy); row < Math.min(height,height-dy); row++) {
        for (let col = Math.max(0,-dx); col < Math.min(width,width-dx); col++) {
          const p = 4 * (row * width + col), q = p + 4 * (dy * width + dx);
          if (!data[q+3] || !data[p+3]) continue;
          for (let c = 0; c < 3; c++) out[p+c] = dilate
            ? Math.max(out[p+c], src[q+c] - penalty)
            : Math.min(out[p+c], src[q+c] + penalty);
        }
      }
    }
  }
  return out;
}

function expand(image, { thickness, patch, brightnessWeight, contrastWeight }) {
  if (!thickness) return image.data;
  const radius = 0.5 + thickness / 2;
  const l = lightness(image.data);
  const weights = localStats(image, l, patch * 2, true);
  const extrema = localStats(image, l, patch, false);
  let lo = 1, hi = 0;
  for (let i = 0; i < weights.length; i++) {
    const m = weights[i];
    weights[i] = 1 / (1 + Math.exp(-((m - 0.5) * brightnessWeight + (2 * m - extrema[i]) * contrastWeight)));
    if (image.data[4*i+3]) {
      lo = Math.min(lo, weights[i]);
      hi = Math.max(hi, weights[i]);
    }
  }
  let pixels = morphology(image, image.data, radius, false);
  const bright = morphology(image, image.data, radius, true);
  for (let i = 0; i < weights.length; i++) {
    const w = (weights[i] - lo) / (hi - lo || 1);
    for (let c = 0; c < 3; c++) {
      const p = 4*i+c;
      pixels[p] = pixels[p]*w + bright[p]*(1-w);
    }
  }
  for (const dilate of [false, true, true, false]) pixels = morphology(image, pixels, Math.max(1, radius - 0.5), dilate);
  return pixels;
}

function nearestSample(src, sw, sh, width, height) {
  const out = new ImageData(width, height);
  for (let y = 0; y < height; y++) {
    const row = Math.floor((y + 0.5) * sh / height);
    for (let x = 0; x < width; x++) {
      const p = 4 * (row * sw + Math.floor((x + 0.5) * sw / width));
      out.data.set(src.subarray(p, p + 4), 4 * (y * width + x));
    }
  }
  return out;
}

self.onmessage = ({ data: { image, width, height, options } }) => {
  const pixels = expand(image, options);
  const sample = options.mode === "nearest" ? nearestSample : contrastSample;
  const result = sample(pixels, image.width, image.height, width, height);
  self.postMessage(result, [result.data.buffer]);
};

// PixelOE contrast downsampling in Lab. Fractional cell boundaries allow
// rectangular crops to fill the requested map grid directly.
function contrastSample(src, sw, sh, width, height) {
  const out = new ImageData(width, height);
  const capacity = Math.ceil(sw / width) * Math.ceil(sh / height);
  const L = new Float32Array(capacity), A = new Float32Array(capacity), B = new Float32Array(capacity);
  const linear = Float32Array.from({length:256}, (_, i) => i <= 10 ? i / 3294.6 : ((i/255+0.055)/1.055)**2.4);
  const f = v => v > 0.008856 ? Math.cbrt(v) : 7.787*v + 16/116;
  const inv = v => v**3 > 0.008856 ? v**3 : (v-16/116)/7.787;
  const srgb = v => 255*(v <= 0.0031308 ? 12.92*v : 1.055*v**(1/2.4)-0.055);
  for (let y = 0; y < height; y++) {
    const top = Math.floor(y*sh/height), bottom = Math.max(top+1, Math.floor((y+1)*sh/height));
    for (let x = 0; x < width; x++) {
      const left = Math.floor(x*sw/width), right = Math.max(left+1, Math.floor((x+1)*sw/width));
      let n = 0, sum = 0, alpha = 0;
      for (let row = top; row < bottom; row++) {
        for (let col = left; col < right; col++) {
          const p = 4*(row*sw+col);
          if (!src[p+3]) continue;
          alpha += src[p+3];
          const r = linear[src[p]], g = linear[src[p+1]], b = linear[src[p+2]];
          const fy = f(0.2126729*r+0.7151522*g+0.072175*b);
          L[n] = 116*fy-16;
          A[n] = 500*(f((0.4124564*r+0.3575761*g+0.1804375*b)/0.95047)-fy);
          B[n] = 200*(fy-f((0.0193339*r+0.119192*g+0.9503041*b)/1.08883));
          sum += L[n++];
        }
      }
      if (!n) continue;
      let light = L[n >> 1];
      const median = (n-1) >> 1;
      const sorted = L.subarray(0,n).sort(), m = sorted[median], low = sorted[0], high = sorted[n-1];
      if (m < sum/n && high-m > m-low) light = low;
      if (m > sum/n && high-m < m-low) light = high;
      const fy = (light+16)/116;
      const X = 0.95047*inv(fy+A.subarray(0,n).sort()[median]/500);
      const Y = inv(fy), Z = 1.08883*inv(fy-B.subarray(0,n).sort()[median]/200);
      const p = 4*(y*width+x);
      out.data[p] = srgb(3.2404542*X-1.5371385*Y-0.4985314*Z);
      out.data[p+1] = srgb(-0.969266*X+1.8760108*Y+0.041556*Z);
      out.data[p+2] = srgb(0.0556434*X-0.2040259*Y+1.0572252*Z);
      out.data[p+3] = alpha/((bottom-top)*(right-left));
    }
  }
  return out;
}
