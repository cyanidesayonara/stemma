/* The mixer demo's audio engine, an AudioWorklet: it mixes a song's stems
   with their gains, loops and seeks, and changes speed and pitch live the
   way the app does (src/stretch.py): a phase vocoder, one frame at a time,
   then a resampler when the pitch moves. Speed alone changes the tempo,
   pitch alone the key.

   The phase vocoder is the app's (librosa's): a periodic Hann window,
   2048-point frames, a 512-sample hop, magnitudes interpolated between two
   analysis frames and the phase carried as unit phasors. The pitch
   resampler here is a variable-rate cubic interpolator rather than the
   app's soxr stream, so a pitch change needs no crossfade. At 1.0x and the
   original pitch the stretcher is skipped, as the app skips it.

   Messages in:  load, play, pause, seek, gains, loop, params, click
   Messages out: pos {frame}, wrap */

const N_FFT = 2048;
const HOP = 512;
const PAD = N_FFT / 2;
const BINS = N_FFT / 2 + 1;
const TINY = 1e-10;
const RING = 1 << 15;          // input ring, in samples per channel
const FADE = 128;              // samples to fade over on a jump
const EDGE = 64;               // samples to fade at a loop wrap

const WINDOW = new Float32Array(N_FFT);
for (let i = 0; i < N_FFT; i++) {
  WINDOW[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT);
}

// ---- an in-place radix-2 complex FFT --------------------------------------
const LEVELS = Math.log2(N_FFT);
const REV = new Uint32Array(N_FFT);
for (let i = 0; i < N_FFT; i++) {
  let r = 0;
  for (let b = 0; b < LEVELS; b++) { r = (r << 1) | ((i >>> b) & 1); }
  REV[i] = r;
}
const COS = new Float32Array(N_FFT / 2);
const SIN = new Float32Array(N_FFT / 2);
for (let i = 0; i < N_FFT / 2; i++) {
  COS[i] = Math.cos((2 * Math.PI * i) / N_FFT);
  SIN[i] = Math.sin((2 * Math.PI * i) / N_FFT);
}

function fft(re, im, inverse) {
  for (let i = 0; i < N_FFT; i++) {
    const j = REV[i];
    if (j > i) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  const sign = inverse ? 1 : -1;
  for (let size = 2; size <= N_FFT; size <<= 1) {
    const half = size >>> 1;
    const step = N_FFT / size;
    for (let start = 0; start < N_FFT; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = COS[k * step];
        const wi = sign * SIN[k * step];
        const a = start + k;
        const b = a + half;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < N_FFT; i++) { re[i] /= N_FFT; im[i] /= N_FFT; }
  }
}

// ---- the stretcher: one channel pair, streaming ---------------------------
class Stretcher {
  constructor(channels) {
    this.ch = channels;
    this.re = new Float32Array(N_FFT);
    this.im = new Float32Array(N_FFT);
    this.reset(0);
  }

  reset(position) {
    const ch = this.ch;
    this.ring = [];
    for (let c = 0; c < ch; c++) { this.ring.push(new Float32Array(RING)); }
    this.inputEnd = PAD;       // padded index after the last input sample
    this.runs = [];            // [padded start, source frame, count]
    this.t = 0;                // analysis position, in frames
    this.cache = new Map();    // frame index -> {mag, ur, ui} per channel
    this.pool = this.pool || [];
    this.phase = null;         // per channel {r, i}
    this.advance = null;
    this.ola = [];
    for (let c = 0; c < ch; c++) { this.ola.push(new Float32Array(N_FFT)); }
    this.norm = new Float32Array(N_FFT);
    this.recentT = [];
    this.drop = PAD;
    // stretched samples waiting for the resampler, and where it reads
    this.fifo = [];
    for (let c = 0; c < ch; c++) { this.fifo.push([]); }
    this.fifoStart = 0;        // stretched index of fifo[c][0]
    this.read = 1;             // resampler position, stretched index
    this.marks = [];           // [stretched index, source frame]
    this.position = position;
  }

  // Pull source audio until the padded input reaches `need`.
  fill(need, pull) {
    while (this.inputEnd < need) {
      const got = pull(2048);
      const n = got.count;
      this.runs.push([this.inputEnd, got.frame, n]);
      if (this.runs.length > 32) { this.runs.shift(); }
      for (let c = 0; c < this.ch; c++) {
        const ring = this.ring[c];
        const src = got.data[c];
        for (let i = 0; i < n; i++) {
          ring[(this.inputEnd + i) & (RING - 1)] = src[i];
        }
      }
      this.inputEnd += n;
    }
  }

  spectrum(j) {
    const hit = this.cache.get(j);
    if (hit) { return hit; }
    // reuse the arrays of a frame the analysis has passed: no allocation
    // on the audio thread once it is running
    const out = this.pool.pop() || [];
    const start = j * HOP;
    for (let c = 0; c < this.ch; c++) {
      // a fresh ring is zeros, so padded indices below PAD (the centring
      // padding) read as silence
      const ring = this.ring[c];
      const re = this.re;
      const im = this.im;
      for (let i = 0; i < N_FFT; i++) {
        re[i] = ring[(start + i) & (RING - 1)] * WINDOW[i];
        im[i] = 0;
      }
      fft(re, im, false);
      if (!out[c]) {
        out[c] = { mag: new Float32Array(BINS), ur: new Float32Array(BINS),
          ui: new Float32Array(BINS) };
      }
      const mag = out[c].mag;
      const ur = out[c].ur;
      const ui = out[c].ui;
      for (let b = 0; b < BINS; b++) {
        const m = Math.hypot(re[b], im[b]);
        mag[b] = m;
        if (m > TINY) { ur[b] = re[b] / m; ui[b] = im[b] / m; }
        else { ur[b] = 1; ui[b] = 0; }
      }
    }
    this.cache.set(j, out);
    return out;
  }

  sourceFrame(t) {
    const padded = t * HOP + PAD;
    for (let r = this.runs.length - 1; r >= 0; r--) {
      const run = this.runs[r];
      if (padded >= run[0]) {
        return run[1] + Math.min(padded - run[0], run[2]);
      }
    }
    return this.position;
  }

  synthesise(rate, pull) {
    const t = this.t;
    const i0 = Math.floor(t);
    const i1 = i0 + 1;
    this.fill(i1 * HOP + N_FFT, pull);
    const s0 = this.spectrum(i0);
    const s1 = this.spectrum(i1);
    const a = t - i0;
    const first = this.phase === null;
    if (first) {
      this.phase = s0.map((s) => ({ r: s.ur.slice(), i: s.ui.slice() }));
      this.advance = s0.map(() => ({ r: new Float32Array(BINS),
        i: new Float32Array(BINS) }));
    }
    for (let c = 0; c < this.ch; c++) {
      const p = this.phase[c];
      const adv = this.advance[c];
      if (!first) {
        for (let b = 0; b < BINS; b++) {
          const r = p.r[b] * adv.r[b] - p.i[b] * adv.i[b];
          const i = p.r[b] * adv.i[b] + p.i[b] * adv.r[b];
          const m = Math.max(Math.hypot(r, i), TINY);
          p.r[b] = r / m; p.i[b] = i / m;
        }
      }
      const u0 = s0[c];
      const u1 = s1[c];
      for (let b = 0; b < BINS; b++) {
        // advance = unit1 * conj(unit0)
        adv.r[b] = u1.ur[b] * u0.ur[b] + u1.ui[b] * u0.ui[b];
        adv.i[b] = u1.ui[b] * u0.ur[b] - u1.ur[b] * u0.ui[b];
      }
      const re = this.re;
      const im = this.im;
      for (let b = 0; b < BINS; b++) {
        const m = u0.mag[b] + a * (u1.mag[b] - u0.mag[b]);
        re[b] = m * p.r[b];
        im[b] = m * p.i[b];
      }
      for (let b = 1; b < BINS - 1; b++) {
        re[N_FFT - b] = re[b];
        im[N_FFT - b] = -im[b];
      }
      im[0] = 0; im[BINS - 1] = 0;
      fft(re, im, true);
      const ola = this.ola[c];
      for (let i = 0; i < N_FFT; i++) { ola[i] += re[i] * WINDOW[i]; }
    }
    for (let i = 0; i < N_FFT; i++) { this.norm[i] += WINDOW[i] * WINDOW[i]; }
    this.recentT.push(t);
    if (this.recentT.length > 3) { this.recentT.shift(); }

    // the first hop takes no more frames: normalise and emit it
    let skip = Math.min(this.drop, HOP);
    this.drop -= skip;
    if (skip < HOP) {
      const index = this.fifoStart + this.fifo[0].length;
      this.marks.push([index, this.sourceFrame(this.recentT[0])]);
      if (this.marks.length > 64) { this.marks.shift(); }
    }
    for (let c = 0; c < this.ch; c++) {
      const ola = this.ola[c];
      const q = this.fifo[c];
      for (let i = skip; i < HOP; i++) {
        q.push(ola[i] / Math.max(this.norm[i], TINY));
      }
      ola.copyWithin(0, HOP);
      ola.fill(0, N_FFT - HOP);
    }
    this.norm.copyWithin(0, HOP);
    this.norm.fill(0, N_FFT - HOP);

    this.t = t + rate;
    const floorNow = Math.floor(this.t);
    for (const [key, spectra] of this.cache) {
      if (key < floorNow) {
        this.cache.delete(key);
        this.pool.push(spectra);
      }
    }
  }

  // Write `count` output samples into `out` (per channel), stretching by
  // `rate` (analysis frames per output frame) and resampling by `pitch`.
  render(out, offset, count, rate, pitch, pull) {
    const q = this.fifo;
    for (let n = 0; n < count; n++) {
      // the cubic needs read-1 .. read+2
      while (Math.floor(this.read) + 2 >= this.fifoStart + q[0].length) {
        this.synthesise(rate, pull);
      }
      const base = Math.floor(this.read);
      const f = this.read - base;
      const k = base - this.fifoStart;
      for (let c = 0; c < this.ch; c++) {
        const y0 = q[c][k - 1] || 0;
        const y1 = q[c][k];
        const y2 = q[c][k + 1];
        const y3 = q[c][k + 2];
        // Catmull-Rom (Hermite) interpolation
        const c1 = 0.5 * (y2 - y0);
        const c2 = y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3;
        const c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
        out[c][offset + n] = ((c3 * f + c2) * f + c1) * f + y1;
      }
      this.read += pitch;
    }
    // forget what the resampler has passed
    const used = Math.floor(this.read) - 2 - this.fifoStart;
    if (used > 1024) {
      for (let c = 0; c < this.ch; c++) { q[c].splice(0, used); }
      this.fifoStart += used;
    }
    // what the listener hears now, in source frames
    while (this.marks.length > 1 && this.marks[1][0] <= this.read) {
      this.marks.shift();
    }
    if (this.marks.length && this.marks[0][0] <= this.read) {
      this.position = this.marks[0][1] + (this.read - this.marks[0][0]) * rate;
    }
  }
}

// ---- the player -----------------------------------------------------------
class StemPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.stems = null;          // {id: [L, R]}
    this.ids = [];
    this.length = 0;
    this.playing = false;
    this.src = 0;               // next source frame to mix
    this.heard = 0;             // source frame the listener hears
    this.target = {};
    this.gain = {};
    this.loopOn = false;
    this.loopA = 0;
    this.loopB = 0;
    this.speed = 1;
    this.semis = 0;
    this.direct = true;         // skipping the stretcher
    this.wantDirect = true;     // the path to take at the next jump
    this.stretcher = new Stretcher(2);
    this.scratch = [new Float32Array(128), new Float32Array(128)];
    this.pullBuf = [new Float32Array(2048), new Float32Array(2048)];
    this.fadeOut = 0;           // samples left in a fade before a jump
    this.fadeIn = 0;
    this.pending = null;        // a jump waiting for its fade out
    this.beats = [];
    this.downbeats = new Set();
    this.clickOn = false;
    this.clickPhase = -1;
    this.clickFreq = 1000;
    this.blocks = 0;
    this.port.onmessage = (ev) => this.message(ev.data);
  }

  message(m) {
    if (m.type === "load") {
      this.stems = m.stems;
      this.ids = Object.keys(m.stems);
      this.length = m.length;
      this.ids.forEach((id) => {
        if (!(id in this.target)) { this.target[id] = 0.8; }
        this.gain[id] = this.target[id];
      });
      this.beats = m.beats || [];
      this.downbeats = new Set(m.downbeats || []);
      this.loopOn = false;
      this.jump(0, true);
    } else if (m.type === "play") {
      this.playing = true;
      this.jump(this.heard, true);
    } else if (m.type === "pause") {
      this.pending = { pause: true };
      this.fadeOut = FADE;
    } else if (m.type === "seek") {
      this.jump(m.frame, !this.playing);
    } else if (m.type === "gains") {
      Object.assign(this.target, m.gains);
    } else if (m.type === "loop") {
      this.loopOn = m.on;
      this.loopA = m.a;
      this.loopB = m.b;
    } else if (m.type === "params") {
      this.speed = m.speed;
      this.semis = m.semis;
      const direct = m.speed === 1 && m.semis === 0;
      if (direct !== this.direct) {
        // switch paths at the listener's position, behind a short fade;
        // the old path plays out the fade
        this.wantDirect = direct;
        this.jump(this.heard, !this.playing);
      }
    } else if (m.type === "click") {
      this.clickOn = m.on;
    }
  }

  // Move to `frame`: at once when nothing is playing, else after a short
  // fade, so a seek or a path switch never clicks.
  jump(frame, now) {
    frame = Math.max(0, Math.min(this.length - 1, Math.floor(frame)));
    if (now) {
      this.src = frame;
      this.heard = frame;
      this.direct = this.wantDirect;
      this.stretcher.reset(frame);
      this.fadeIn = FADE;
      this.pending = null;
      this.fadeOut = 0;
    } else {
      this.pending = { frame };
      this.fadeOut = FADE;
    }
  }

  // Mix `count` source frames from this.src, looping, into this.pullBuf.
  pull(count) {
    const L = this.pullBuf[0];
    const R = this.pullBuf[1];
    const start = this.src;
    let end = this.loopOn ? this.loopB : this.length;
    if (start >= end) { this.wrap(); return this.pull(count); }
    const n = Math.min(count, end - start);
    L.fill(0, 0, n);
    R.fill(0, 0, n);
    for (const id of this.ids) {
      const st = this.stems[id];
      const sl = st[0];
      const sr = st[1];
      let g = this.gain[id];
      const want = this.target[id];
      const step = (want - g) / 256;
      for (let i = 0; i < n; i++) {
        if (g !== want) {
          g = Math.abs(want - g) <= Math.abs(step) ? want : g + step;
        }
        L[i] += sl[start + i] * g;
        R[i] += sr[start + i] * g;
      }
      this.gain[id] = g;
    }
    // soften the seams of a loop
    const a = this.loopOn ? this.loopA : 0;
    for (let i = 0; i < n; i++) {
      const frame = start + i;
      let e = 1;
      if (end - frame < EDGE) { e = (end - frame) / EDGE; }
      if (frame - a < EDGE && this.wrapped) { e = Math.min(e, (frame - a) / EDGE); }
      if (e < 1) { L[i] *= e; R[i] *= e; }
    }
    this.src = start + n;
    if (this.src >= end) { this.wrap(); }
    return { data: [L, R], frame: start, count: n };
  }

  wrap() {
    this.src = this.loopOn ? this.loopA : 0;
    this.wrapped = true;
    this.port.postMessage({ type: "wrap" });
  }

  renderBlock(out, n) {
    if (this.direct) {
      const got = this.pull(n);
      let done = got.count;
      out[0].set(got.data[0].subarray(0, done));
      out[1].set(got.data[1].subarray(0, done));
      this.heard = got.frame + done;
      while (done < n) {
        const more = this.pull(n - done);
        out[0].set(more.data[0].subarray(0, more.count), done);
        out[1].set(more.data[1].subarray(0, more.count), done);
        done += more.count;
        this.heard = more.frame + more.count;
      }
    } else {
      const pitch = Math.pow(2, this.semis / 12);
      const rate = this.speed / pitch;
      this.stretcher.render(out, 0, n, rate, pitch, (c) => this.pull(c));
      this.heard = this.stretcher.position;
    }
  }

  click(out, n, before, after) {
    if (this.clickOn && after > before) {
      for (const beat of this.beats) {
        if (beat >= before && beat < after) {
          this.clickPhase = 0;
          this.clickFreq = this.downbeats.has(beat) ? 1600 : 1100;
          break;
        }
      }
    }
    if (this.clickPhase < 0) { return; }
    for (let i = 0; i < n && this.clickPhase >= 0; i++) {
      const t = this.clickPhase / sampleRate;
      const v = Math.sin(2 * Math.PI * this.clickFreq * t) *
        Math.exp(-t * 90) * 0.35;
      out[0][i] += v;
      out[1][i] += v;
      this.clickPhase++;
      if (t > 0.06) { this.clickPhase = -1; }
    }
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const n = out[0].length;
    if (!this.stems || (!this.playing && !this.pending)) {
      return true;
    }
    const L = this.scratch[0].length === n ? this.scratch :
      [new Float32Array(n), new Float32Array(n)];
    const before = this.heard;
    this.renderBlock(L, n);
    this.click(L, n, before, this.heard);
    for (let i = 0; i < n; i++) {
      let e = 1;
      if (this.fadeOut > 0) {
        e = Math.max(0, this.fadeOut - i) / FADE;
      } else if (this.fadeIn > 0) {
        e = Math.min(1, (FADE - this.fadeIn + i) / FADE);
      }
      out[0][i] = L[0][i] * e;
      if (out[1]) { out[1][i] = L[1][i] * e; }
    }
    if (this.fadeOut > 0) {
      this.fadeOut = Math.max(0, this.fadeOut - n);
      if (this.fadeOut === 0 && this.pending) {
        const p = this.pending;
        this.pending = null;
        if (p.pause) {
          this.playing = false;
          this.wrapped = false;
        } else {
          this.wrapped = false;
          this.jump(p.frame, true);
        }
      }
    } else if (this.fadeIn > 0) {
      this.fadeIn = Math.max(0, this.fadeIn - n);
    }
    if (++this.blocks % 6 === 0 || !this.playing) {
      this.port.postMessage({ type: "pos", frame: this.heard });
    }
    return true;
  }
}

registerProcessor("stem-player", StemPlayer);
