/* The mixer demo's audio engine, an AudioWorklet: it mixes a song's stems
   with their gains, loops and seeks, and changes speed and pitch live the
   way the app does (src/stretch.py): a phase vocoder, one frame at a time,
   then a resampler when the pitch moves. Speed alone changes the tempo,
   pitch alone the key.

   The phase vocoder is the app's (librosa's): a periodic Hann window,
   2048-point frames, a 512-sample hop, magnitudes interpolated between two
   analysis frames and the phase carried as unit phasors. The pitch
   resampler here is a variable-rate cubic interpolator rather than the
   app's soxr stream, so a pitch change needs no crossfade. It has no
   anti-alias filter: at +5 semitones, content above about 16.5 kHz aliases,
   which was judged inaudible enough for a demo. At 1.0x and the original
   pitch the stretcher is skipped, as the app skips it.

   Every buffer is allocated up front: nothing allocates on the audio
   thread while it plays, so the garbage collector never interrupts it.

   Messages in:  load {gen, stems, length, beats, downbeats}, play, pause,
                 seek {frame}, gains {gains}, loop {on, a, b},
                 params {speed, semis}, click {on}
   Messages out: pos {gen, frame}, wrap {gen} (when the listener hears the
                 loop start again) */

const N_FFT = 2048;
const HOP = 512;
const PAD = N_FFT / 2;
const BINS = N_FFT / 2 + 1;
const TINY = 1e-10;
const RING = 1 << 15;          // input ring, samples per channel
const FIFO = 1 << 14;          // stretched samples waiting to be resampled
const RUNS = 32;               // pulled runs remembered, for positions
const MARKS = 64;              // emitted hops remembered, for positions
const PULL = 2048;             // source frames mixed per pull
const FADE = 128;              // samples to fade over before a jump
const EDGE = 64;               // samples to fade at a loop seam

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

// One analysis frame's spectrum: magnitude and unit phasor per channel.
class Spectrum {
  constructor(channels) {
    this.j = -1;
    this.mag = [];
    this.ur = [];
    this.ui = [];
    for (let c = 0; c < channels; c++) {
      this.mag.push(new Float32Array(BINS));
      this.ur.push(new Float32Array(BINS));
      this.ui.push(new Float32Array(BINS));
    }
  }
}

// ---- the stretcher: one channel pair, streaming ---------------------------
class Stretcher {
  constructor(channels) {
    const ch = channels;
    this.ch = ch;
    this.re = new Float32Array(N_FFT);
    this.im = new Float32Array(N_FFT);
    this.ring = [];
    this.ola = [];
    this.fifo = [];
    this.pr = [];
    this.pi = [];
    this.ar = [];
    this.ai = [];
    for (let c = 0; c < ch; c++) {
      this.ring.push(new Float32Array(RING));
      this.ola.push(new Float32Array(N_FFT));
      this.fifo.push(new Float32Array(FIFO));
      this.pr.push(new Float32Array(BINS));
      this.pi.push(new Float32Array(BINS));
      this.ar.push(new Float32Array(BINS));
      this.ai.push(new Float32Array(BINS));
    }
    this.norm = new Float32Array(N_FFT);
    // two frames are in use at a time (i0, i1); the third is the next one
    this.slots = [new Spectrum(ch), new Spectrum(ch), new Spectrum(ch)];
    this.runs = new Float64Array(RUNS * 3);   // padded start, frame, count
    this.marks = new Float64Array(MARKS * 2); // stretched index, frame
    this.recentT = new Float64Array(3);
    this.reset(0);
  }

  reset(position) {
    for (let c = 0; c < this.ch; c++) {
      // the ring must start silent: padded indices below PAD are the
      // centring padding, and read as zeros
      this.ring[c].fill(0);
      this.ola[c].fill(0);
    }
    this.norm.fill(0);
    for (const s of this.slots) { s.j = -1; }
    this.inputEnd = PAD;       // padded index after the last input sample
    this.runCount = 0;
    this.runNext = 0;
    this.t = 0;                // analysis position, in frames
    this.started = false;      // phase seeded from the first frame yet
    this.recentCount = 0;
    this.drop = PAD;           // leading output samples that are padding
    this.fifoStart = 0;        // stretched index of fifo[c][0]
    this.fifoLen = 0;
    this.read = 1;             // resampler position, stretched index
    this.markCount = 0;
    this.markNext = 0;
    this.position = position;
  }

  // Pull source audio until the padded input reaches `need`.
  fill(need, source) {
    while (this.inputEnd < need) {
      const got = source.pull(PULL);
      const n = got.count;
      const r = this.runNext * 3;
      this.runs[r] = this.inputEnd;
      this.runs[r + 1] = got.frame;
      this.runs[r + 2] = n;
      this.runNext = (this.runNext + 1) % RUNS;
      this.runCount = Math.min(RUNS, this.runCount + 1);
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
    let oldest = this.slots[0];
    for (const s of this.slots) {
      if (s.j === j) { return s; }
      if (s.j < oldest.j) { oldest = s; }
    }
    const s = oldest;
    s.j = j;
    const start = j * HOP;
    const re = this.re;
    const im = this.im;
    for (let c = 0; c < this.ch; c++) {
      const ring = this.ring[c];
      for (let i = 0; i < N_FFT; i++) {
        re[i] = ring[(start + i) & (RING - 1)] * WINDOW[i];
        im[i] = 0;
      }
      fft(re, im, false);
      const mag = s.mag[c];
      const ur = s.ur[c];
      const ui = s.ui[c];
      for (let b = 0; b < BINS; b++) {
        const m = Math.sqrt(re[b] * re[b] + im[b] * im[b]);
        mag[b] = m;
        if (m > TINY) { ur[b] = re[b] / m; ui[b] = im[b] / m; }
        else { ur[b] = 1; ui[b] = 0; }
      }
    }
    return s;
  }

  sourceFrame(t) {
    const padded = t * HOP + PAD;
    for (let k = 1; k <= this.runCount; k++) {
      const r = ((this.runNext - k + RUNS) % RUNS) * 3;
      if (padded >= this.runs[r]) {
        return this.runs[r + 1] + Math.min(padded - this.runs[r],
          this.runs[r + 2]);
      }
    }
    return this.position;
  }

  synthesise(rate, source) {
    const t = this.t;
    const i0 = Math.floor(t);
    this.fill((i0 + 1) * HOP + N_FFT, source);
    const s0 = this.spectrum(i0);
    const s1 = this.spectrum(i0 + 1);
    const a = t - i0;
    const re = this.re;
    const im = this.im;
    for (let c = 0; c < this.ch; c++) {
      const pr = this.pr[c];
      const pi = this.pi[c];
      const ar = this.ar[c];
      const ai = this.ai[c];
      if (!this.started) {
        pr.set(s0.ur[c]);
        pi.set(s0.ui[c]);
      } else {
        for (let b = 0; b < BINS; b++) {
          const r = pr[b] * ar[b] - pi[b] * ai[b];
          const i = pr[b] * ai[b] + pi[b] * ar[b];
          const m = Math.max(Math.sqrt(r * r + i * i), TINY);
          pr[b] = r / m; pi[b] = i / m;
        }
      }
      const u0r = s0.ur[c], u0i = s0.ui[c], u1r = s1.ur[c], u1i = s1.ui[c];
      const m0 = s0.mag[c], m1 = s1.mag[c];
      for (let b = 0; b < BINS; b++) {
        // advance = unit1 * conj(unit0)
        ar[b] = u1r[b] * u0r[b] + u1i[b] * u0i[b];
        ai[b] = u1i[b] * u0r[b] - u1r[b] * u0i[b];
        const m = m0[b] + a * (m1[b] - m0[b]);
        re[b] = m * pr[b];
        im[b] = m * pi[b];
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
    this.started = true;
    for (let i = 0; i < N_FFT; i++) { this.norm[i] += WINDOW[i] * WINDOW[i]; }
    if (this.recentCount < 3) {
      this.recentT[this.recentCount++] = t;
    } else {
      this.recentT[0] = this.recentT[1];
      this.recentT[1] = this.recentT[2];
      this.recentT[2] = t;
    }

    // the first hop takes no more frames: normalise and emit it
    const skip = Math.min(this.drop, HOP);
    this.drop -= skip;
    this.compact();
    if (skip < HOP) {
      const m = this.markNext * 2;
      this.marks[m] = this.fifoStart + this.fifoLen;
      this.marks[m + 1] = this.sourceFrame(this.recentT[0]);
      this.markNext = (this.markNext + 1) % MARKS;
      this.markCount = Math.min(MARKS, this.markCount + 1);
    }
    for (let c = 0; c < this.ch; c++) {
      const ola = this.ola[c];
      const q = this.fifo[c];
      let w = this.fifoLen;
      for (let i = skip; i < HOP; i++) {
        q[w++] = ola[i] / Math.max(this.norm[i], TINY);
      }
      ola.copyWithin(0, HOP);
      ola.fill(0, N_FFT - HOP);
    }
    this.fifoLen += HOP - skip;
    this.norm.copyWithin(0, HOP);
    this.norm.fill(0, N_FFT - HOP);
    this.t = t + rate;
  }

  // Make room for one more hop by dropping what the resampler has passed
  // (keeping read-1 for the cubic). The resampler consumes as fast as hops
  // arrive, so the FIFO only ever holds a few hops.
  compact() {
    if (this.fifoLen + HOP <= FIFO) { return; }
    const used = Math.floor(this.read) - 1 - this.fifoStart;
    if (used <= 0) { return; }
    for (let c = 0; c < this.ch; c++) {
      this.fifo[c].copyWithin(0, used, this.fifoLen);
    }
    this.fifoLen -= used;
    this.fifoStart += used;
  }

  // Write `count` output samples into `out` from `offset`, stretching by
  // `rate` (analysis frames per output frame) and resampling by `pitch`.
  render(out, offset, count, rate, pitch, source) {
    const q = this.fifo;
    for (let n = 0; n < count; n++) {
      // the cubic needs read-1 .. read+2
      while (Math.floor(this.read) + 2 >= this.fifoStart + this.fifoLen) {
        this.synthesise(rate, source);
      }
      const base = Math.floor(this.read);
      const f = this.read - base;
      const k = base - this.fifoStart;
      for (let c = 0; c < this.ch; c++) {
        const y0 = k > 0 ? q[c][k - 1] : 0;
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
    // what the listener hears now: the latest hop it has reached
    for (let k = 1; k <= this.markCount; k++) {
      const m = ((this.markNext - k + MARKS) % MARKS) * 2;
      if (this.marks[m] <= this.read) {
        this.position = this.marks[m + 1] + (this.read - this.marks[m]) * rate;
        break;
      }
    }
  }
}

// ---- the player -----------------------------------------------------------
class StemPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.gen = 0;               // which load the engine holds
    this.stems = null;          // {id: [L, R]}
    this.ids = [];
    this.length = 0;
    this.playing = false;
    this.pausing = false;       // a pause waiting for its fade out
    this.jumpTo = -1;           // a jump waiting for its fade out
    this.fadeOut = 0;
    this.fadeIn = 0;
    this.src = 0;               // next source frame to mix
    this.heard = 0;             // source frame the listener hears
    this.jumped = false;        // heard moved by a jump, not a wrap
    this.wrapped = false;       // the source has wrapped since the last jump
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
    this.block = [new Float32Array(128), new Float32Array(128)];
    this.pullBuf = [new Float32Array(PULL), new Float32Array(PULL)];
    this.got = { data: this.pullBuf, frame: 0, count: 0 };
    this.beats = new Float64Array(0);
    this.downbeats = new Set();
    this.clickOn = false;
    this.clickPhase = -1;
    this.clickFreq = 1000;
    this.blocks = 0;
    this.port.onmessage = (ev) => this.message(ev.data);
  }

  message(m) {
    if (m.type === "load") {
      this.gen = m.gen;
      this.stems = m.stems;
      this.ids = Object.keys(m.stems);
      this.length = m.length;
      for (const id of this.ids) {
        if (!(id in this.target)) { this.target[id] = 0.8; }
        this.gain[id] = this.target[id];
      }
      this.beats = Float64Array.from(m.beats || []);
      this.downbeats = new Set(m.downbeats || []);
      this.loopOn = false;
      this.playing = false;
      this.pausing = false;
      this.jump(0, true);
    } else if (m.type === "play") {
      this.pausing = false;
      if (!this.playing) {
        this.playing = true;
        this.jump(this.heard, true);
      }
    } else if (m.type === "pause") {
      if (this.playing && !this.pausing) {
        this.pausing = true;
        if (this.fadeOut === 0) { this.fadeOut = FADE; }
      }
    } else if (m.type === "seek") {
      this.jump(m.frame, !this.playing);
    } else if (m.type === "gains") {
      Object.assign(this.target, m.gains);
    } else if (m.type === "loop") {
      const a = Math.max(0, Math.min(this.length, Math.floor(m.a)));
      const b = Math.max(0, Math.min(this.length, Math.floor(m.b)));
      this.loopOn = Boolean(m.on) && b - a > EDGE * 2;
      this.loopA = a;
      this.loopB = b;
      this.wrapped = false;
    } else if (m.type === "params") {
      this.speed = m.speed;
      this.semis = m.semis;
      const direct = m.speed === 1 && m.semis === 0;
      if (direct !== this.wantDirect) {
        // switch paths at the listener's position, behind a short fade;
        // the old path plays out the fade
        this.wantDirect = direct;
        // a jump already waiting (a seek) keeps its target
        this.jump(this.jumpTo >= 0 ? this.jumpTo : this.heard, !this.playing);
      }
    } else if (m.type === "click") {
      this.clickOn = m.on;
    }
  }

  // Move to `frame`: at once when nothing is playing, else after a short
  // fade, so a seek or a path switch never clicks.
  jump(frame, now) {
    frame = Math.max(0, Math.min(Math.max(0, this.length - 1),
      Math.floor(frame)));
    this.wrapped = false;
    if (now) {
      this.src = frame;
      this.heard = frame;
      this.jumped = true;
      this.direct = this.wantDirect;
      this.stretcher.reset(frame);
      this.fadeIn = FADE;
      this.jumpTo = -1;
      this.fadeOut = 0;
    } else {
      this.jumpTo = frame;
      if (this.fadeOut === 0) { this.fadeOut = FADE; }
    }
  }

  loopStart() { return this.loopOn ? this.loopA : 0; }
  loopEnd() { return this.loopOn ? this.loopB : this.length; }

  // Mix up to `count` source frames from this.src into this.pullBuf.
  pull(count) {
    const L = this.pullBuf[0];
    const R = this.pullBuf[1];
    let end = this.loopEnd();
    if (this.src >= end) {
      this.src = this.loopStart();
      this.wrapped = true;
    }
    const start = this.src;
    const n = Math.max(1, Math.min(count, end - start));
    L.fill(0, 0, n);
    R.fill(0, 0, n);
    for (let s = 0; s < this.ids.length; s++) {
      const id = this.ids[s];
      const sl = this.stems[id][0];
      const sr = this.stems[id][1];
      let g = this.gain[id];
      const want = this.target[id];
      const step = (want - g) / 256;
      for (let i = 0; i < n; i++) {
        if (g !== want) {
          g = Math.abs(want - g) <= Math.abs(step) ? want : g + step;
        }
        L[i] += (sl[start + i] || 0) * g;
        R[i] += (sr[start + i] || 0) * g;
      }
      this.gain[id] = g;
    }
    // soften the seams of a loop: out before its end, in after its start
    const a = this.loopStart();
    for (let i = 0; i < n; i++) {
      const frame = start + i;
      let e = 1;
      if (end - frame < EDGE) { e = (end - frame) / EDGE; }
      if (this.wrapped && frame >= a && frame - a < EDGE) {
        e = Math.min(e, (frame - a + 1) / EDGE);
      }
      if (e < 1) { L[i] *= e; R[i] *= e; }
    }
    this.src = start + n;
    if (this.src >= end) {
      this.src = this.loopStart();
      this.wrapped = true;
    }
    this.got.frame = start;
    this.got.count = n;
    return this.got;
  }

  renderBlock(out, n) {
    if (this.direct) {
      let done = 0;
      while (done < n) {
        const got = this.pull(n - done);
        out[0].set(got.data[0].subarray(0, got.count), done);
        out[1].set(got.data[1].subarray(0, got.count), done);
        done += got.count;
        this.heard = got.frame + got.count;
      }
    } else {
      const pitch = Math.pow(2, this.semis / 12);
      const rate = this.speed / pitch;
      this.stretcher.render(out, 0, n, rate, pitch, this);
      // the position runs on from the last hop it saw; inside a loop it
      // never passes the loop's end (the wrap is already on its way)
      this.heard = this.loopOn ?
        Math.min(this.stretcher.position, this.loopB - 1) :
        this.stretcher.position;
    }
  }

  // Start a click on any beat the listener passed in this block; across a
  // loop wrap that is [before, loop end) and [loop start, after).
  clickBeats(before, after, wrapped) {
    if (!this.clickOn) { return; }
    if (wrapped) {
      this.clickRange(before, this.loopEnd());
      // the loop's first beat counts even when the block lands right on it
      this.clickRange(this.loopStart(), Math.max(after, this.loopStart() + 1));
    } else {
      this.clickRange(before, after);
    }
  }

  clickRange(from, to) {
    for (let k = 0; k < this.beats.length; k++) {
      const beat = this.beats[k];
      if (beat >= from && beat < to) {
        this.clickPhase = 0;
        this.clickFreq = this.downbeats.has(beat) ? 1600 : 1100;
      }
    }
  }

  addClick(out, n) {
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
    if (!this.stems || !this.playing) { return true; }
    if (this.block[0].length !== n) {
      this.block = [new Float32Array(n), new Float32Array(n)];
    }
    const buf = this.block;
    const before = this.heard;
    this.jumped = false;
    this.renderBlock(buf, n);
    // the listener heard the song move back, and no jump moved it: a wrap
    const wrapped = !this.jumped && this.heard < before - 1;
    this.clickBeats(before, this.heard, wrapped);
    this.addClick(buf, n);
    for (let i = 0; i < n; i++) {
      let e = 1;
      if (this.fadeOut > 0) {
        e = Math.max(0, this.fadeOut - i) / FADE;
      } else if (this.fadeIn > 0) {
        e = Math.min(1, (FADE - this.fadeIn + i) / FADE);
      }
      out[0][i] = buf[0][i] * e;
      if (out[1]) { out[1][i] = buf[1][i] * e; }
    }
    if (wrapped) { this.port.postMessage({ type: "wrap", gen: this.gen }); }
    if (this.fadeOut > 0) {
      this.fadeOut = Math.max(0, this.fadeOut - n);
      if (this.fadeOut === 0) {
        if (this.jumpTo >= 0) {
          this.jump(this.jumpTo, true);
        } else {
          this.fadeIn = FADE;   // a pause called off mid-fade comes back in
        }
        if (this.pausing) {
          this.pausing = false;
          this.playing = false;
        }
      }
    } else if (this.fadeIn > 0) {
      this.fadeIn = Math.max(0, this.fadeIn - n);
    }
    if (++this.blocks % 6 === 0 || !this.playing) {
      this.port.postMessage({ type: "pos", gen: this.gen, frame: this.heard });
    }
    return true;
  }
}

registerProcessor("stem-player", StemPlayer);
