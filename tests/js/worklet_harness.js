/* Runs site/stretch-worklet.js outside a browser for tests/test_site.py.

   node worklet_harness.js WORKLET SCENARIO [INPUT.f32 OUTPUT.f32]

   Stubs the AudioWorklet globals, loads the file, plays a scenario and
   prints its findings as JSON. "stretch" instead reads a stereo float32
   signal (channel-major), stretches it, and writes the result for the
   test to compare against src/stretch.py. */
"use strict";

const fs = require("fs");

const SR = 44100;
globalThis.sampleRate = SR;
globalThis.registerProcessor = () => {};
globalThis.AudioWorkletProcessor = class {
  constructor() {
    this.sent = [];
    this.port = { postMessage: (m) => this.sent.push(m) };
  }
};
const src = fs.readFileSync(process.argv[2], "utf8");
const { Stretcher, StemPlayer } = new Function(
  src + "\nreturn { Stretcher, StemPlayer };")();

const scenario = process.argv[3];

// six stems of a steady tone, peak 0.1 each, mixed at 0.8: peak 0.48
function player(seconds) {
  const len = Math.round(SR * seconds);
  const stems = {};
  ["vocals", "drums", "bass", "other", "guitar", "piano"].forEach((id, k) => {
    const l = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      l[i] = 0.1 * Math.sin(2 * Math.PI * (220 + 30 * k) * i / SR);
    }
    stems[id] = [l, l.slice()];
  });
  const p = new StemPlayer();
  const beats = [];
  for (let t = 0; t < seconds; t += 0.5) { beats.push(Math.round(t * SR)); }
  p.message({ type: "load", gen: 1, stems, length: len, beats,
    downbeats: beats.filter((_, i) => i % 4 === 0) });
  return p;
}

function run(p, seconds) {
  const out = [[new Float32Array(128), new Float32Array(128)]];
  let peak = 0;
  let nan = false;
  const blocks = Math.round(SR * seconds / 128);
  for (let b = 0; b < blocks; b++) {
    p.process([], out);
    for (const ch of out[0]) {
      for (const v of ch) {
        if (!Number.isFinite(v)) { nan = true; }
        peak = Math.max(peak, Math.abs(v));
      }
    }
  }
  return { peak, nan };
}

const results = {};
const loop = (p, a, b, on = true) =>
  p.message({ type: "loop", on, a: Math.round(a * SR), b: Math.round(b * SR) });

if (scenario === "stretch") {
  const raw = new Float32Array(fs.readFileSync(process.argv[4]).buffer.slice(0));
  const speed = parseFloat(process.argv[6]);
  const len = raw.length / 2;
  const L = raw.subarray(0, len);
  const R = raw.subarray(len);
  const s = new Stretcher(2);
  const buf = [new Float32Array(2048), new Float32Array(2048)];
  const got = { data: buf, frame: 0, count: 0 };
  let pos = 0;
  const source = { pull(n) {
    const c = Math.min(n, len - pos);
    buf[0].fill(0); buf[1].fill(0);
    buf[0].set(L.subarray(pos, pos + c));
    buf[1].set(R.subarray(pos, pos + c));
    got.frame = pos; got.count = Math.max(c, 1); pos += c;
    return got;
  } };
  const n = Math.floor(len / speed * 0.8 / 128) * 128;
  const out = [new Float32Array(n), new Float32Array(n)];
  for (let o = 0; o < n; o += 128) { s.render(out, o, 128, speed, 1, source); }
  const flat = new Float32Array(n * 2);
  flat.set(out[0]); flat.set(out[1], n);
  fs.writeFileSync(process.argv[5], Buffer.from(flat.buffer));
} else if (scenario === "loop-moves") {
  // a loop at 1-2 s cycles, then a new loop is dragged at 6-8 s (the page
  // sends loop, then seek), and later the loop is turned off and back on
  const p = player(10);
  p.message({ type: "play" });
  loop(p, 1, 2);
  p.message({ type: "seek", frame: SR });
  results.cycling = run(p, 3);
  loop(p, 6, 8);
  p.message({ type: "seek", frame: 6 * SR });
  results.moved = run(p, 1);
  loop(p, 6, 8, false);
  p.message({ type: "seek", frame: 9 * SR });
  results.wrapToStart = run(p, 1.5);
  loop(p, 6, 8, true);
  results.loopBackOn = run(p, 3);
  p.message({ type: "params", speed: 0.75, semis: 0 });
  results.stretched = run(p, 3);
} else if (scenario === "pause-in-fade") {
  const p = player(10);
  p.message({ type: "play" });
  run(p, 0.5);
  p.message({ type: "pause" });
  p.message({ type: "seek", frame: 3 * SR });
  p.message({ type: "params", speed: 0.8, semis: 0 });
  run(p, 0.2);
  results.playingAfterPause = p.playing;
  results.heard = p.heard / SR;
} else if (scenario === "switch-in-fade") {
  // playing at 1.0x, then 0.75x and straight back to 1.0x inside one fade
  const p = player(10);
  p.message({ type: "play" });
  run(p, 0.5);
  p.message({ type: "params", speed: 0.75, semis: 0 });
  p.message({ type: "params", speed: 1, semis: 0 });
  run(p, 0.2);
  results.direct = p.direct;
} else if (scenario === "clicks") {
  // beats every 0.5 s; a loop from 2.0 to 3.0 s holds beats 2.0 and 2.5
  for (const speed of [1, 0.75]) {
    const p = player(10);
    p.message({ type: "click", on: true });
    p.message({ type: "params", speed, semis: 0 });
    loop(p, 2, 3);
    p.message({ type: "seek", frame: 2 * SR });
    p.message({ type: "play" });
    let clicks = 0;
    let wraps = 0;
    const out = [[new Float32Array(128), new Float32Array(128)]];
    const blocks = Math.round(SR * 4 / speed / 128);
    for (let b = 0; b < blocks; b++) {
      const before = p.clickPhase;
      p.process([], out);
      if (p.clickPhase >= 0 && (before < 0 || p.clickPhase < before)) {
        clicks++;
      }
    }
    wraps = p.sent.filter((m) => m.type === "wrap").length;
    results["speed" + speed] = { clicks, wraps };
  }
} else if (scenario === "past-the-end") {
  const p = player(5);
  const len = p.length;
  p.message({ type: "play" });
  p.message({ type: "loop", on: true, a: len - SR, b: len + 500 });
  p.message({ type: "seek", frame: len - SR });
  results.direct = run(p, 3);
  p.message({ type: "params", speed: 0.75, semis: 0 });
  results.stretched = run(p, 3);
  results.loopB = p.loopB;
  results.length = len;
} else if (scenario === "gens") {
  const p = player(5);
  p.message({ type: "play" });
  run(p, 0.2);
  results.gens = [...new Set(p.sent.map((m) => m.gen))];
}

process.stdout.write(JSON.stringify(results));
