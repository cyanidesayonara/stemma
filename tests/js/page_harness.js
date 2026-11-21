/* Runs site/demo.js against a fake page and audio engine, for
   tests/test_site.py.

   node page_harness.js DEMO SONGS.json SCENARIO

   The fake page has just what demo.js touches; the fake AudioContext
   decodes each song after a short delay, and the fake worklet records
   every message the page sends. Each scenario clicks like a visitor and
   prints, as JSON, what reached the engine and what the page shows. */
"use strict";

const fs = require("fs");

const songs = JSON.parse(fs.readFileSync(process.argv[3], "utf8")).songs;
const scenario = process.argv[4];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- a fake page ------------------------------------------------------------
class El {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.attrs = {};
    this.dataset = {};
    this.style = {};
    this.listeners = {};
    this.textContent = "";
    this.value = "";
    this.checked = false;
    this.options = [];
    this.clientWidth = 800;
    this.clientHeight = 240;
    this.width = 0;
    this.height = 0;
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  appendChild(c) { this.children.push(c); return c; }
  set innerHTML(_v) { this.children = []; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  fire(t, ev) {
    (this.listeners[t] || []).forEach((f) => f(Object.assign({
      target: this, preventDefault() {} }, ev || {})));
  }
  click() { this.fire("click"); }
  getContext() {
    return new Proxy({}, { get: (o, k) => (k in o ? o[k] : () => {}),
      set: (o, k, v) => { o[k] = v; return true; } });
  }
  getBoundingClientRect() { return { left: 0, top: 0 }; }
  setPointerCapture() {}
}

const byId = {};
["cockpit", "lanes", "songs", "credit", "stems"].forEach((id) => {
  byId[id] = new El(id === "lanes" ? "canvas" : "div");
});
const readouts = ["key", "chord", "tempo", "time", "loop", "status"].map(
  (name) => { const e = new El("span"); e.attrs["data-readout"] = name; return e; });
const acts = {};
["play", "speed", "pitch", "click", "loop", "clear", "trainer"].forEach((a) => {
  acts[a] = new El("control");
});
acts.speed.options = [0.5, 0.75, 0.85, 1, 1.25, 1.5, 2].map(
  (v) => ({ value: String(v) }));
acts.speed.value = "1";
byId.cockpit.querySelectorAll = () => readouts;
byId.cockpit.querySelector = (sel) => acts[/data-act="(\w+)"/.exec(sel)[1]];

globalThis.window = globalThis;
globalThis.devicePixelRatio = 1;
globalThis.document = {
  getElementById: (id) => byId[id] || null,
  createElement: (tag) => new El(tag),
  createTextNode: (text) => ({ textContent: text }),
  documentElement: new El("html"),
  body: new El("body"),
  addEventListener() {},
};
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "",
  fontFamily: "sans-serif" });
globalThis.addEventListener = () => {};
globalThis.requestAnimationFrame = () => 0;

// ---- a fake engine ----------------------------------------------------------
const sent = [];
const calls = [];
class FakeContext {
  constructor() {
    this.state = "running";
    this.sampleRate = 44100;
    this.destination = {};
    this.audioWorklet = { addModule: () => Promise.resolve() };
  }
  resume() { calls.push("resume"); this.state = "running"; return Promise.resolve(); }
  suspend() { calls.push("suspend"); this.state = "suspended"; return Promise.resolve(); }
  createGain() { return { gain: {}, connect() {} }; }
  decodeAudioData(bytes) {
    // each song's stems decode in 40 ms; the length tells the songs apart
    const length = bytes.length;
    return sleep(40).then(() => ({
      length, numberOfChannels: 2,
      getChannelData: () => new Float32Array(length),
    }));
  }
}
globalThis.AudioContext = FakeContext;
globalThis.AudioWorkletNode = class {
  constructor() {
    this.port = { postMessage: (m) => sent.push(m), onmessage: null };
    globalThis.engineNode = this;
  }
  connect() {}
};
globalThis.fetch = (url) => {
  if (url.endsWith("songs.json")) {
    return Promise.resolve({ ok: true, json: () => Promise.resolve({ songs }) });
  }
  const id = url.split("/")[1];
  const length = 1000 + songs.findIndex((s) => s.id === id);
  return Promise.resolve({ ok: true,
    arrayBuffer: () => Promise.resolve({ length }) });
};

new Function(fs.readFileSync(process.argv[2], "utf8"))();

const button = (id) => byId.songs.children.find((b) => b.dataset.id === id);
const loads = () => sent.filter((m) => m.type === "load")
  .map((m) => songs[m.length - 1000].id);
// the engine reports that the pause before the switch has faded out
const stopped = () => globalThis.engineNode.port.onmessage(
  { data: { type: "stopped", gen: 1 } });

(async () => {
  await sleep(10);   // songs.json
  const [a, b, c] = songs.map((s) => s.id);
  acts.play.click();
  await sleep(100);
  const results = { first: loads() };
  if (scenario === "switch-twice") {
    // while A plays: B, then C straight away
    button(b).click();
    await sleep(5);
    button(c).click();
  } else if (scenario === "switch-back") {
    // while A plays: B, then back to A straight away
    button(b).click();
    await sleep(5);
    button(a).click();
  }
  await sleep(3);
  stopped();
  await sleep(200);
  const after = sent.slice(sent.map((m) => m.type).lastIndexOf("load"));
  results.loads = loads();
  results.playAfterLastLoad = after.some((m) => m.type === "play");
  results.pressed = acts.play.getAttribute("aria-pressed");
  results.suspends = calls.filter((x) => x === "suspend").length;
  process.stdout.write(JSON.stringify(results));
})();
