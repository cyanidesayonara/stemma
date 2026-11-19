/* The "Try the mixer" demo: an eight-bar tune made for this page, played as
   six separate stems through the Web Audio API, with stemma's practice
   controls (mute, solo, volume, A-B loop, Loop Trainer, speed, pitch,
   metronome) and its stacked, stem-coloured waveform lanes.

   The tune is sequenced, not recorded, so speed changes the tempo without
   touching the pitch and pitch changes the key without touching the tempo,
   the same promise the app keeps for real songs. Nothing here separates
   audio; the app does that, on the visitor's PC. */
(function () {
  "use strict";

  var cockpit = document.getElementById("cockpit");
  var canvas = document.getElementById("lanes");
  if (!cockpit || !canvas || !canvas.getContext) { return; }

  var AudioCtx = window.AudioContext || window.webkitAudioContext;
  var BASE_BPM = 100;
  var BARS = 8;
  var STEPS_PER_BAR = 16;
  var STEPS = BARS * STEPS_PER_BAR;
  var NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A",
    "Bb", "B"];

  // the progression, one chord per bar: root pitch class, quality, voicing
  var CHORDS = [
    { root: 9, minor: true, bass: 45, tones: [57, 60, 64] },   // Am
    { root: 5, minor: false, bass: 41, tones: [57, 60, 65] },  // F
    { root: 0, minor: false, bass: 48, tones: [55, 60, 64] },  // C
    { root: 7, minor: false, bass: 43, tones: [55, 59, 62] },  // G
    { root: 5, minor: false, bass: 41, tones: [57, 60, 65] },  // F
    { root: 7, minor: false, bass: 43, tones: [55, 59, 62] },  // G
    { root: 9, minor: true, bass: 45, tones: [57, 60, 64] },   // Am
    { root: 9, minor: true, bass: 45, tones: [57, 60, 64] }    // Am
  ];

  // the sung line, per bar: [step, midi note, length in steps]
  var MELODY = [
    [[0, 76, 4], [4, 74, 2], [6, 72, 2], [8, 69, 6], [14, 72, 2]],
    [[0, 72, 4], [4, 69, 4], [8, 65, 6], [14, 67, 2]],
    [[0, 67, 2], [2, 69, 2], [4, 72, 4], [8, 76, 4], [12, 74, 4]],
    [[0, 74, 6], [6, 71, 2], [8, 67, 8]],
    [[0, 69, 4], [4, 72, 4], [8, 77, 6], [14, 76, 2]],
    [[0, 74, 4], [4, 76, 2], [6, 74, 2], [8, 71, 8]],
    [[0, 72, 4], [4, 71, 2], [6, 72, 2], [8, 69, 8]],
    [[0, 64, 4], [4, 69, 12]]
  ];

  function readVar(name) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
  }

  var STEMS = [
    { id: "vocals", label: "Vocals" },
    { id: "drums", label: "Drums" },
    { id: "bass", label: "Bass" },
    { id: "other", label: "Other" },
    { id: "guitar", label: "Guitar" },
    { id: "piano", label: "Piano" }
  ];

  // every note of the tune, per stem, in steps: {step, len, midi, vel, kind}
  function compose() {
    var notes = { vocals: [], drums: [], bass: [], other: [], guitar: [],
      piano: [] };
    for (var bar = 0; bar < BARS; bar++) {
      var at = bar * STEPS_PER_BAR;
      var chord = CHORDS[bar];
      MELODY[bar].forEach(function (n) {
        notes.vocals.push({ step: at + n[0], len: n[2], midi: n[1], vel: 1 });
      });
      // drums: kick, snare and hats, with a fill into the top
      var kicks = bar % 2 ? [0, 6, 8, 11] : [0, 6, 8];
      kicks.forEach(function (s) {
        notes.drums.push({ step: at + s, len: 2, kind: "kick", vel: 1 });
      });
      var snares = bar === BARS - 1 ? [4, 12, 13, 14, 15] : [4, 12];
      snares.forEach(function (s, i) {
        notes.drums.push({ step: at + s, len: 2, kind: "snare",
          vel: s > 12 ? 0.55 + 0.1 * i : 0.9 });
      });
      for (var h = 0; h < 16; h += 2) {
        if (bar === BARS - 1 && h >= 12) { break; }
        notes.drums.push({ step: at + h, len: 1, kind: h === 14 ? "open" :
          "hat", vel: h % 4 ? 0.5 : 0.8 });
      }
      // bass: root, octave pop, and a walk to the next bar
      [[0, 3, 0], [3, 1, 12], [6, 2, 0], [8, 4, 0], [12, 2, 7], [14, 2, 12]]
        .forEach(function (b) {
          notes.bass.push({ step: at + b[0], len: b[1], midi: chord.bass + b[2],
            vel: b[0] === 0 ? 1 : 0.8 });
        });
      // piano: two chord stabs a bar
      [0, 8].forEach(function (s) {
        chord.tones.forEach(function (m) {
          notes.piano.push({ step: at + s, len: 7, midi: m + 12, vel: 0.8 });
        });
      });
      // guitar: picked eighths across the chord
      var pattern = [0, 1, 2, 1, 0, 1, 2, 1];
      pattern.forEach(function (t, i) {
        notes.guitar.push({ step: at + i * 2, len: 2,
          midi: chord.tones[t] + (i === 2 || i === 6 ? 12 : 0),
          vel: i % 2 ? 0.6 : 0.9 });
      });
      // other: a pad holding the chord for the whole bar
      chord.tones.forEach(function (m) {
        notes.other.push({ step: at, len: 16, midi: m, vel: 0.7 });
      });
    }
    return notes;
  }

  var NOTES = compose();

  // per-stem "waveforms" for the lanes: an envelope per sixteenth step
  var BINS_PER_STEP = 4;
  function envelopes() {
    var out = {};
    var seed = 7;
    function rand() {
      seed = (seed * 16807) % 2147483647;
      return seed / 2147483647;
    }
    STEMS.forEach(function (stem) {
      var env = new Float32Array(STEPS * BINS_PER_STEP);
      var decay = { vocals: 0.12, drums: 0.5, bass: 0.18, other: 0.02,
        guitar: 0.35, piano: 0.22 }[stem.id];
      NOTES[stem.id].forEach(function (n) {
        var start = n.step * BINS_PER_STEP;
        var len = n.len * BINS_PER_STEP;
        for (var i = 0; i < len && start + i < env.length; i++) {
          var attack = Math.min(1, (i + 1) / (stem.id === "other" ? 10 : 2));
          var v = n.vel * attack * Math.exp(-decay * i);
          if (stem.id === "vocals") { v = n.vel * attack * (0.75 + 0.25 * Math.sin(i)); }
          env[start + i] = Math.min(1, env[start + i] + v * 0.8);
        }
      });
      for (var j = 0; j < env.length; j++) {
        env[j] = env[j] * (0.7 + 0.3 * rand());
      }
      out[stem.id] = env;
    });
    return out;
  }

  var ENV = envelopes();

  // ---- state --------------------------------------------------------------
  var state = {
    playing: false,
    speed: 1,
    pitch: 0,
    click: false,
    trainer: false,
    looping: false,
    loopA: null,   // in steps
    loopB: null,
    step: 0,       // the next step to schedule
    nextTime: 0,   // when it plays, in AudioContext seconds
    queue: [],     // scheduled steps, for the playhead: {step, time, dur}
    mute: {},
    solo: {},
    volume: {}
  };
  STEMS.forEach(function (s) {
    state.mute[s.id] = false;
    state.solo[s.id] = false;
    state.volume[s.id] = 0.8;
  });

  var ctx = null;
  var master = null;
  var buses = {};
  var noise = null;
  var timer = null;

  function stepDur() {
    return 60 / (BASE_BPM * state.speed) / 4;
  }

  function freq(midi) {
    return 440 * Math.pow(2, (midi + state.pitch - 69) / 12);
  }

  function audible(id) {
    var anySolo = STEMS.some(function (s) { return state.solo[s.id]; });
    if (anySolo) { return state.solo[id]; }
    return !state.mute[id];
  }

  function applyGains() {
    if (!ctx) { return; }
    STEMS.forEach(function (s) {
      var target = audible(s.id) ? state.volume[s.id] : 0;
      buses[s.id].gain.setTargetAtTime(target, ctx.currentTime, 0.02);
    });
  }

  function setupAudio() {
    ctx = new AudioCtx();
    master = ctx.createGain();
    master.gain.value = 0.7;
    var comp = ctx.createDynamicsCompressor();
    master.connect(comp);
    comp.connect(ctx.destination);
    STEMS.forEach(function (s) {
      var g = ctx.createGain();
      g.connect(master);
      buses[s.id] = g;
    });
    buses.click = ctx.createGain();
    buses.click.gain.value = 0.5;
    buses.click.connect(master);
    noise = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
    var data = noise.getChannelData(0);
    for (var i = 0; i < data.length; i++) { data[i] = Math.random() * 2 - 1; }
    applyGains();
  }

  // ---- instruments --------------------------------------------------------
  function envGain(t, peak, attack, hold, release) {
    var g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + attack);
    g.gain.setValueAtTime(peak, t + attack + hold);
    g.gain.exponentialRampToValueAtTime(0.0001, t + attack + hold + release);
    return g;
  }

  function osc(type, f, t, end, out, detune) {
    var o = ctx.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(f, t);
    if (detune) { o.detune.value = detune; }
    o.connect(out);
    o.start(t);
    o.stop(end);
    return o;
  }

  function noiseHit(t, len, filterType, f, peak, out) {
    var src = ctx.createBufferSource();
    src.buffer = noise;
    var filt = ctx.createBiquadFilter();
    filt.type = filterType;
    filt.frequency.value = f;
    var g = envGain(t, peak, 0.002, 0, len);
    src.connect(filt);
    filt.connect(g);
    g.connect(out);
    src.start(t);
    src.stop(t + len + 0.05);
  }

  var play = {
    drums: function (n, t) {
      var out = buses.drums;
      if (n.kind === "kick") {
        var g = envGain(t, n.vel, 0.003, 0.02, 0.28);
        g.connect(out);
        var o = osc("sine", 150, t, t + 0.4, g);
        o.frequency.exponentialRampToValueAtTime(42, t + 0.14);
      } else if (n.kind === "snare") {
        noiseHit(t, 0.16, "highpass", 1400, 0.5 * n.vel, out);
        var tg = envGain(t, 0.25 * n.vel, 0.002, 0, 0.09);
        tg.connect(out);
        osc("triangle", 190, t, t + 0.15, tg);
      } else {
        noiseHit(t, n.kind === "open" ? 0.22 : 0.045, "highpass", 7500,
          0.18 * n.vel, out);
      }
    },
    bass: function (n, t, d) {
      var filt = ctx.createBiquadFilter();
      filt.type = "lowpass";
      filt.Q.value = 5;
      filt.frequency.setValueAtTime(900, t);
      filt.frequency.exponentialRampToValueAtTime(220, t + Math.max(0.1, d));
      var g = envGain(t, 0.45 * n.vel, 0.005, d * 0.6, 0.12);
      filt.connect(g);
      g.connect(buses.bass);
      osc("sawtooth", freq(n.midi), t, t + d + 0.2, filt);
    },
    piano: function (n, t, d) {
      var g = envGain(t, 0.11 * n.vel, 0.004, 0, d + 0.25);
      g.connect(buses.piano);
      osc("triangle", freq(n.midi), t, t + d + 0.35, g);
      osc("sine", freq(n.midi) * 2, t, t + d + 0.35, g, 3);
    },
    guitar: function (n, t, d) {
      var filt = ctx.createBiquadFilter();
      filt.type = "lowpass";
      filt.frequency.setValueAtTime(3200, t);
      filt.frequency.exponentialRampToValueAtTime(700, t + 0.25);
      var g = envGain(t, 0.07 * n.vel, 0.002, 0, Math.max(0.2, d * 1.5));
      filt.connect(g);
      g.connect(buses.guitar);
      osc("square", freq(n.midi), t, t + d * 1.5 + 0.3, filt);
    },
    other: function (n, t, d) {
      var filt = ctx.createBiquadFilter();
      filt.type = "lowpass";
      filt.frequency.value = 1100;
      var g = envGain(t, 0.035 * n.vel, Math.min(0.35, d / 3), d * 0.55, 0.45);
      filt.connect(g);
      g.connect(buses.other);
      osc("sawtooth", freq(n.midi), t, t + d + 0.6, filt, -8);
      osc("sawtooth", freq(n.midi), t, t + d + 0.6, filt, 8);
    },
    vocals: function (n, t, d) {
      // a sawtooth through two "ah" formants, with a late vibrato
      var g = envGain(t, 0.22 * n.vel, 0.05, Math.max(0, d - 0.08), 0.12);
      g.connect(buses.vocals);
      var src = ctx.createOscillator();
      src.type = "sawtooth";
      src.frequency.setValueAtTime(freq(n.midi), t);
      var lfo = ctx.createOscillator();
      lfo.frequency.value = 5.5;
      var depth = ctx.createGain();
      depth.gain.setValueAtTime(0, t);
      depth.gain.linearRampToValueAtTime(freq(n.midi) * 0.012, t + Math.min(0.5, d));
      lfo.connect(depth);
      depth.connect(src.frequency);
      var split = ctx.createGain();
      src.connect(split);
      // [centre Hz, Q, level] for each formant
      [[800, 6, 1], [1150, 8, 0.6], [2900, 10, 0.25]].forEach(function (fm) {
        var bp = ctx.createBiquadFilter();
        bp.type = "bandpass";
        bp.frequency.value = fm[0];
        bp.Q.value = fm[1];
        var level = ctx.createGain();
        level.gain.value = fm[2] * 4;
        split.connect(bp);
        bp.connect(level);
        level.connect(g);
      });
      var end = t + d + 0.25;
      src.start(t);
      lfo.start(t);
      src.stop(end);
      lfo.stop(end);
    }
  };

  function clickAt(t, accent) {
    var g = envGain(t, accent ? 0.5 : 0.3, 0.001, 0, 0.04);
    g.connect(buses.click);
    osc("square", accent ? 1800 : 1200, t, t + 0.06, g);
  }

  // ---- the sequencer ------------------------------------------------------
  var byStep = (function () {
    var table = [];
    for (var i = 0; i < STEPS; i++) { table.push([]); }
    STEMS.forEach(function (s) {
      NOTES[s.id].forEach(function (n) { table[n.step].push([s.id, n]); });
    });
    return table;
  })();

  function scheduleStep(step, t) {
    var d = stepDur();
    byStep[step].forEach(function (pair) {
      play[pair[0]](pair[1], t, pair[1].len * d);
    });
    if (state.click && step % 4 === 0) { clickAt(t, step % 16 === 0); }
    state.queue.push({ step: step, time: t, dur: d });
    if (state.queue.length > 64) { state.queue.shift(); }
  }

  function advance() {
    var next = state.step + 1;
    var loopOn = state.looping && state.loopA !== null;
    if (loopOn && next >= state.loopB) {
      next = state.loopA;
      if (state.trainer && state.speed < 1) {
        setSpeed(Math.min(1, Math.round((state.speed + 0.05) * 100) / 100));
      }
    } else if (next >= STEPS) {
      next = 0;
    }
    state.step = next;
  }

  function tick() {
    while (state.nextTime < ctx.currentTime + 0.12) {
      scheduleStep(state.step, state.nextTime);
      state.nextTime += stepDur();
      advance();
    }
  }

  function start() {
    if (!AudioCtx) {
      cockpit.querySelector(".transport .note").textContent =
        "This browser cannot play the demo.";
      return;
    }
    if (!ctx) { setupAudio(); }
    if (ctx.state === "suspended") { ctx.resume(); }
    state.playing = true;
    state.queue = [];
    state.nextTime = ctx.currentTime + 0.06;
    timer = setInterval(tick, 25);
    tick();
    playButton.setAttribute("aria-pressed", "true");
    playButton.setAttribute("aria-label", "Pause");
    requestAnimationFrame(frame);
  }

  function stop() {
    state.playing = false;
    clearInterval(timer);
    // pick up where the ear is, not where the scheduler got to
    state.step = Math.floor(playhead()) % STEPS;
    state.queue = [];
    if (ctx) {
      STEMS.forEach(function (s) {
        var g = buses[s.id].gain;
        g.cancelScheduledValues(ctx.currentTime);
        g.setTargetAtTime(0, ctx.currentTime, 0.01);
      });
      setTimeout(function () { if (!state.playing) { applyGains(); } }, 120);
    }
    playButton.setAttribute("aria-pressed", "false");
    playButton.setAttribute("aria-label", "Play");
    draw();
  }

  function seek(step) {
    state.step = Math.max(0, Math.min(STEPS - 1, step));
    state.queue = [];
    if (state.playing) { state.nextTime = ctx.currentTime + 0.03; }
    draw();
  }

  // where the listener is, in fractional steps
  function playhead() {
    if (!state.playing || !ctx || !state.queue.length) { return state.step; }
    var now = ctx.currentTime;
    for (var i = state.queue.length - 1; i >= 0; i--) {
      var q = state.queue[i];
      if (q.time <= now) {
        return q.step + Math.min(1, (now - q.time) / q.dur);
      }
    }
    return state.queue[0].step;
  }

  // ---- readouts -----------------------------------------------------------
  function chordName(bar) {
    var c = CHORDS[bar];
    return NOTE_NAMES[(c.root + state.pitch + 120) % 12] + (c.minor ? "m" : "");
  }

  function stepLabel(step) {
    var bar = Math.floor(step / STEPS_PER_BAR) + 1;
    var beat = Math.floor((step % STEPS_PER_BAR) / 4) + 1;
    return bar + "." + beat;
  }

  var readouts = {};
  cockpit.querySelectorAll("[data-readout]").forEach(function (el) {
    readouts[el.getAttribute("data-readout")] = el;
  });

  function updateReadouts(pos) {
    var bar = Math.floor(pos / STEPS_PER_BAR) % BARS;
    readouts.key.textContent = NOTE_NAMES[(9 + state.pitch + 120) % 12] + " minor";
    readouts.chord.textContent = chordName(bar);
    readouts.tempo.textContent = Math.round(BASE_BPM * state.speed) + " BPM";
    readouts.time.textContent = "Bar " + (bar + 1) + " of " + BARS;
    readouts.loop.textContent = state.loopA === null ? "No loop set" :
      "A " + stepLabel(state.loopA) + "  B " + stepLabel(state.loopB);
  }

  // ---- the lanes ----------------------------------------------------------
  var g2 = canvas.getContext("2d");
  var colors = {};

  function readColors() {
    STEMS.forEach(function (s) { colors[s.id] = readVar("--stem-" + s.id); });
    colors.bg = readVar("--mantle");
    colors.grid = readVar("--surface0");
    colors.label = readVar("--subtext");
    colors.head = readVar("--text");
    colors.loop = readVar("--recording");
  }

  function size() {
    var ratio = window.devicePixelRatio || 1;
    var w = canvas.clientWidth;
    var h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * ratio) ||
        canvas.height !== Math.round(h * ratio)) {
      canvas.width = Math.round(w * ratio);
      canvas.height = Math.round(h * ratio);
    }
    g2.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { w: w, h: h };
  }

  var LABEL_W = 58;

  function stepToX(step, w) {
    return LABEL_W + (step / STEPS) * (w - LABEL_W - 8);
  }

  function xToStep(x, w) {
    var s = ((x - LABEL_W) / (w - LABEL_W - 8)) * STEPS;
    return Math.max(0, Math.min(STEPS, s));
  }

  function draw() {
    var dims = size();
    var w = dims.w;
    var h = dims.h;
    var pos = playhead();
    g2.clearRect(0, 0, w, h);
    var laneH = (h - 8) / STEMS.length;
    // bar lines
    g2.fillStyle = colors.grid;
    for (var b = 0; b <= BARS; b++) {
      g2.fillRect(Math.round(stepToX(b * STEPS_PER_BAR, w)), 0, 1, h);
    }
    var bins = STEPS * BINS_PER_STEP;
    var span = w - LABEL_W - 8;
    var barW = Math.max(1, span / bins);
    STEMS.forEach(function (s, i) {
      var mid = 4 + laneH * i + laneH / 2;
      var on = audible(s.id);
      g2.globalAlpha = on ? 1 : 0.22;
      g2.fillStyle = colors[s.id];
      var env = ENV[s.id];
      var stride = Math.max(1, Math.floor(bins / span));
      for (var k = 0; k < bins; k += stride) {
        var v = env[k] * state.volume[s.id] / 0.8;
        var amp = Math.max(0.5, v * (laneH / 2 - 3));
        g2.fillRect(LABEL_W + (k / bins) * span, mid - amp, Math.max(1, barW * stride - 0.5), amp * 2);
      }
      g2.globalAlpha = on ? 1 : 0.5;
      g2.fillStyle = on ? colors[s.id] : colors.label;
      g2.font = "600 11px " + getComputedStyle(document.body).fontFamily;
      g2.textBaseline = "middle";
      g2.fillText(s.label, 8, mid);
    });
    g2.globalAlpha = 1;
    // loop shading and its flags
    var a = drag ? Math.min(drag.from, drag.to) : state.loopA;
    var z = drag ? Math.max(drag.from, drag.to) : state.loopB;
    if (a !== null && z !== null && z > a) {
      var xa = stepToX(a, w);
      var xz = stepToX(z, w);
      g2.globalAlpha = state.looping || drag ? 0.16 : 0.08;
      g2.fillStyle = colors.loop;
      g2.fillRect(xa, 0, xz - xa, h);
      g2.globalAlpha = 1;
      g2.fillRect(xa, 0, 2, h);
      g2.fillRect(xz - 2, 0, 2, h);
      g2.font = "700 10px " + getComputedStyle(document.body).fontFamily;
      g2.textBaseline = "top";
      g2.fillRect(xa, 0, 16, 14);
      g2.fillRect(xz - 16, 0, 16, 14);
      g2.fillStyle = colors.bg;
      g2.fillText("A", xa + 4, 2);
      g2.fillText("B", xz - 12, 2);
    }
    // playhead
    g2.fillStyle = colors.head;
    g2.fillRect(Math.round(stepToX(pos % STEPS, w)) - 1, 0, 2, h);
    updateReadouts(pos);
  }

  function frame() {
    draw();
    if (state.playing) { requestAnimationFrame(frame); }
  }

  // click to jump, drag to loop
  var drag = null;
  function localX(ev) {
    var r = canvas.getBoundingClientRect();
    return ev.clientX - r.left;
  }
  canvas.addEventListener("pointerdown", function (ev) {
    var step = xToStep(localX(ev), canvas.clientWidth);
    drag = { from: step, to: step, x: localX(ev) };
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener("pointermove", function (ev) {
    if (!drag) { return; }
    drag.to = xToStep(localX(ev), canvas.clientWidth);
    if (!state.playing) { draw(); }
  });
  canvas.addEventListener("pointerup", function (ev) {
    if (!drag) { return; }
    drag.to = xToStep(localX(ev), canvas.clientWidth);
    var moved = Math.abs(localX(ev) - drag.x) > 6;
    if (moved) {
      // snap the loop to beats, at least one beat long
      var a = Math.round(Math.min(drag.from, drag.to) / 4) * 4;
      var z = Math.round(Math.max(drag.from, drag.to) / 4) * 4;
      if (z <= a) { z = a + 4; }
      state.loopA = Math.min(a, STEPS - 4);
      state.loopB = Math.min(z, STEPS);
      setLooping(true);
      drag = null;
      if (state.trainer) { setSpeed(0.75); }
      seek(state.loopA);
    } else {
      drag = null;
      seek(Math.floor(xToStep(localX(ev), canvas.clientWidth)));
    }
  });

  // ---- controls -----------------------------------------------------------
  var stemsBox = document.getElementById("stems");
  STEMS.forEach(function (s) {
    var name = document.createElement("span");
    name.className = "name";
    name.textContent = s.label;
    name.style.color = "var(--stem-" + s.id + ")";
    var mute = document.createElement("button");
    mute.type = "button";
    mute.textContent = "Mute";
    mute.setAttribute("aria-pressed", "false");
    mute.setAttribute("aria-label", "Mute " + s.label);
    var solo = document.createElement("button");
    solo.type = "button";
    solo.textContent = "Solo";
    solo.setAttribute("aria-pressed", "false");
    solo.setAttribute("aria-label", "Solo " + s.label);
    var vol = document.createElement("input");
    vol.type = "range";
    vol.min = "0";
    vol.max = "1";
    vol.step = "0.01";
    vol.value = String(state.volume[s.id]);
    vol.setAttribute("aria-label", s.label + " volume");
    mute.addEventListener("click", function () {
      state.mute[s.id] = !state.mute[s.id];
      mute.setAttribute("aria-pressed", String(state.mute[s.id]));
      applyGains();
      draw();
    });
    solo.addEventListener("click", function () {
      state.solo[s.id] = !state.solo[s.id];
      solo.setAttribute("aria-pressed", String(state.solo[s.id]));
      applyGains();
      draw();
    });
    vol.addEventListener("input", function () {
      state.volume[s.id] = parseFloat(vol.value);
      applyGains();
      draw();
    });
    [name, mute, solo, vol].forEach(function (el) { stemsBox.appendChild(el); });
  });

  function control(act) {
    return cockpit.querySelector('[data-act="' + act + '"]');
  }

  var playButton = control("play");
  playButton.setAttribute("aria-pressed", "false");
  playButton.addEventListener("click", function () {
    if (state.playing) { stop(); } else { start(); }
  });

  var speedSelect = control("speed");
  function setSpeed(value) {
    state.speed = value;
    var best = null;
    Array.prototype.forEach.call(speedSelect.options, function (o) {
      if (Math.abs(parseFloat(o.value) - value) < 0.001) { best = o; }
    });
    if (!best) {
      best = document.createElement("option");
      best.value = String(value);
      best.textContent = value.toFixed(2) + "x";
      best.setAttribute("data-trainer", "");
      speedSelect.appendChild(best);
    }
    speedSelect.value = best.value;
  }
  speedSelect.addEventListener("change", function () {
    state.speed = parseFloat(speedSelect.value);
  });

  control("pitch").addEventListener("change", function (ev) {
    state.pitch = parseInt(ev.target.value, 10);
    draw();
  });

  control("click").addEventListener("change", function (ev) {
    state.click = ev.target.checked;
  });

  var loopButton = control("loop");
  function setLooping(on) {
    state.looping = on && state.loopA !== null;
    loopButton.setAttribute("aria-pressed", String(state.looping));
  }
  loopButton.addEventListener("click", function () {
    if (state.loopA === null) {
      // no loop yet: take the bar under the playhead
      var bar = Math.floor(playhead() / STEPS_PER_BAR) % BARS;
      state.loopA = bar * STEPS_PER_BAR;
      state.loopB = state.loopA + STEPS_PER_BAR;
    }
    setLooping(!state.looping);
    draw();
  });

  control("clear").addEventListener("click", function () {
    state.loopA = null;
    state.loopB = null;
    setLooping(false);
    draw();
  });

  control("trainer").addEventListener("change", function (ev) {
    state.trainer = ev.target.checked;
    if (state.trainer) { setSpeed(0.75); }
  });

  // the theme toggle repaints the lanes in the other palette
  document.addEventListener("stemma-theme", function () {
    readColors();
    draw();
  });
  window.addEventListener("resize", function () {
    if (!state.playing) { draw(); }
  });
  document.addEventListener("visibilitychange", function () {
    if (document.hidden && state.playing) { stop(); }
  });

  readColors();
  draw();
})();
