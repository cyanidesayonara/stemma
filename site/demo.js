/* The "Try the mixer" demo: free-to-use songs, each separated into six
   stems by stemma itself, played with the app's practice controls (mute,
   solo, volume, A-B loop, Loop Trainer, speed, pitch, metronome) over its
   stacked, stem-coloured waveform lanes.

   The audio runs in stretch-worklet.js, which mixes the stems and changes
   speed and pitch live the way the app does. The tempo, key, beats and
   chords come from the app's own detector, run on each clip by
   scripts/make_demo_songs.py and stored in demo/songs.json. */
(function () {
  "use strict";

  var cockpit = document.getElementById("cockpit");
  var canvas = document.getElementById("lanes");
  var picker = document.getElementById("songs");
  var credit = document.getElementById("credit");
  if (!cockpit || !canvas || !canvas.getContext || !picker) { return; }

  var AudioCtx = window.AudioContext || window.webkitAudioContext;
  var NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A",
    "Bb", "B"];
  var STEMS = [
    { id: "vocals", label: "Vocals" },
    { id: "drums", label: "Drums" },
    { id: "bass", label: "Bass" },
    { id: "other", label: "Other" },
    { id: "guitar", label: "Guitar" },
    { id: "piano", label: "Piano" }
  ];

  var state = {
    songs: [],
    song: null,       // the chosen entry from songs.json
    loaded: null,     // id of the song whose audio the engine holds
    loading: false,
    playing: false,
    speed: 1,
    pitch: 0,
    click: false,
    trainer: false,
    looping: false,
    loopA: null,      // seconds
    loopB: null,
    pos: 0,           // seconds, what the listener hears
    posAt: 0,         // performance.now() of the last report
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
  var node = null;
  var decoded = {};   // song id -> {stems, length}

  function readVar(name) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
  }

  function audible(id) {
    var anySolo = STEMS.some(function (s) { return state.solo[s.id]; });
    if (anySolo) { return state.solo[id]; }
    return !state.mute[id];
  }

  function send(message, transfer) {
    if (node) { node.port.postMessage(message, transfer || []); }
  }

  function sendGains() {
    var gains = {};
    STEMS.forEach(function (s) {
      gains[s.id] = audible(s.id) ? state.volume[s.id] : 0;
    });
    send({ type: "gains", gains: gains });
  }

  function sendLoop() {
    var sr = ctx ? ctx.sampleRate : 44100;
    send({ type: "loop", on: state.looping && state.loopA !== null,
      a: Math.round((state.loopA || 0) * sr),
      b: Math.round((state.loopB || 0) * sr) });
  }

  function sendParams() {
    send({ type: "params", speed: state.speed, semis: state.pitch });
  }

  // ---- loading ------------------------------------------------------------
  function setNote(text) {
    readouts.status.textContent = text;
  }

  function ensureEngine() {
    if (ctx) { return Promise.resolve(); }
    if (!AudioCtx || !window.AudioWorkletNode) {
      return Promise.reject(new Error("no worklet"));
    }
    ctx = new AudioCtx();
    return ctx.audioWorklet.addModule("stretch-worklet.js").then(function () {
      node = new AudioWorkletNode(ctx, "stem-player", {
        numberOfInputs: 0, outputChannelCount: [2] });
      var master = ctx.createGain();
      master.gain.value = 0.9;
      node.connect(master);
      master.connect(ctx.destination);
      node.port.onmessage = function (ev) {
        var m = ev.data;
        // reports about another song (still pausing) do not move this one
        if (m.type === "pos" && state.song && state.loaded === state.song.id) {
          state.pos = m.frame / ctx.sampleRate;
          state.posAt = performance.now();
        } else if (m.type === "wrap") {
          onWrap();
        }
      };
    });
  }

  function decode(song) {
    if (decoded[song.id]) { return Promise.resolve(decoded[song.id]); }
    return Promise.all(song.stems.map(function (id) {
      return fetch("demo/" + song.id + "/" + id + ".mp3")
        .then(function (r) {
          if (!r.ok) { throw new Error(r.status); }
          return r.arrayBuffer();
        })
        .then(function (bytes) { return ctx.decodeAudioData(bytes); });
    })).then(function (buffers) {
      var length = Math.min.apply(null, buffers.map(function (b) {
        return b.length;
      }));
      var stems = {};
      buffers.forEach(function (b, i) {
        var left = b.getChannelData(0).slice(0, length);
        var right = (b.numberOfChannels > 1 ? b.getChannelData(1) :
          b.getChannelData(0)).slice(0, length);
        stems[song.stems[i]] = [left, right];
      });
      decoded[song.id] = { stems: stems, length: length };
      return decoded[song.id];
    });
  }

  // Hand the chosen song's audio to the engine (copies, so a song can be
  // handed over again after switching back to it).
  function load(song) {
    state.loading = true;
    setNote("Loading " + song.title + "...");
    return ensureEngine().then(function () {
      return decode(song);
    }).then(function (audio) {
      if (state.song !== song) { return false; }
      var sr = ctx.sampleRate;
      var stems = {};
      var transfer = [];
      Object.keys(audio.stems).forEach(function (id) {
        var l = audio.stems[id][0].slice();
        var r = audio.stems[id][1].slice();
        stems[id] = [l, r];
        transfer.push(l.buffer, r.buffer);
      });
      sendGains();
      send({ type: "load", stems: stems, length: audio.length,
        beats: song.beats.map(function (t) { return Math.round(t * sr); }),
        downbeats: song.downbeats.map(function (t) {
          return Math.round(t * sr);
        }) }, transfer);
      sendParams();
      sendLoop();
      send({ type: "click", on: state.click });
      state.loaded = song.id;
      state.loading = false;
      setNote("");
      return true;
    }).catch(function () {
      state.loading = false;
      setNote(window.AudioWorkletNode ? "The song could not be loaded." :
        "This browser cannot play the demo.");
      return false;
    });
  }

  // ---- transport ----------------------------------------------------------
  var playButton = control("play");

  function showPlaying() {
    playButton.setAttribute("aria-pressed", String(state.playing));
    playButton.setAttribute("aria-label", state.playing ? "Pause" : "Play");
  }

  function play() {
    var song = state.song;
    var ready = state.loaded === song.id ? ensureEngine() : load(song);
    Promise.resolve(ready).then(function (ok) {
      if (ok === false || state.song !== song) { return; }
      if (ctx.state === "suspended") { ctx.resume(); }
      if (state.pos > 0) { seek(state.pos); }
      send({ type: "play" });
      state.playing = true;
      state.posAt = performance.now();
      showPlaying();
      requestAnimationFrame(frame);
    });
  }

  function pause() {
    send({ type: "pause" });
    state.playing = false;
    state.pos = playhead();
    showPlaying();
    draw();
  }

  function seek(seconds) {
    var dur = state.song.duration;
    state.pos = Math.max(0, Math.min(dur - 0.05, seconds));
    state.posAt = performance.now();
    if (ctx && state.loaded === state.song.id) {
      send({ type: "seek", frame: Math.round(state.pos * ctx.sampleRate) });
    }
    draw();
  }

  // the playhead between engine reports, run on at the playback speed
  function playhead() {
    if (!state.playing) { return state.pos; }
    var ahead = (performance.now() - state.posAt) / 1000 * state.speed;
    var p = state.pos + Math.min(ahead, 0.25);
    var end = state.looping && state.loopA !== null ? state.loopB :
      state.song.duration;
    return Math.min(p, end);
  }

  function onWrap() {
    if (state.trainer && state.looping && state.speed < 1) {
      setSpeed(Math.min(1, Math.round((state.speed + 0.05) * 100) / 100));
    }
  }

  function pick(song) {
    var wasPlaying = state.playing;
    if (state.playing) { pause(); }
    state.song = song;
    state.loaded = null;
    state.pos = 0;
    state.loopA = null;
    state.loopB = null;
    setLooping(false);
    Array.prototype.forEach.call(picker.children, function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.id === song.id));
    });
    credit.innerHTML = "";
    var text = document.createElement("span");
    text.textContent = "“" + song.title + "” by " + song.artist +
      ", ";
    var lic = document.createElement("a");
    lic.href = song.licence_url;
    lic.textContent = song.licence;
    var src = document.createElement("a");
    src.href = song.source;
    src.textContent = "source";
    credit.appendChild(text);
    credit.appendChild(lic);
    credit.appendChild(document.createTextNode(" ("));
    credit.appendChild(src);
    credit.appendChild(document.createTextNode("). A " +
      Math.round(song.duration) + "-second clip, separated by stemma."));
    draw();
    if (wasPlaying) { play(); }
  }

  // ---- readouts -----------------------------------------------------------
  function transposeName(root) {
    var i = NOTE_NAMES.indexOf(root);
    return i < 0 ? root : NOTE_NAMES[(i + state.pitch + 120) % 12];
  }

  function keyName(key) {
    if (!key) { return "unknown"; }
    var parts = key.split(" ");
    return transposeName(parts[0]) + (parts[1] ? " " + parts[1] : "");
  }

  function chordAt(t) {
    var chords = state.song.chords;
    var name = "";
    for (var i = 0; i < chords.length && chords[i][0] <= t; i++) {
      name = chords[i][1];
    }
    if (!name || name === "N") { return "–"; }
    var minor = name.slice(-1) === "m";
    return transposeName(minor ? name.slice(0, -1) : name) + (minor ? "m" : "");
  }

  function clock(t) {
    var s = Math.max(0, Math.floor(t));
    return Math.floor(s / 60) + ":" + ("0" + (s % 60)).slice(-2);
  }

  var readouts = {};
  cockpit.querySelectorAll("[data-readout]").forEach(function (el) {
    readouts[el.getAttribute("data-readout")] = el;
  });

  function updateReadouts(pos) {
    var song = state.song;
    readouts.key.textContent = keyName(song.key);
    readouts.chord.textContent = chordAt(pos);
    readouts.tempo.textContent = song.bpm ?
      "~" + Math.round(song.bpm * state.speed) + " BPM" : "unknown";
    readouts.time.textContent = clock(pos) + " / " + clock(song.duration);
    readouts.loop.textContent = state.loopA === null ? "No loop set" :
      "A " + clock(state.loopA) + "  B " + clock(state.loopB);
  }

  // ---- the lanes ----------------------------------------------------------
  var g2 = canvas.getContext("2d");
  var colors = {};
  var LABEL_W = 58;

  function readColors() {
    STEMS.forEach(function (s) { colors[s.id] = readVar("--stem-" + s.id); });
    colors.grid = readVar("--surface0");
    colors.label = readVar("--subtext");
    colors.head = readVar("--text");
    colors.loop = readVar("--recording");
    colors.bg = readVar("--mantle");
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

  function timeToX(t, w) {
    return LABEL_W + (t / state.song.duration) * (w - LABEL_W - 8);
  }

  function xToTime(x, w) {
    var t = ((x - LABEL_W) / (w - LABEL_W - 8)) * state.song.duration;
    return Math.max(0, Math.min(state.song.duration, t));
  }

  function nearestBeat(t) {
    var beats = state.song.beats;
    if (!beats.length) { return t; }
    var best = beats[0];
    beats.forEach(function (b) {
      if (Math.abs(b - t) < Math.abs(best - t)) { best = b; }
    });
    return best;
  }

  function draw() {
    if (!state.song) { return; }
    var dims = size();
    var w = dims.w;
    var h = dims.h;
    var pos = playhead();
    var song = state.song;
    g2.clearRect(0, 0, w, h);
    var laneH = (h - 8) / STEMS.length;
    g2.fillStyle = colors.grid;
    song.downbeats.forEach(function (t) {
      g2.fillRect(Math.round(timeToX(t, w)), 0, 1, h);
    });
    var span = w - LABEL_W - 8;
    STEMS.forEach(function (s, i) {
      var peaks = song.peaks[s.id] || [];
      var mid = 4 + laneH * i + laneH / 2;
      var on = audible(s.id);
      g2.globalAlpha = on ? 1 : 0.22;
      g2.fillStyle = colors[s.id];
      var barW = span / peaks.length;
      var scale = state.volume[s.id] / 0.8;
      for (var k = 0; k < peaks.length; k++) {
        var v = Math.min(1, peaks[k] / 99 * scale);
        var amp = Math.max(0.5, v * (laneH / 2 - 3));
        g2.fillRect(LABEL_W + k * barW, mid - amp, Math.max(1, barW - 0.6),
          amp * 2);
      }
      g2.globalAlpha = on ? 1 : 0.5;
      g2.fillStyle = on ? colors[s.id] : colors.label;
      g2.font = "600 11px " + getComputedStyle(document.body).fontFamily;
      g2.textBaseline = "middle";
      g2.fillText(s.label, 8, mid);
    });
    g2.globalAlpha = 1;
    var a = drag ? Math.min(drag.from, drag.to) : state.loopA;
    var z = drag ? Math.max(drag.from, drag.to) : state.loopB;
    if (a !== null && z !== null && z > a) {
      var xa = timeToX(a, w);
      var xz = timeToX(z, w);
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
    g2.fillStyle = colors.head;
    g2.fillRect(Math.round(timeToX(pos, w)) - 1, 0, 2, h);
    updateReadouts(pos);
  }

  function frame() {
    draw();
    if (state.playing) { requestAnimationFrame(frame); }
  }

  // click to jump, drag to loop
  var drag = null;
  function localX(ev) {
    return ev.clientX - canvas.getBoundingClientRect().left;
  }
  canvas.addEventListener("pointerdown", function (ev) {
    if (ev.button !== 0 || !state.song) { return; }
    var t = xToTime(localX(ev), canvas.clientWidth);
    drag = { from: t, to: t, x: localX(ev) };
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener("pointermove", function (ev) {
    if (!drag) { return; }
    drag.to = xToTime(localX(ev), canvas.clientWidth);
    if (!state.playing) { draw(); }
  });
  canvas.addEventListener("pointerup", function (ev) {
    if (!drag) { return; }
    drag.to = xToTime(localX(ev), canvas.clientWidth);
    var moved = Math.abs(localX(ev) - drag.x) > 6;
    var from = Math.min(drag.from, drag.to);
    var to = Math.max(drag.from, drag.to);
    drag = null;
    if (moved) {
      // snap the loop to the song's beats, at least a second long
      var a = nearestBeat(from);
      var z = nearestBeat(to);
      if (z - a < 1) { z = Math.min(state.song.duration, a + 2); }
      state.loopA = a;
      state.loopB = z;
      setLooping(true);
      if (state.trainer) { setSpeed(0.75); }
      seek(a);
    } else {
      seek(xToTime(localX(ev), canvas.clientWidth));
    }
  });
  canvas.addEventListener("pointercancel", function () {
    drag = null;
    draw();
  });

  // ---- controls -----------------------------------------------------------
  function control(act) {
    return cockpit.querySelector('[data-act="' + act + '"]');
  }

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
      sendGains();
      draw();
    });
    solo.addEventListener("click", function () {
      state.solo[s.id] = !state.solo[s.id];
      solo.setAttribute("aria-pressed", String(state.solo[s.id]));
      sendGains();
      draw();
    });
    vol.addEventListener("input", function () {
      state.volume[s.id] = parseFloat(vol.value);
      sendGains();
      draw();
    });
    [name, mute, solo, vol].forEach(function (el) { stemsBox.appendChild(el); });
  });

  playButton.setAttribute("aria-pressed", "false");
  playButton.addEventListener("click", function () {
    if (state.playing) { pause(); } else { play(); }
  });

  var speedSelect = control("speed");
  function setSpeed(value) {
    state.speed = value;
    var best = null;
    Array.prototype.forEach.call(speedSelect.options, function (o) {
      if (Math.abs(parseFloat(o.value) - value) < 0.001) { best = o; }
    });
    if (!best) {
      // a speed the list lacks: add it in order, labelled like the rest
      best = document.createElement("option");
      best.value = String(value);
      best.textContent = value.toFixed(2) + "x";
      var after = null;
      Array.prototype.forEach.call(speedSelect.options, function (o) {
        if (!after && parseFloat(o.value) > value) { after = o; }
      });
      speedSelect.insertBefore(best, after);
    }
    speedSelect.value = best.value;
    sendParams();
  }
  speedSelect.addEventListener("change", function () {
    state.speed = parseFloat(speedSelect.value);
    sendParams();
  });

  control("pitch").addEventListener("change", function (ev) {
    state.pitch = parseInt(ev.target.value, 10);
    sendParams();
    draw();
  });

  control("click").addEventListener("change", function (ev) {
    state.click = ev.target.checked;
    send({ type: "click", on: state.click });
  });

  var loopButton = control("loop");
  function setLooping(on) {
    state.looping = on && state.loopA !== null;
    loopButton.setAttribute("aria-pressed", String(state.looping));
    sendLoop();
  }

  // with no loop yet, loop the bar under the playhead
  function loopThisBar() {
    if (state.loopA !== null) { return; }
    var song = state.song;
    var t = playhead();
    var bars = song.downbeats.length ? song.downbeats : [0];
    var a = 0;
    bars.forEach(function (d) { if (d <= t) { a = d; } });
    var next = bars.filter(function (d) { return d > a + 0.1; });
    // two bars, so there is something to drill
    var z = next.length > 1 ? next[1] : next.length ? next[0] :
      Math.min(song.duration, a + 4);
    state.loopA = a;
    state.loopB = z;
  }

  loopButton.addEventListener("click", function () {
    loopThisBar();
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
    if (state.trainer) {
      // the Trainer steps up on each repeat, so it needs a loop to repeat
      loopThisBar();
      setLooping(true);
      setSpeed(0.75);
      draw();
    }
  });

  document.addEventListener("stemma-theme", function () {
    readColors();
    draw();
  });
  window.addEventListener("resize", function () {
    if (!state.playing) { draw(); }
  });

  // ---- the songs ----------------------------------------------------------
  readColors();
  fetch("demo/songs.json").then(function (r) { return r.json(); })
    .then(function (data) {
      state.songs = data.songs;
      data.songs.forEach(function (song) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "song";
        b.dataset.id = song.id;
        b.setAttribute("aria-pressed", "false");
        var genre = document.createElement("span");
        genre.className = "genre";
        genre.textContent = song.genre;
        var title = document.createElement("span");
        title.className = "title";
        title.textContent = song.title;
        b.appendChild(genre);
        b.appendChild(title);
        b.addEventListener("click", function () {
          if (state.song !== song) { pick(song); }
        });
        picker.appendChild(b);
      });
      pick(data.songs[0]);
    })
    .catch(function () { setNote("The songs could not be loaded."); });
})();
