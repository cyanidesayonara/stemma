/* The "Try the mixer" demo: free-to-use songs, each separated into stems by
   stemma itself, played with the app's practice controls (mute, solo,
   volume, A-B loop, Loop Trainer, speed, pitch, metronome) over its
   stacked, stem-coloured waveform lanes.

   The audio runs in stretch-worklet.js, which mixes the stems and changes
   speed and pitch live the way the app does. The tempo, key, beats and
   chords come from the app's own detector, run on each clip by
   scripts/make_demo_songs.py and stored in demo/songs.json. A song may
   have any of the six stems (a 4-stem song has vocals, drums, bass and
   other); the lanes and the mixer rows follow the song. */
(function () {
  "use strict";

  // the app's stems, in its order (src/ui/main_window.py ALL_STEM_NAMES)
  var STEMS = [
    { id: "vocals", label: "Vocals" },
    { id: "drums", label: "Drums" },
    { id: "bass", label: "Bass" },
    { id: "other", label: "Other" },
    { id: "guitar", label: "Guitar" },
    { id: "piano", label: "Piano" }
  ];
  // the app's speeds (src/player.py SPEED_PRESETS); the Loop Trainer
  // steps through them, one per repeat, up to 1.0x
  var SPEED_PRESETS = [0.5, 0.75, 0.85, 1.0, 1.25, 1.5, 2.0];
  var TRAINER_START = 0.75;
  var SEEK_STEP = 5;   // seconds, as the app's Left and Right keys

  // the stems a song has, in the app's order
  function songStems(song) {
    var have = (song && song.stems) || [];
    return STEMS.filter(function (s) { return have.indexOf(s.id) >= 0; });
  }

  // the next speed the Trainer steps to, or null at 1.0x
  function nextTrainerSpeed(speed) {
    for (var i = 0; i < SPEED_PRESETS.length; i++) {
      var p = SPEED_PRESETS[i];
      if (p > speed + 1e-6 && p <= 1 + 1e-6) { return p; }
    }
    return null;
  }

  // for tests/test_site.py, which runs this file in Node
  window.stemmaDemo = { STEMS: STEMS, SPEED_PRESETS: SPEED_PRESETS,
    songStems: songStems, nextTrainerSpeed: nextTrainerSpeed };

  var cockpit = document.getElementById("cockpit");
  var canvas = document.getElementById("lanes");
  var picker = document.getElementById("songs");
  var credit = document.getElementById("credit");
  if (!cockpit || !canvas || !canvas.getContext || !picker) { return; }

  var AudioCtx = window.AudioContext || window.webkitAudioContext;
  var NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A",
    "Bb", "B"];

  var state = {
    songs: [],
    song: null,       // the chosen entry from songs.json
    stems: [],        // its stems, in the app's order
    loaded: null,     // id of the song whose audio the engine holds
    pick: 0,          // counts song picks; a load or play started for an
                      // earlier pick is stale and changes nothing
    loading: null,    // {pick, promise} while the song is being decoded
    gen: 0,           // which load that is; the engine echoes it back
    wantPlay: false,  // what the play button shows
    playing: false,   // the engine is playing
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
  var engineReady = null;

  function readVar(name) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
  }

  function audible(id) {
    var anySolo = state.stems.some(function (s) { return state.solo[s.id]; });
    if (anySolo) { return state.solo[id]; }
    return !state.mute[id];
  }

  function send(message, transfer) {
    if (node) { node.port.postMessage(message, transfer || []); }
  }

  function sendGains() {
    var gains = {};
    state.stems.forEach(function (s) {
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

  // ---- the engine ---------------------------------------------------------
  function setNote(text) {
    readouts.status.textContent = text;
  }

  // Create the audio context at once (inside the click that asked for it,
  // so iOS lets it play), and load the worklet.
  function ensureEngine() {
    if (engineReady) { return engineReady; }
    if (!AudioCtx || !window.AudioWorkletNode) {
      return Promise.reject(new Error("no worklet"));
    }
    // iOS: play as media, so the demo is heard with the ringer muted
    try {
      if (navigator.audioSession) { navigator.audioSession.type = "playback"; }
    } catch (e) {}
    ctx = new AudioCtx();
    engineReady = ctx.audioWorklet.addModule("stretch-worklet.js")
      .then(function () {
        node = new AudioWorkletNode(ctx, "stem-player", {
          numberOfInputs: 0, outputChannelCount: [2] });
        var master = ctx.createGain();
        master.gain.value = 0.9;
        node.connect(master);
        master.connect(ctx.destination);
        node.port.onmessage = onEngineMessage;
      });
    return engineReady;
  }

  function onEngineMessage(ev) {
    var m = ev.data;
    if (m.type === "stopped") {
      // a pause has faded out: let the audio thread rest until the next
      // play (not when this is the song before a switch, or play was
      // pressed again meanwhile)
      if (!state.wantPlay) { suspend(); }
      return;
    }
    // reports about an earlier load (another song, still fading) do not
    // move this one
    if (m.gen !== state.gen || state.loaded !== state.song.id) { return; }
    if (m.type === "pos") {
      state.pos = m.frame / ctx.sampleRate;
      state.posAt = performance.now();
      if (!state.playing) { draw(); }
    } else if (m.type === "wrap") {
      onWrap();
    } else if (m.type === "ended") {
      state.wantPlay = false;
      state.playing = false;
      state.pos = 0;
      showPlaying();
      draw();
      suspend();
    }
  }

  function suspend() {
    if (ctx && ctx.state !== "closed") { ctx.suspend(); }
  }

  // Fetch and decode a song's stems (the MP3s are HTTP-cached, so a song
  // decoded again later costs little). The page keeps no copy: the
  // samples move to the worklet.
  function decode(song) {
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
        // slices: the worklet takes ownership of these copies
        var left = b.getChannelData(0).slice(0, length);
        var right = (b.numberOfChannels > 1 ? b.getChannelData(1) :
          b.getChannelData(0)).slice(0, length);
        stems[song.stems[i]] = [left, right];
      });
      return { stems: stems, length: length };
    });
  }

  // Decode the chosen song and hand its audio to the engine, transferred
  // rather than copied. Asking twice during one pick shares one load; a
  // load for an earlier pick sends nothing and resolves "stale".
  function load(song) {
    var pick = state.pick;
    if (state.loading && state.loading.pick === pick) {
      return state.loading.promise;
    }
    setNote("Loading " + song.title + "...");
    var promise = ensureEngine().then(function () {
      return decode(song);
    }).then(function (audio) {
      if (state.pick !== pick) { return "stale"; }
      var sr = ctx.sampleRate;
      var transfer = [];
      Object.keys(audio.stems).forEach(function (id) {
        transfer.push(audio.stems[id][0].buffer, audio.stems[id][1].buffer);
      });
      state.gen += 1;
      // the engine holds this back until a playing song has faded out
      send({ type: "load", gen: state.gen, stems: audio.stems,
        length: audio.length,
        beats: song.beats.map(function (t) { return Math.round(t * sr); }),
        downbeats: song.downbeats.map(function (t) {
          return Math.round(t * sr);
        }) }, transfer);
      sendGains();
      sendParams();
      sendLoop();
      send({ type: "click", on: state.click });
      state.loaded = song.id;
      setNote("");
      return true;
    }).catch(function () {
      if (state.pick !== pick) { return "stale"; }
      setNote(window.AudioWorkletNode ? "The song could not be loaded." :
        "This browser cannot play the demo.");
      return false;
    }).then(function (ok) {
      if (state.loading && state.loading.pick === pick) {
        state.loading = null;
      }
      return ok;
    });
    state.loading = { pick: pick, promise: promise };
    return promise;
  }

  // ---- transport ----------------------------------------------------------
  var playButton = control("play");

  function showPlaying() {
    playButton.setAttribute("aria-pressed", String(state.wantPlay));
    playButton.setAttribute("aria-label", state.wantPlay ? "Pause" : "Play");
  }

  // Called from a click: the context is created or resumed right here,
  // inside the user's gesture, before anything waits.
  function play() {
    var song = state.song;
    var pick = state.pick;
    state.wantPlay = true;
    showPlaying();
    var ready = state.loaded === song.id ? ensureEngine() : load(song);
    // always resume: a suspend from the last pause may still be pending
    var resumed = ctx ? ctx.resume() : Promise.resolve();
    Promise.all([ready, resumed]).then(function (done) {
      // another song was picked meanwhile: its own play is in charge
      if (state.pick !== pick || done[0] === "stale") { return; }
      if (done[0] === false) {
        state.wantPlay = false;
        showPlaying();
        return;
      }
      // paused meanwhile, or an earlier click already started it
      if (!state.wantPlay || state.playing) { return; }
      if (state.pos > 0) { seek(state.pos); }
      send({ type: "play" });
      state.playing = true;
      state.posAt = performance.now();
      requestAnimationFrame(frame);
    }, function () {
      if (state.pick !== pick) { return; }
      state.wantPlay = false;
      showPlaying();
      setNote("This browser cannot play the demo.");
    });
  }

  function pause() {
    state.pos = playhead();
    state.wantPlay = false;
    if (state.playing) {
      // the engine fades out, then says "stopped", and the context rests
      send({ type: "pause" });
    }
    state.playing = false;
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

  // the loop came round again: the Trainer steps up one preset
  function onWrap() {
    if (!state.trainer || !state.looping) { return; }
    var next = nextTrainerSpeed(state.speed);
    if (next !== null) { setSpeed(next); }
  }

  function pick(song) {
    var resume = state.wantPlay;
    if (resume) { pause(); }
    state.pick += 1;
    state.song = song;
    state.stems = songStems(song);
    state.loaded = null;
    state.pos = 0;
    state.loopA = null;
    state.loopB = null;
    setLooping(false);
    Array.prototype.forEach.call(picker.children, function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.id === song.id));
    });
    buildRows();
    credit.innerHTML = "";
    var text = document.createElement("span");
    text.textContent = "“" + song.title + "” by " +
      (song.credit_as || song.artist) + ", ";
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
      Math.round(song.duration) + "-second clip in " + state.stems.length +
      " stems, separated by stemma."));
    canvas.setAttribute("aria-valuemax", String(Math.round(song.duration)));
    draw();
    if (resume) { play(); }
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

  var spokenSecond = -1;
  function updateReadouts(pos) {
    var song = state.song;
    readouts.key.textContent = keyName(song.key);
    readouts.chord.textContent = chordAt(pos);
    readouts.tempo.textContent = song.bpm ?
      "~" + Math.round(song.bpm * state.speed) + " BPM" : "unknown";
    readouts.time.textContent = clock(pos) + " / " + clock(song.duration);
    readouts.loop.textContent = state.loopA === null ? "No loop set" :
      "A " + clock(state.loopA) + "  B " + clock(state.loopB);
    // the canvas is a position slider for assistive technology; update it
    // once a second, not every frame
    var second = Math.floor(pos);
    if (second !== spokenSecond) {
      spokenSecond = second;
      canvas.setAttribute("aria-valuenow", String(second));
      canvas.setAttribute("aria-valuetext",
        clock(pos) + " of " + clock(song.duration));
    }
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
    var lanes = state.stems;
    g2.clearRect(0, 0, w, h);
    var laneH = (h - 8) / Math.max(1, lanes.length);
    g2.fillStyle = colors.grid;
    song.downbeats.forEach(function (t) {
      g2.fillRect(Math.round(timeToX(t, w)), 0, 1, h);
    });
    var span = w - LABEL_W - 8;
    var font = getComputedStyle(document.body).fontFamily;
    lanes.forEach(function (s, i) {
      var peaks = (song.peaks && song.peaks[s.id]) || [];
      var mid = 4 + laneH * i + laneH / 2;
      var on = audible(s.id);
      g2.globalAlpha = on ? 1 : 0.22;
      g2.fillStyle = colors[s.id];
      var barW = span / Math.max(1, peaks.length);
      var scale = state.volume[s.id] / 0.8;
      for (var k = 0; k < peaks.length; k++) {
        var v = Math.min(1, peaks[k] / 99 * scale);
        var amp = Math.max(0.5, v * (laneH / 2 - 3));
        g2.fillRect(LABEL_W + k * barW, mid - amp, Math.max(1, barW - 0.6),
          amp * 2);
      }
      g2.globalAlpha = on ? 1 : 0.5;
      g2.fillStyle = on ? colors[s.id] : colors.label;
      g2.font = "600 11px " + font;
      g2.textBaseline = "middle";
      g2.fillText(s.label, 8, mid);
    });
    g2.globalAlpha = 1;
    var showDrag = drag && drag.active;
    var a = showDrag ? Math.min(drag.from, drag.to) : state.loopA;
    var z = showDrag ? Math.max(drag.from, drag.to) : state.loopB;
    if (a !== null && z !== null && z > a) {
      var xa = timeToX(a, w);
      var xz = timeToX(z, w);
      g2.globalAlpha = state.looping || showDrag ? 0.16 : 0.08;
      g2.fillStyle = colors.loop;
      g2.fillRect(xa, 0, xz - xa, h);
      g2.globalAlpha = 1;
      g2.fillRect(xa, 0, 2, h);
      g2.fillRect(xz - 2, 0, 2, h);
      g2.font = "700 10px " + font;
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

  // Tap or click to jump; drag sideways to loop. A vertical swipe is left
  // to the browser (touch-action: pan-y), so the page still scrolls.
  var drag = null;
  var SLOP = 8;
  function localX(ev) {
    return ev.clientX - canvas.getBoundingClientRect().left;
  }
  canvas.addEventListener("pointerdown", function (ev) {
    if (ev.button !== 0 || !state.song) { return; }
    var t = xToTime(localX(ev), canvas.clientWidth);
    drag = { from: t, to: t, x: ev.clientX, y: ev.clientY, active: false,
      moved: false };
    if (ev.pointerType === "mouse") { canvas.setPointerCapture(ev.pointerId); }
  });
  canvas.addEventListener("pointermove", function (ev) {
    if (!drag) { return; }
    var dx = Math.abs(ev.clientX - drag.x);
    var dy = Math.abs(ev.clientY - drag.y);
    if (!drag.active) {
      if (dx < SLOP && dy < SLOP) { return; }
      drag.moved = true;
      // mostly sideways: a loop; otherwise it is a scroll, not ours
      if (dx > dy * 1.5) {
        drag.active = true;
        canvas.setPointerCapture(ev.pointerId);
      } else {
        drag = null;
        return;
      }
    }
    drag.to = xToTime(localX(ev), canvas.clientWidth);
    if (!state.playing) { draw(); }
  });
  canvas.addEventListener("pointerup", function (ev) {
    if (!drag) { return; }
    var d = drag;
    drag = null;
    if (d.active) {
      d.to = xToTime(localX(ev), canvas.clientWidth);
      // snap the loop to the song's beats, at least a second long
      var a = nearestBeat(Math.min(d.from, d.to));
      var z = nearestBeat(Math.max(d.from, d.to));
      if (z - a < 1) { z = Math.min(state.song.duration, a + 2); }
      state.loopA = a;
      state.loopB = z;
      setLooping(true);
      if (state.trainer) { setSpeed(TRAINER_START); }
      seek(a);
    } else if (!d.moved) {
      seek(xToTime(localX(ev), canvas.clientWidth));
    }
  });
  canvas.addEventListener("pointercancel", function () {
    drag = null;
    draw();
  });

  // as the app: Left and Right (and Down and Up) move 5 s; Page Down and
  // Page Up move 15 s; Home and End go to the start and the end
  canvas.addEventListener("keydown", function (ev) {
    if (!state.song || ev.altKey || ev.ctrlKey || ev.metaKey) { return; }
    var here = playhead();
    if (ev.key === "ArrowLeft") {
      seek(here - SEEK_STEP);
    } else if (ev.key === "ArrowDown") {
      seek(here - SEEK_STEP);
    } else if (ev.key === "ArrowUp") {
      seek(here + SEEK_STEP);
    } else if (ev.key === "PageDown") {
      seek(here - 3 * SEEK_STEP);
    } else if (ev.key === "PageUp") {
      seek(here + 3 * SEEK_STEP);
    } else if (ev.key === "End") {
      seek(state.song.duration);
    } else if (ev.key === "ArrowRight") {
      seek(here + SEEK_STEP);
    } else if (ev.key === "Home") {
      seek(0);
    } else {
      return;
    }
    ev.preventDefault();
  });

  // ---- controls -----------------------------------------------------------
  function control(act) {
    return cockpit.querySelector('[data-act="' + act + '"]');
  }

  // one mixer row per stem the song has
  var stemsBox = document.getElementById("stems");
  function buildRows() {
    stemsBox.innerHTML = "";
    state.stems.forEach(function (s) {
      var name = document.createElement("span");
      name.className = "name";
      name.textContent = s.label;
      name.style.color = "var(--stem-" + s.id + ")";
      var mute = toggle("Mute", s, state.mute);
      var solo = toggle("Solo", s, state.solo);
      var vol = document.createElement("input");
      vol.type = "range";
      vol.min = "0";
      vol.max = "1";
      vol.step = "0.01";
      vol.value = String(state.volume[s.id]);
      vol.setAttribute("aria-label", s.label + " volume");
      vol.addEventListener("input", function () {
        state.volume[s.id] = parseFloat(vol.value);
        sendGains();
        draw();
      });
      [name, mute, solo, vol].forEach(function (el) {
        stemsBox.appendChild(el);
      });
    });
  }

  function toggle(word, stem, flags) {
    var b = document.createElement("button");
    b.type = "button";
    b.textContent = word;
    b.setAttribute("aria-pressed", String(flags[stem.id]));
    b.setAttribute("aria-label", word + " " + stem.label);
    b.addEventListener("click", function () {
      flags[stem.id] = !flags[stem.id];
      b.setAttribute("aria-pressed", String(flags[stem.id]));
      sendGains();
      draw();
    });
    return b;
  }

  playButton.setAttribute("aria-pressed", "false");
  playButton.addEventListener("click", function () {
    if (!state.song) { return; }
    if (state.wantPlay) { pause(); } else { play(); }
  });

  var speedSelect = control("speed");
  function setSpeed(value) {
    state.speed = value;
    Array.prototype.forEach.call(speedSelect.options, function (o) {
      if (Math.abs(parseFloat(o.value) - value) < 0.001) {
        speedSelect.value = o.value;
      }
    });
    sendParams();
    draw();
  }
  speedSelect.addEventListener("change", function () {
    setSpeed(parseFloat(speedSelect.value));
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
    if (!state.song) { return; }
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
    if (!state.song) { return; }
    if (state.trainer) {
      // the Trainer steps up on each repeat, so it needs a loop to repeat
      loopThisBar();
      setLooping(true);
      setSpeed(TRAINER_START);
    } else {
      setSpeed(1);
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
