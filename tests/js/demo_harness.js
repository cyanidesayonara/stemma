/* Runs site/demo.js outside a browser for tests/test_site.py.

   node demo_harness.js DEMO SONGS.json

   With no cockpit on the "page", demo.js only sets up its pure helpers
   (window.stemmaDemo) and stops. Prints, as JSON, the stem list, the speed
   presets, each song's lanes, a 4-stem song's lanes, and the Loop
   Trainer's steps from 0.75x. */
"use strict";

const fs = require("fs");

globalThis.window = {};
globalThis.document = { getElementById: () => null };
new Function(fs.readFileSync(process.argv[2], "utf8"))();
const demo = globalThis.window.stemmaDemo;
const songs = JSON.parse(fs.readFileSync(process.argv[3], "utf8")).songs;

const ids = (stems) => stems.map((s) => s.id);
const steps = [0.75];
for (let next = demo.nextTrainerSpeed(0.75); next !== null;
  next = demo.nextTrainerSpeed(next)) {
  steps.push(next);
}

process.stdout.write(JSON.stringify({
  stems: ids(demo.STEMS),
  presets: demo.SPEED_PRESETS,
  songs: Object.fromEntries(songs.map((s) => [s.id, ids(demo.songStems(s))])),
  fourStem: ids(demo.songStems({ stems: ["other", "bass", "vocals", "drums"] })),
  trainer: steps,
}));
