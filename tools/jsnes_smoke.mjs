/* Smoke test for jsnes (ESM).  Boots Adventure Island, dumps a few frames as
 * PPM and counts APU audio samples so we know an offline, audio-capable
 * replay is possible.
 *
 *   node tools/jsnes_smoke.mjs <rom> <outdir>
 */
import fs from "node:fs";
import path from "node:path";

const JSNES = "file:///C:/Users/Administrator/.workbuddy/binaries/node/workspace/node_modules/jsnes/src";
const { default: NES } = await import(`${JSNES}/nes.js`);
const { default: Controller } = await import(`${JSNES}/controller.js`);

const ROM = process.argv[2] || "roms/Adventure Island (USA).nes";
const OUT = process.argv[3] || "outputs/jsnes_smoke";
fs.mkdirSync(OUT, { recursive: true });

let samples = 0;
let peak = 0;
const nes = new NES({
  onAudioSample: (l, r) => {
    samples++;
    peak = Math.max(peak, Math.abs(l), Math.abs(r));
  },
  sampleRate: 44100,
  emulateSound: true,
});

nes.loadROM(fs.readFileSync(ROM));

function dumpPPM(file) {
  const buf = nes.ppu.buffer;
  const w = 256,
    h = 240;
  const px = Buffer.alloc(w * h * 3);
  for (let i = 0; i < w * h; i++) {
    const v = buf[i];
    px[i * 3] = (v >> 16) & 0xff;
    px[i * 3 + 1] = (v >> 8) & 0xff;
    px[i * 3 + 2] = v & 0xff;
  }
  fs.writeFileSync(file, Buffer.concat([Buffer.from(`P6\n${w} ${h}\n255\n`, "ascii"), px]));
  console.log(`  ppm ${path.basename(file)}`);
}

// ctrl bitmask matches cynes: A=1 B=2 SELECT=4 START=8 UP=16 DOWN=32 LEFT=64 RIGHT=128
const BUTTONS = {
  1: Controller.BUTTON_A,
  2: Controller.BUTTON_B,
  4: Controller.BUTTON_SELECT,
  8: Controller.BUTTON_START,
  16: Controller.BUTTON_UP,
  32: Controller.BUTTON_DOWN,
  64: Controller.BUTTON_LEFT,
  128: Controller.BUTTON_RIGHT,
};

function setCtrl(mask) {
  for (const bit of Object.keys(BUTTONS)) {
    const b = parseInt(bit, 10);
    if (mask & b) nes.buttonDown(1, BUTTONS[b]);
    else nes.buttonUp(1, BUTTONS[b]);
  }
}

const run = (n, mask) => {
  setCtrl(mask);
  for (let i = 0; i < n; i++) nes.frame();
};

console.log("booting ...");
run(300, 0);
dumpPPM(path.join(OUT, "f_300.ppm"));
run(60, 8);
dumpPPM(path.join(OUT, "f_360.ppm"));
run(60, 0);
dumpPPM(path.join(OUT, "f_420.ppm"));
const before = samples;
run(120, 128);
dumpPPM(path.join(OUT, "f_540.ppm"));

console.log(`audio samples: ${samples} total, ${samples - before} in 120 frames`);
console.log(`  -> ${((samples - before) / 120).toFixed(2)} samples/frame @44100`);
console.log(`peak amplitude: ${peak.toFixed(4)}`);
