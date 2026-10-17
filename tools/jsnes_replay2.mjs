/* Replay a logged run from POWER-ON and capture the NES APU output.
 *
 * Unlike tools/jsnes_replay.mjs (which replays a fixed boot sequence and
 * then the inputs) this version takes ONE controller byte per game frame
 * from power-on, plus a list of RAM pokes with the frame range they apply
 * to.  That is what makes the $0038 stage warp reproducible: cynes hammers
 * the round register while START is held, and jsnes has to do the same or
 * it boots into 1-1 instead of 1-2.
 *
 *   node tools/jsnes_replay2.mjs <rom> <inputs.bin> <meta.json> <outdir>
 *
 * meta.json fields:
 *   inputs_from_power_on  true
 *   pokes                 [{from, to, addr, value}]   half-open frame range
 *   axe_frame             frame index whose step must see $006D = 1
 *   audio_start_frame     first frame whose audio belongs to the video
 *   video_frames          how many video frames to expect
 *   sample_rate, fps_nes
 *
 * Outputs: <outdir>/game_audio.wav (44.1 kHz stereo, video aligned)
 *          <outdir>/replay_frames_jsnes/f%06d.png  (sync check)
 *          <outdir>/replay_progress_jsnes.json
 */
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

const JSNES =
  "file:///C:/Users/Administrator/.workbuddy/binaries/node/workspace/node_modules/jsnes/src";
const { default: NES } = await import(`${JSNES}/nes.js`);
const { default: Controller } = await import(`${JSNES}/controller.js`);

const [romPath, inputsPath, metaPath, outDir] = process.argv.slice(2, 6);
if (!romPath || !inputsPath || !metaPath || !outDir) {
  console.error("usage: node tools/jsnes_replay2.mjs <rom> <inputs.bin> <meta.json> <outdir>");
  process.exit(1);
}
const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));
const inputs = fs.readFileSync(inputsPath);
fs.mkdirSync(outDir, { recursive: true });
const dumpDir = path.join(outDir, "replay_frames_jsnes");
fs.mkdirSync(dumpDir, { recursive: true });
const dumpEvery = meta.dump_every || 30;

const estSamples =
  Math.ceil(meta.video_frames * (meta.sample_rate / meta.fps_nes) * 1.05) + 8192;
const left = new Float32Array(estSamples);
const right = new Float32Array(estSamples);
let nSamples = 0;
let capturing = false;

const nes = new NES({
  onAudioSample: (l, r) => {
    if (capturing && nSamples < estSamples) {
      left[nSamples] = l;
      right[nSamples] = r;
      nSamples++;
    }
  },
  sampleRate: meta.sample_rate,
  emulateSound: true,
});
nes.loadROM(fs.readFileSync(romPath));

// cynes logs a NON-standard bit layout (RIGHT=0x01 ... A=0x80).
const BUTTONS = {
  1: Controller.BUTTON_RIGHT,
  2: Controller.BUTTON_LEFT,
  4: Controller.BUTTON_DOWN,
  8: Controller.BUTTON_UP,
  16: Controller.BUTTON_START,
  32: Controller.BUTTON_SELECT,
  64: Controller.BUTTON_B,
  128: Controller.BUTTON_A,
};
function setCtrl(mask) {
  for (const bit of Object.keys(BUTTONS)) {
    const b = parseInt(bit, 10);
    if (mask & b) nes.buttonDown(1, BUTTONS[b]);
    else nes.buttonUp(1, BUTTONS[b]);
  }
}

const pokes = meta.pokes || [];
function applyPokes(i) {
  for (const p of pokes) {
    if (i >= p.from && i < p.to) nes.cpu.mem[p.addr] = p.value;
  }
}

// minimal PNG writer (RGBA in, truecolor out)
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}
function dumpPNG(file, buf) {
  const w = 256, h = 240;
  const raw = Buffer.alloc(h * (1 + w * 3));
  for (let y = 0; y < h; y++) {
    const row = y * (1 + w * 3);
    raw[row] = 0;
    for (let x = 0; x < w; x++) {
      const v = buf[(y << 8) + x];
      raw[row + 1 + x * 3] = v & 0xff;
      raw[row + 1 + x * 3 + 1] = (v >> 8) & 0xff;
      raw[row + 1 + x * 3 + 2] = (v >> 16) & 0xff;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  fs.writeFileSync(
    file,
    Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      chunk("IHDR", ihdr),
      chunk("IDAT", zlib.deflateSync(raw, { level: 6 })),
      chunk("IEND", Buffer.alloc(0)),
    ]),
  );
}

const progress = [];
let videoFrame = -1;
const total = inputs.length;
for (let i = 0; i < total; i++) {
  applyPokes(i);
  if (i === meta.axe_frame) nes.cpu.mem[0x6d] = 1; // start_with_axe
  setCtrl(inputs[i]);
  if (!capturing && i >= meta.audio_start_frame) capturing = true;
  nes.frame();
  if (capturing) {
    videoFrame++;
    if (videoFrame % dumpEvery === 0) {
      dumpPNG(
        path.join(dumpDir, `f${String(videoFrame).padStart(6, "0")}.png`),
        nes.ppu.buffer,
      );
      progress.push({
        frame: videoFrame,
        camera: nes.cpu.mem[0x593],
        lives: nes.cpu.mem[0x3f],
        round: nes.cpu.mem[0x38],
      });
    }
    if (videoFrame % 600 === 0) {
      console.log(`  frame ${videoFrame}/${meta.video_frames}`);
    }
  }
  if (i % 1000 === 0) console.log(`  game frame ${i}/${total}`);
}

const nFrames = nSamples;
const dataBytes = nFrames * 4;
const wav = Buffer.alloc(44 + dataBytes);
wav.write("RIFF", 0);
wav.writeUInt32LE(36 + dataBytes, 4);
wav.write("WAVE", 8);
wav.write("fmt ", 12);
wav.writeUInt32LE(16, 16);
wav.writeUInt16LE(1, 20);
wav.writeUInt16LE(2, 22);
wav.writeUInt32LE(meta.sample_rate, 24);
wav.writeUInt32LE(meta.sample_rate * 4, 28);
wav.writeUInt16LE(4, 32);
wav.writeUInt16LE(16, 34);
wav.write("data", 36);
wav.writeUInt32LE(dataBytes, 40);
for (let i = 0; i < nFrames; i++) {
  const l = Math.max(-1, Math.min(1, left[i]));
  const r = Math.max(-1, Math.min(1, right[i]));
  wav.writeInt16LE(Math.round(l * 32767), 44 + i * 4);
  wav.writeInt16LE(Math.round(r * 32767), 44 + i * 4 + 2);
}
fs.writeFileSync(path.join(outDir, "game_audio.wav"), wav);
fs.writeFileSync(
  path.join(outDir, "replay_progress_jsnes.json"),
  JSON.stringify(progress, null, 1),
);

console.log(`replayed ${total} game frames, ${videoFrame + 1} video frames`);
console.log(`audio: ${(nFrames / meta.sample_rate).toFixed(2)}s`);
console.log(`wav  -> ${path.join(outDir, "game_audio.wav")}`);
