/* Replay a logged Adventure Island run (controller bytes recorded by
 * tools/record_clear_sound.py) in jsnes and capture the NES APU output,
 * so the annotated cyNES recording can be given its real soundtrack.
 *
 *   node tools/jsnes_replay.mjs <rom> <inputs.bin> <meta.json> <outdir> [dumpEvery]
 *
 * Outputs:
 *   <outdir>/game_audio.wav            44.1 kHz stereo PCM, video-aligned
 *   <outdir>/replay_frames_jsnes/*.ppm raw frames, same cadence as the
 *                                      recording, for a pixel-level sync check
 *   <outdir>/replay_progress_jsnes.json camera/lives sampled per dump
 */
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

const JSNES =
  "file:///C:/Users/Administrator/.workbuddy/binaries/node/workspace/node_modules/jsnes/src";
const { default: NES } = await import(`${JSNES}/nes.js`);
const { default: Controller } = await import(`${JSNES}/controller.js`);

const [romPath, inputsPath, metaPath, outDir] = process.argv.slice(2, 6);
const dumpEvery = parseInt(process.argv[6] || "30", 10);
if (!romPath || !inputsPath || !metaPath || !outDir) {
  console.error("usage: node tools/jsnes_replay.mjs <rom> <inputs.bin> <meta.json> <outdir> [dumpEvery]");
  process.exit(1);
}
const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));
const inputs = fs.readFileSync(inputsPath);
fs.mkdirSync(outDir, { recursive: true });
const dumpDir = path.join(outDir, "replay_frames_jsnes");
fs.mkdirSync(dumpDir, { recursive: true });

// ---------- audio capture ----------
const estSamples = Math.ceil(meta.video_frames * (meta.sample_rate / meta.fps_nes) * 1.02) + 4096;
const left = new Float32Array(estSamples);
const right = new Float32Array(estSamples);
let nSamples = 0;
let capturing = false;

// ---------- emulator ----------
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

// cynes logs a NON-standard bit layout (RIGHT=0x01 ... A=0x80, see
// cynes/__init__.py); translate it to jsnes' standard controller bits.
const BUTTONS = {
  1: Controller.BUTTON_RIGHT, // cynes RIGHT
  2: Controller.BUTTON_LEFT, // cynes LEFT
  4: Controller.BUTTON_DOWN, // cynes DOWN
  8: Controller.BUTTON_UP, // cynes UP
  16: Controller.BUTTON_START, // cynes START
  32: Controller.BUTTON_SELECT, // cynes SELECT
  64: Controller.BUTTON_B, // cynes B
  128: Controller.BUTTON_A, // cynes A
};
function setCtrl(mask) {
  for (const bit of Object.keys(BUTTONS)) {
    const b = parseInt(bit, 10);
    if (mask & b) nes.buttonDown(1, BUTTONS[b]);
    else nes.buttonUp(1, BUTTONS[b]);
  }
}

// minimal PNG writer (RGBA in, truecolor out) so we can eyeball frames later
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
  const w = 256,
    h = 240;
  const raw = Buffer.alloc(h * (1 + w * 3));
  for (let y = 0; y < h; y++) {
    const row = y * (1 + w * 3);
    raw[row] = 0; // filter none
    for (let x = 0; x < w; x++) {
      const v = buf[(y << 8) + x]; // ppu.buffer is 0xBBGGRR (LE RGBA view)
      raw[row + 1 + x * 3] = v & 0xff;
      raw[row + 1 + x * 3 + 1] = (v >> 8) & 0xff;
      raw[row + 1 + x * 3 + 2] = (v >> 16) & 0xff;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // color type RGB
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", zlib.deflateSync(raw, { level: 6 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
  fs.writeFileSync(file, png);
}

// ---------- replay ----------
let gameFrame = 0;
let idx = 0;
const progress = [];

const run = (n, mask) => {
  setCtrl(mask);
  for (let i = 0; i < n; i++) {
    nes.frame();
    gameFrame++;
    if (!capturing && gameFrame >= meta.audio_start_frame) capturing = true;
  }
};

console.log(`boot: ${meta.boot_frames} frames`);
for (const [n, mask] of meta.boot_seq) run(n, mask);

console.log(`warmup: ${meta.warmup_frames} frames from inputs.bin`);
for (let i = 0; i < meta.warmup_frames; i++) {
  setCtrl(inputs[idx++]);
  nes.frame();
  gameFrame++;
  if (!capturing && gameFrame >= meta.audio_start_frame) capturing = true;
}
nes.cpu.mem[0x6d] = 1; // start_with_axe, same as env.reset()
run(meta.settle_frames, inputs[idx++]); // settle frame

let videoFrame = 0;
const totalVideo = meta.video_frames;
while (idx < inputs.length) {
  setCtrl(inputs[idx++]);
  nes.frame();
  gameFrame++;
  if (videoFrame % dumpEvery === 0) {
    dumpPNG(path.join(dumpDir, `f${String(videoFrame).padStart(6, "0")}.png`), nes.ppu.buffer);
    progress.push({
      frame: videoFrame,
      camera: nes.cpu.mem[0x593],
      lives: nes.cpu.mem[0x3f],
    });
  }
  videoFrame++;
  if (!capturing && gameFrame >= meta.audio_start_frame) capturing = true;
  if (videoFrame % 600 === 0) {
    console.log(`  frame ${videoFrame}/${totalVideo}`);
  }
}

// ---------- write WAV ----------
const nFrames = nSamples;
const dataBytes = nFrames * 4; // 16-bit stereo
const wav = Buffer.alloc(44 + dataBytes);
wav.write("RIFF", 0);
wav.writeUInt32LE(36 + dataBytes, 4);
wav.write("WAVE", 8);
wav.write("fmt ", 12);
wav.writeUInt32LE(16, 16);
wav.writeUInt16LE(1, 20); // PCM
wav.writeUInt16LE(2, 22); // stereo
wav.writeUInt32LE(meta.sample_rate, 24);
wav.writeUInt32LE(meta.sample_rate * 4, 28);
wav.writeUInt16LE(4, 32); // block align
wav.writeUInt16LE(16, 34); // bits
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

console.log(`replayed ${videoFrame} video frames (${gameFrame} game frames total)`);
console.log(`audio: ${nFrames} samples = ${(nFrames / meta.sample_rate).toFixed(2)}s`);
console.log(`wav  -> ${path.join(outDir, "game_audio.wav")}`);
