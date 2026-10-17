"""Annotated window: shows exactly WHAT the AI sees.

Draws, live on top of the game picture:
  green  box  = player   (id 64)  with vx / gravity
  orange box  = enemies & objects, with their id and dx/dy to the player
  top bar     = lives, stamina, score, weapon, area/round, camera
  bottom      = the action the agent is taking (or your own keys)

Because the camera sits at 0 at the start of a stage, world pixels and
screen pixels coincide - so the boxes line up exactly with the sprites.

    python visualize.py                       # you play, boxes are shown
    python visualize.py runs/ppo/ppo_00081920.pt   # watch the agent
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import sdl2

from cynes import NES, NES_INPUT_A, NES_INPUT_B, NES_INPUT_LEFT
from cynes import NES_INPUT_RIGHT, NES_INPUT_START
from cynes.windowed import SDLContext

import ram_map as R

ROM = None  # resolved from env.ROM at startup (USA ROM)

BOX_H_PLAYER = 24
BOX_W_PLAYER = 16
BOX_H_OBJ = 20
BOX_W_OBJ = 16
# An object slot stores the sprite ANCHOR (its centre-ish).  A sprite is
# ~16 px wide, so an enemy whose anchor sits at x = 256..270 already shows
# its left half on screen.  Clipping the box test at exactly x < 256 left
# those edge enemies unboxed (measured: every "slotless snail" at the right
# edge had a live slot at x = 256..276).  Keep a one-sprite margin so a
# partially visible sprite is still labelled.
SPRITE_EDGE = 12

# Frames to keep showing a death in the window before auto-restarting.
# The trainer restarts with zero delay; a small delay here just makes the
# death visible to a human watching the window.
DEATH_RESTART_DELAY = 20

# keyboard fallback controls (arrow keys + z/x)
KEY_RIGHT = sdl2.SDL_SCANCODE_RIGHT
KEY_LEFT = sdl2.SDL_SCANCODE_LEFT
KEY_A = sdl2.SDL_SCANCODE_X
KEY_B = sdl2.SDL_SCANCODE_Z


def boot(nes, start_round=0, start_state=None):
    """Boot into stage 1-1, or into a later stage / a dumped save state.

    ``start_round`` is the 0-based $0038 value (1 = 1-2, 第二关);
    ``start_state`` is a file written by tools/make_round2_state.py.
    """
    if start_state is not None:
        import env as _e
        nes.load(_e.load_state_file(start_state))
        return
    if start_round > 0:
        import env as _e
        _e.boot_to_round(nes, start_round)
        return
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def objects(nes):
    """Return list of (slot, id, screen_x, screen_y, vx) for all 16 slots.

    CORRECTION (2026-10-13): slot 15 used to be skipped here because its
    x_lo byte ($0593) was thought to be the scroll register.  That was a
    misidentification: $0593 only *behaves* like a scroll for world-fixed
    objects.  Measured with tools/probe_snailslot.py -- whenever slot 15's
    3-byte decode lands in 0..255 it sits on a REAL ground snail 99.3% of
    the time (5782/5823 frames; only 41 empty).  Skipping it is exactly
    why the main snails were left unboxed.  All 16 slots are scanned now;
    the zombie filter still hides post-death stale ones.
    """
    out = []
    for s in range(R.N_OBJ_SLOTS):
        oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
        if oid == 0:
            continue
        x = R.px(R.read_obj_world_x(nes, s))
        y = R.px(R.read_obj_world_y(nes, s))
        vx = nes[R.OBJ.spd_x + R.OBJ.STRIDE * s]
        vx = vx - 256 if vx > 127 else vx
        out.append((s, oid, x, y, vx))
    return out


# every SNAP_EVERY frames the annotated frame is written here (3x zoom)
SNAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "outputs", "live_frame.png")
SNAP_EVERY = 60
SNAP_SCALE = 3
# A folder being watched for new checkpoints is polled on this wall-clock
# cadence, not per frame: at 300 fps the old 1800-frame rule reloaded the
# model several times per second for no benefit.
RELOAD_SECS = 5.0

# --- salmon-blob finder (kept for diagnostics tools ONLY) ---------------
# HISTORY: this started as a "GHOST" overlay for sprite images the game
# stopped updating (dead-enemy leftovers, the checkpoint "S" sign) which
# have no live object slot.  REMOVED from the overlay: live snails share
# the exact salmon palette (236,106,100), so colour alone cannot tell a
# leftover image from a REAL enemy -- verified 2026-10-13 with
# tools/probe_ghost_vs_enemy.py (GHOST boxes landed on live snails).
# A collision test (tools/probe_ghost_collision.py) walked right up to a
# slotless salmon blob without dying, so the leftovers appear harmless;
# slotless sprites are simply left unboxed now.
GHOST_RGB = (236, 106, 100)
GHOST_TOL = 26


def _ghost_blobs(frame: np.ndarray, exclude: list[tuple[float, float]],
                 min_px: int = 30) -> list[tuple[float, float, int]]:
    """Centers of salmon-coloured pixel blobs away from `exclude` points."""
    r = frame[:, :, 0].astype(int)
    g = frame[:, :, 1].astype(int)
    b = frame[:, :, 2].astype(int)
    mask = ((np.abs(r - GHOST_RGB[0]) < GHOST_TOL)
            & (np.abs(g - GHOST_RGB[1]) < GHOST_TOL)
            & (np.abs(b - GHOST_RGB[2]) < GHOST_TOL))
    for ex, ey in exclude:                     # player & already-boxed objs
        mask[max(0, int(ey) - 32):int(ey) + 18,
             max(0, int(ex) - 18):int(ex) + 18] = False
    m = mask[::4, ::4]
    seen = np.zeros_like(m, dtype=bool)
    out = []
    H, W = m.shape
    for y in range(H):
        for x in range(W):
            if m[y, x] and not seen[y, x]:
                stack, cells = [(y, x)], []
                seen[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    cells.append((cy, cx))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if (0 <= ny < H and 0 <= nx < W
                                    and m[ny, nx] and not seen[ny, nx]):
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                if len(cells) * 16 >= min_px:
                    out.append((float(np.mean([c[1] for c in cells])) * 4,
                                float(np.mean([c[0] for c in cells])) * 4,
                                len(cells) * 16))
    return out


def newest_checkpoint(folder: str) -> str | None:
    """Newest *.pt in `folder` by modification time ('' if there is none)."""
    try:
        cands = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.endswith(".pt")]
    except OSError:
        return None
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def annotate(frame: np.ndarray, nes, action_name: str,
             tracker: "R.SlotActivityTracker | None" = None,
             hud_extra: str = "") -> np.ndarray:
    img = Image.fromarray(frame, mode="RGB")
    d = ImageDraw.Draw(img)
    objs = objects(nes)
    # hide zombie enemies: slots the game froze at a death and no longer
    # renders (their boxes used to stay stuck on screen forever)
    if tracker is not None:
        tracker.update(nes, [o[0] for o in objs])
        n_zombie = sum(1 for o in objs
                       if o[1] in R.ENEMY_IDS and tracker.is_stale(o[0]))
        objs = [o for o in objs
                if not (o[1] in R.ENEMY_IDS and tracker.is_stale(o[0]))]
    else:
        n_zombie = 0
    player = next((o for o in objs if o[0] == R.PLAYER_SLOT), None)
    px_x, px_y = (player[2], player[3]) if player else (0, 0)

    # $0584/$0594 are SCREEN coordinates already (the camera keeps the
    # player pinned near x=96), so boxes are drawn as-is - no offset.
    for s, oid, x, y, vx in objs:
        if not (-SPRITE_EDGE <= x < 256 + SPRITE_EDGE
                and -SPRITE_EDGE <= y < 240 + SPRITE_EDGE):
            continue
        label, colour = R.kind_of(oid)
        if s == R.PLAYER_SLOT:
            box = [x - BOX_W_PLAYER / 2, y - BOX_H_PLAYER + 4,
                   x + BOX_W_PLAYER / 2, y + 4]
            d.rectangle(box, outline=(60, 255, 60), width=2)
            d.text((x - 14, y - BOX_H_PLAYER - 6), "PLAYER", fill=(60, 255, 60))
            d.text((x - 14, y + 6), f"v{vx}", fill=(60, 255, 60))
            continue

        # objects are colour-coded by kind: enemy / fruit / item
        box = [x - BOX_W_OBJ / 2, y - BOX_H_OBJ + 4,
               x + BOX_W_OBJ / 2, y + 4]
        d.rectangle(box, outline=colour, width=2 if label == "ENEMY" else 1)
        d.text((x - 12, y - BOX_H_OBJ - 6),
               f"{label}{oid}", fill=colour)
        d.text((x - 12, y + 6), f"dx{x - px_x:+.0f}", fill=colour)
        # line from player to nearby object
        if abs(x - px_x) < 140:
            d.line([px_x, px_y - 8, x, y - 8], fill=colour)

    # ---- HUD ----
    def onscreen(pred):
        return sum(1 for o in objs if pred(o[1])
                   and -SPRITE_EDGE <= o[2] < 256 + SPRITE_EDGE
                   and -SPRITE_EDGE <= o[3] < 240 + SPRITE_EDGE)

    n_enemy = onscreen(lambda i: i in R.ENEMY_IDS)
    n_fruit = onscreen(lambda i: i in R.FRUIT_IDS)
    n_item = onscreen(lambda i: i in R.ITEM_IDS)
    n_block = onscreen(lambda i: i in R.OBSTACLE_IDS)
    n_shot = onscreen(lambda i: i == R.ID_PROJECTILE)
    d.rectangle([0, 0, 256, 26 + (12 if hud_extra else 0)], fill=(0, 0, 0))
    armed = nes[R.WEAPON] > 0
    # The board is a spare life: it eats one enemy hit (falls still kill).
    # Shout about it, or the window gives no clue why a hit did not hurt.
    boarding = nes[R.SKATEBOARD] > 0
    d.text((3, 2), f"LIVES {nes[R.LIVES]}  STA {nes[R.STAMINA]}  "
                   f"AXE {'YES' if armed else 'no '}  "
                   f"A/R {nes[R.AREA]}/{nes[R.ROUND]} "
                   f"CK {nes[R.CHECKPOINT]}",
           fill=(120, 255, 120) if armed else (255, 255, 255))
    if boarding:
        d.rectangle([196, 1, 254, 11], fill=(0, 90, 0))
        d.text((199, 2), "* SKATE *", fill=(150, 255, 150))
    else:
        d.text((199, 2), "  skate -", fill=(110, 110, 110))
    d.text((3, 13), f"P x={px_x:6.1f} y={px_y:5.1f} scroll={nes[R.CAMERA]:3d} "
                    f"| E{n_enemy} F{n_fruit} I{n_item} "
                    f"B{n_block} S{n_shot} Z{n_zombie} "
                    f"ACT={action_name}", fill=(150, 255, 255))
    if hud_extra:
        d.text((3, 26), hud_extra, fill=(255, 235, 120))
    return np.asarray(img, dtype=np.uint8)


def load_agent(path: str, net):
    """Load `path` into `net`, tolerating a different action-set size.

    A checkpoint trained with 8 actions has a policy head of 8 rows while
    the current env exposes 10.  Every other tensor (the conv stack, the
    value head) still transfers, so the mismatched rows are simply dropped
    and re-initialised -- the same trick train.py uses for warm starts.
    """
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = net.state_dict()
    usable = {k: v for k, v in ck["model"].items()
              if k in sd and sd[k].shape == v.shape}
    dropped = [k for k in ck["model"] if k not in usable]
    net.load_state_dict(usable, strict=False)
    net.eval()
    return int(ck.get("steps", 0)), dropped


def guess_stage_idx(start_state, start_round):
    """0-based stage index for the one-hot: (area-1) * 4 + (round-1)."""
    if start_state:
        # saves/stages/stage_1-2.state -> "1-2"
        stem = os.path.splitext(os.path.basename(start_state))[0]
        tag = stem.replace("stage_", "").replace("round", "")
        if "-" in tag:
            try:
                a, r = (int(x) for x in tag.split("-")[:2])
                return (a - 1) * 4 + (r - 1)
            except ValueError:
                pass
    return int(start_round or 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=None,
                    help="if given, the agent plays instead of you")
    ap.add_argument("--watch-dir", default=None,
                    help="hot-reload the newest checkpoint from this folder, "
                         "so the window shows the agent getting better while "
                         "training runs (default: folder of `checkpoint`).  "
                         "With no checkpoint given the window waits for the "
                         "first one to appear.")
    ap.add_argument("--max-fps", type=float, default=0.0,
                    help="cap the rendering rate (0 = run as fast as the "
                         "emulator allows, which is what training does).  "
                         "Use 60 to watch at normal game speed.")
    ap.add_argument("--scale", type=int, default=3,
                    help="window zoom factor (default 3)")
    ap.add_argument("--start-round", type=int, default=0,
                    help="0-based stage to open: 0 = 1-1, 1 = 1-2 (第二关)")
    ap.add_argument("--start-state", type=str, default=None,
                   help="open this dumped save state instead of booting "
                        "(e.g. saves/stages/stage_1-2.state)")
    ap.add_argument("--stage-idx", type=int, default=None,
                   help="which stage the one-hot tail should say we are in "
                        "(0 = 1-1, 1 = 1-2, 4 = 2-1 ...).  Guessed from "
                        "--start-state / --start-round if omitted.")
    args = ap.parse_args()

    net = None
    net_steps = 0
    if args.watch_dir is None and args.checkpoint:
        args.watch_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    # imported unconditionally: the wait-for-checkpoint path needs
    # push_img/ActorCritic too once training catches up
    from enjoy import build_obs, obs_tensor, push_img, reset_cam  # noqa: F401
    from train import ActorCritic  # noqa: F401

    if args.checkpoint:
        import torch
        from env import ACTIONS, OBS_DIM
        net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
        net_steps, dropped = load_agent(args.checkpoint, net)
        print(f"agent loaded: {args.checkpoint} ({net_steps} steps)")
        if dropped:
            print(f"  re-initialised (action set changed): {dropped}")
    elif args.watch_dir:
        # no checkpoint yet (training saves one every ~20 k steps): open the
        # window anyway and play a scripted dummy until the first one lands
        print(f"window open - waiting for the first checkpoint in "
              f"{args.watch_dir} (scripted demo meanwhile)", flush=True)
    else:
        print("manual mode: arrows to move, X = A (jump), Z = B, ESC to quit")

    # Watching must not starve the training run: drop our own priority so
    # the trainer keeps the CPU whenever it needs it (Windows only).
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:                                   # non-Windows: ignore
        pass

    import env as _envmod
    # the one-hot stage tail has to match what is on screen, or the policy
    # thinks it is playing a different stage
    stage_idx = args.stage_idx
    if stage_idx is None:
        stage_idx = guess_stage_idx(args.start_state, args.start_round)
    print(f"stage one-hot index: {stage_idx}", flush=True)
    nes = NES(_envmod.ROM)
    boot(nes, start_round=args.start_round, start_state=args.start_state)
    start_lives = nes[R.LIVES]
    # Same curriculum as the training env: every episode starts with the axe
    # armed, otherwise a death means replaying the whole "go fetch the axe"
    # prelude.  Writing $006D = 1 is verified to arm it for real.
    nes[R.WEAPON] = 1
    start_state = nes.save()      # snapshot so we can restart after GAME OVER

    print(f"booted: lives={start_lives}, axe armed (weapon={nes[R.WEAPON]})")
    ctx = SDLContext(window_name="Adventure Island - AI annotations",
                     scaling_factor=args.scale)

    def prime_img() -> None:
        """Fill the frame stack after a boot/reset.

        The training env pushes ONE frame per 4-frame step, so its two
        stacked images are 4 frames apart.  Pushing every rendered frame
        here used to give the window a different observation distribution
        from the one the policy was trained on -- the agent then behaved
        differently in the window than in training.  We now push on
        decision frames only, and seed the stack here.
        """
        if net is None:
            return
        buf0 = np.asarray(nes.step(), dtype=np.uint8)
        for _ in range(2):
            push_img(nes, buf0)

    def restart_game() -> None:
        """Rewind to the boot snapshot and re-seed all derived state."""
        nes.load(start_state)
        tracker.reset()
        if net is not None:
            reset_cam(nes)      # drops the cached camera AND the frame stack
            prime_img()         # ...so the stack has to be refilled

    action, action_name = 0, "noop"
    tracker = R.SlotActivityTracker()   # zombie-slot filter (arms on death)
    frame = 0
    restarts = 0
    death_countdown = None
    last_restart_frame = 0
    prime_img()

    # --- live monitoring -------------------------------------------------
    # The emulator is NOT locked to 60 Hz: like training, it just steps as
    # fast as the CPU allows (typically 150-400 game frames/s).  --max-fps
    # is only there if you prefer to watch at normal game speed.
    fps_t0 = time.perf_counter()
    fps_n = 0
    fps_shown = 0.0
    reload_t0 = time.perf_counter()
    RELOAD_SECS = 5.0               # check the training folder this often
    frame_budget = (1.0 / args.max_fps) if args.max_fps > 0 else 0.0
    last_frame_at = time.perf_counter()
    hud = ""
    running = True
    while running:
        event = sdl2.SDL_Event()
        while sdl2.SDL_PollEvent(event):
            if event.type == sdl2.SDL_QUIT:
                running = False

        # first checkpoint of a training run may still be missing
        if net is None and args.watch_dir and frame % 30 == 0:
            p = newest_checkpoint(args.watch_dir)
            if p:
                try:
                    import torch
                    from env import ACTIONS, OBS_DIM
                    from train import ActorCritic
                    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
                    net_steps, dropped = load_agent(p, net)
                    args.checkpoint = p
                    prime_img()
                    print(f"agent loaded: {p} ({net_steps} steps)", flush=True)
                    if dropped:
                        print(f"  re-initialised: {dropped}", flush=True)
                except (OSError, RuntimeError, KeyError) as exc:
                    print(f"  load failed ({exc}), retrying", flush=True)
                    net = None

        if net is not None:
            if frame % 4 == 0:
                import torch
                from env import ACTIONS
                from enjoy import obs_tensor
                obs = obs_tensor(nes, stage_idx)   # {"ram": (1, D), ...}
                with torch.no_grad():
                    idx = int(net(obs)[0].argmax())
                # index -> controller bitmask (NOT the index itself!)
                action = ACTIONS[idx][0]
                action_name = ACTIONS[idx][1]
        elif args.watch_dir:
            # scripted placeholder so the window is not dead while waiting
            m = NES_INPUT_RIGHT | (NES_INPUT_A if (frame // 14) % 2 else 0)
            action, action_name = m, "DEMO(waiting for checkpoint)"
        else:
            ks = ctx.keyboard_state
            sdl2.SDL_PumpEvents()
            m = 0
            if ks[KEY_RIGHT]:
                m |= NES_INPUT_RIGHT
            if ks[KEY_LEFT]:
                m |= NES_INPUT_LEFT
            if ks[KEY_A]:
                m |= NES_INPUT_A
            if ks[KEY_B]:
                m |= NES_INPUT_B
            action = m
            action_name = ("R" if m & NES_INPUT_RIGHT else "") + \
                          ("L" if m & NES_INPUT_LEFT else "") + \
                          ("A" if m & NES_INPUT_A else "") + \
                          ("B" if m & NES_INPUT_B else "") or "noop"

        nes.controller = action
        buf = np.asarray(nes.step(), dtype=np.uint8)
        if net is not None and frame % 4 == 3:
            # one push per decision, mirroring env.step()'s frame_skip=4
            push_img(nes, buf)
        shown = annotate(buf, nes, action_name, tracker, hud)
        ctx.render_frame(shown)
        frame += 1
        fps_n += 1

        # measured throughput, refreshed a few times per second
        now = time.perf_counter()
        if now - fps_t0 >= 0.5:
            fps_shown = fps_n / (now - fps_t0)
            fps_t0, fps_n = now, 0
            model = (os.path.basename(args.checkpoint)
                     if args.checkpoint else "waiting for checkpoint")
            hud = (f"{fps_shown:6.0f} game fps "
                   f"({fps_shown / 60:.1f}x realtime)   "
                   f"model {model} {net_steps} steps   "
                   f"restarts {restarts}")
            print(f"  {hud}", flush=True)

        # throttle only if the user asked for real-time playback
        if frame_budget:
            now = time.perf_counter()
            wait = frame_budget - (now - last_frame_at)
            if wait > 0:
                time.sleep(wait)
            last_frame_at = time.perf_counter()

        # keep a still of what the agent is doing right now, so the run can
        # be inspected without sitting in front of the window
        if frame % SNAP_EVERY == 0:
            try:
                Image.fromarray(shown).resize(
                    (256 * SNAP_SCALE, 240 * SNAP_SCALE),
                    Image.NEAREST).save(SNAP_PATH)
            except OSError:                      # file busy, try next time
                pass

        # pick up a newer checkpoint if training has produced one
        if net is not None and args.watch_dir and frame:
            now = time.perf_counter()
            if now - reload_t0 >= RELOAD_SECS:
                reload_t0 = now
                newest = newest_checkpoint(args.watch_dir)
                if newest and newest != args.checkpoint:
                    try:
                        steps, dropped = load_agent(newest, net)
                        args.checkpoint = newest
                        net_steps = steps
                        print(f"  hot-reloaded {os.path.basename(newest)} "
                              f"({steps} steps)", flush=True)
                    except (OSError, RuntimeError, KeyError) as exc:
                        print(f"  reload failed ({exc}), keeping old weights")

        # Restart the moment the agent dies, instead of sitting through the
        # ~236-frame death + respawn animation and then playing on without
        # the axe.  Mirrors the training env, which terminates on $0072.
        # A short delay keeps the death visible in the window (the trainer
        # itself restarts with zero delay).
        if nes[R.DEATH] != 0 and nes[R.STAGE_COMPLETE] == 0:
            if death_countdown is None:
                death_countdown = DEATH_RESTART_DELAY
        if death_countdown is not None:
            death_countdown -= 1
            if death_countdown <= 0:
                death_countdown = None
                restart_game()
                restarts += 1
                print(f"  died -> restart #{restarts} "
                      f"(after {frame - last_restart_frame} frames)",
                      flush=True)
                last_restart_frame = frame
        elif nes[R.LIVES] == 0:
            # safety net: GAME OVER screen
            restart_game()
            restarts += 1
            last_restart_frame = frame
            print(f"  GAME OVER -> restarting (restart #{restarts})", flush=True)

    del nes


if __name__ == "__main__":
    sys.exit(main())
