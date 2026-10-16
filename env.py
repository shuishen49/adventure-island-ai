"""Gym-style RL environment for Adventure Island running on the cynes emulator.

Observation has two parts:
  ram  : player x/y/vx/gravity, stamina, weapon, nearest enemies and fruits
  img  : a down-sampled grey-scale frame stack, so the policy can actually
         SEE the platforms, pits and cliffs.  RAM alone cannot express
         terrain, and without terrain the agent cannot learn to jump.

Reward: forward progress (dominant) + score pickups + a large one-off
bonus for grabbing the axe ($006D 0 -> 1) + death/stall penalties.

The emulator is fast in headless mode (~300 FPS/core); a save state taken
right after boot makes `reset` nearly free.

Two training-speed features (added 2026-10-13, both verified):

1. INSTANT DEATH RESET.  The game only decrements lives ($003F) 236
   frames AFTER the death flag $0072 goes non-zero -- the agent used to
   sit through the whole death + respawn animation before the episode
   ended.  We now end the episode the moment $0072 fires
   (tools/probe_death_timing.py: lag = 236 frames).
2. START WITH THE AXE (``start_with_axe``).  Writing 1 to $006D really
   arms the weapon: with $006D = 0 tapping B never spawns the id-66
   projectile (0/60 taps), with $006D = 1 it does (354 frames)
   -- tools/probe_axe_cheat.py.  Dying clears the flag, so without this
   every respawn meant re-learning "go fetch the axe" before the
   actually interesting task (kill snails, keep moving).  This is
   curriculum shaping, not a reward change; pass ``start_with_axe=False``
   to train the real game.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from cynes import (
    NES,
    NES_INPUT_A,
    NES_INPUT_B,
    NES_INPUT_LEFT,
    NES_INPUT_RIGHT,
    NES_INPUT_START,
)

import ram_map as R

_HERE = os.path.dirname(os.path.abspath(__file__))
# The ROM is copyrighted game data and is NOT distributed with this repo.
# Drop your own "Adventure Island (USA).nes" into roms/, or point AI_ROM
# at it:
#     set AI_ROM=D:\path\to\Adventure Island (USA).nes        (Windows)
#     export AI_ROM=/path/to/Adventure\ Island\ \(USA\).nes   (Linux/macOS)
ROM = os.environ.get(
    "AI_ROM",
    os.path.join(_HERE, "roms", "Adventure Island (USA).nes"),
)

# Discrete action set: (controller bitmask, human name)
ACTIONS = [
    (0x00, "noop"),
    (NES_INPUT_RIGHT, "right"),
    (NES_INPUT_LEFT, "left"),
    (NES_INPUT_A, "jump"),
    (NES_INPUT_RIGHT | NES_INPUT_A, "right+jump"),
    (NES_INPUT_LEFT | NES_INPUT_A, "left+jump"),
    (NES_INPUT_B, "throw"),
    (NES_INPUT_RIGHT | NES_INPUT_B, "right+throw"),
    # 2026-10-13: the policy used to hold right+jump into every snail and
    # died on it (tools/probe_fall.py: death at 436 px with a snail 13 px
    # away).  It could never jump AND throw in the same decision, so the
    # axe was useless while airborne -- which is when you meet a snail.
    (NES_INPUT_A | NES_INPUT_B, "jump+throw"),
    (NES_INPUT_RIGHT | NES_INPUT_A | NES_INPUT_B, "right+jump+throw"),
]
N_ACTIONS = len(ACTIONS)

K_NEAREST_ENEMIES = 4
N_SLOTS_SCANNED = 16        # slots 1..15 (0 = player).  Slot 15 is a NORMAL
                            # object slot: measured 2026-10-13, its in-range
                            # positions sit on a real ground snail 99.3% of
                            # the time.  It was wrongly excluded as "camera".
FRAME_SKIP = 4
MAX_STEPS = 4500          # ~5 min of game time at 60fps / skip 4

OBS_DIM = (8                             # player
           + K_NEAREST_ENEMIES * 4       # nearest object slots
           + 4                           # global
           + 6                           # nearest enemy / fruit dx,dy,seen
           + 3)                          # nearest item (axe!) dx,dy,seen

# --- reward weights ----------------------------------------------------
# The axe is the single most important pickup in the game: without it the
# player cannot kill anything and the later rounds are close to impossible.
# $006D flips 0 -> 1 the moment the axe is picked up (verified with
# tools/probe_weapon.py -- event "f=453 WEAPON 0 -> 1" right next to an
# id-28 item), so we pay a big one-off bonus for it.  The skateboard is
# worth less but still absorbs one hit.
WEAPON_GAIN = 3.0           # picked up the axe
SKATE_GAIN = 1.5            # picked up the skateboard
SCORE_SCALE = 0.01          # per score point (fruit, kills, ...)
KILL_GAIN = 1.0             # an enemy slot died right next to us
SHOT_MEMORY = 3             # decision steps an axe stays "in flight"
# 2026-10-13: "kill" alone is too sparse to learn from.  Measured on the
# 102 k checkpoint: the agent throws plenty (an axe is airborne 32 % of
# the time) yet scored 0 kills in 6 episodes -- random throwing almost
# never connects, so the kill reward never fires and there is no gradient
# pointing at "aim at the snail".  Pay a smaller amount for simply flying
# an axe past an enemy, which is a far denser signal.
NEAR_HIT_GAIN = 0.3         # an axe flew right through an enemy
NEAR_HIT_X = 48             # px (an axe travels ~32 px per decision step,
NEAR_HIT_Y = 40             # so this window cannot be stepped over)
# ...and one step earlier still: just pulling the trigger while an enemy
# is in front.  The agent currently dies on the FIRST snail it meets, so
# a whole episode only offers one or two chances to score a near hit.
THROW_AIM_GAIN = 0.1        # threw while an enemy was ahead of us
THROW_AIM_MIN = 16          # px - closer than this and the axe cannot
THROW_AIM_MAX = 200         # reach it before we bump into it anyway
THROW_AIM_CD = 8            # steps between two such rewards (anti-farm)

# How many random frames of play to insert after a reset.  The game is
# completely deterministic, so without this all 8 parallel worlds run
# byte-identical trajectories (and every "20 episode" evaluation is the
# same episode 20 times).  Measured 2026-10-13: with warm-up the policy
# reaches 1990..3118 px instead of one single fixed 1975 px run.
WARMUP_MAX = 90

# Fleeing backwards must lose more than standing still would.  Without
# this the policy discovered it could dodge every snail by holding
# left+jump against the left screen edge (measured 2026-10-13).
BACKWARD_PENALTY = 3.0      # multiplier on the (already negative) progress
                            # reward when the agent moves backwards

# ...and it must not be able to burn the clock either: an episode that has
# not set a new progress record for this many decisions is over.
STALL_STEPS = 250           # ~1000 game frames / 17 s without advancing
STALL_PENALTY = 1.0         # charged once, on the terminating step

# --- vision -------------------------------------------------------------
# Every Nth pixel of a grey frame: 240x256 -> 80x86.  Two consecutive
# decision frames are stacked so the network can tell which way things move.
IMG_H, IMG_W, IMG_STACK = 80, 86, 2
DOWNSAMPLE = 3
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def to_grey(frame: np.ndarray) -> np.ndarray:
    """256x240 RGB -> 80x86 grey, uint8.  Cheap enough to run every step."""
    g = frame.astype(np.float32) @ _LUMA
    return g[::DOWNSAMPLE, ::DOWNSAMPLE].astype(np.uint8)


class AdventureIslandEnv:
    """Headless cynes environment with a RAM feature vector observation."""

    metadata = {"render_modes": []}

    def __init__(self, rom_path: str = ROM, frame_skip: int = FRAME_SKIP,
                 seed: Optional[int] = None,
                 start_with_axe: bool = True) -> None:
        self.rom_path = rom_path
        self.frame_skip = frame_skip
        self.rng = np.random.default_rng(seed)
        self.start_with_axe = start_with_axe

        self._nes = NES(rom_path)
        self._save_state = None

        self.action_space_n = N_ACTIONS
        self.observation_space_shape = (OBS_DIM,)

        self._reset_internal_state()

    # ------------------------------------------------------------------
    # bootstrapping
    # ------------------------------------------------------------------
    def _reset_internal_state(self) -> None:
        """Boot the game once and snapshot the state for fast resets."""
        nes = self._nes
        nes.step(frames=300)                       # title screen appears
        nes.controller = NES_INPUT_START
        nes.step(frames=60)
        nes.controller = 0
        nes.step(frames=60)                        # let gameplay settle
        self._start_lives = nes[R.LIVES]
        self._save_state = nes.save()
        self._img = np.zeros((IMG_STACK, IMG_H, IMG_W), dtype=np.uint8)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def _warmup(self) -> None:
        """Play a few random frames so episodes do not start identically.

        Only harmless inputs (right / right+jump / nothing) are used: the
        agent still starts at the beginning of the stage, just at a
        slightly different point in the run-up.
        """
        n = int(self.rng.integers(0, WARMUP_MAX + 1))
        for _ in range(n):
            self._nes.controller = int(self.rng.choice(
                (0x00, NES_INPUT_RIGHT, NES_INPUT_RIGHT | NES_INPUT_A)))
            self._nes.step()
        self._nes.controller = 0

    def reset(self, seed: Optional[int] = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._nes.load(self._save_state)
        if getattr(self, "_activity", None) is not None:
            self._activity.reset()          # new episode: disarm zombies
        self._warmup()
        if self.start_with_axe:
            # Curriculum: arm the axe right after the reset so every episode
            # starts with the tool already in hand.  Verified to work -- see
            # the module docstring.  Set before _last_weapon is sampled so
            # no phantom "picked up the axe" reward is paid.
            self._nes[R.WEAPON] = 1
        frame = np.asarray(self._nes.step(), dtype=np.uint8)
        self._img[:] = to_grey(frame)          # no history yet: duplicate it
        self._cam = R.CameraTracker(self._nes)
        self._last_x = R.px(R.read_obj_world_x(self._nes, R.PLAYER_SLOT))
        self._last_progress = self._cam.progress(self._nes)
        self._last_score = self._read_score()
        self._last_stamina = self._nes[R.STAMINA]
        self._last_weapon = self._nes[R.WEAPON]
        self._last_skate = self._nes[R.SKATEBOARD]
        self._last_area = self._nes[R.AREA]
        self._last_round = self._nes[R.ROUND]
        self._last_checkpoint = self._nes[R.CHECKPOINT]
        self._steps = 0
        self._x_progress = 0.0
        self._best_progress = self._last_progress
        self._best_step = 0
        self._prev_enemies = {}
        self._hit_pairs: set[tuple[int, int]] = set()
        self._aim_cd = 0
        self._action = 0
        self._kills = 0
        self._shot_steps = 0
        return self._obs()

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def _push_frame(self, frame: np.ndarray) -> None:
        self._img[:-1] = self._img[1:]
        self._img[-1] = to_grey(frame)

    def _obs(self) -> dict[str, np.ndarray]:
        """{"ram": float32 vector, "img": uint8 stack}."""
        return {"ram": self._build_obs(), "img": self._img.copy()}

    def step(self, action: int) -> tuple[dict, float, bool, dict[str, Any]]:
        assert 0 <= action < N_ACTIONS
        self._action = action          # reward shaping needs to know if B
        self._nes.controller = ACTIONS[action][0]
        for _ in range(self.frame_skip):
            frame = np.asarray(self._nes.step(), dtype=np.uint8)
        self._nes.controller = 0
        self._push_frame(frame)
        self._last_frame = frame          # kept for debugging / snapshots
        self._steps += 1

        obs = self._obs()
        reward, info, terminated = self._compute_reward()
        truncated = self._steps >= MAX_STEPS
        return obs, reward, terminated or truncated, info

    def render(self) -> None:  # headless: no-op
        pass

    def close(self) -> None:
        del self._nes

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _read_score(self) -> int:
        """Score is stored in a quirky BCD variant at $06A1 (low nibble).
        Only deltas matter for the reward, so raw value is fine."""
        return self._nes[R.SCORE + 7]

    def _track_slots(self) -> None:
        """Feed the zombie-slot detector (call ONCE per observation).

        After a death/respawn the game keeps stale enemy slots forever
        (all field bytes frozen, no longer rendered -- see
        ram_map.SlotActivityTracker).  Without filtering, the agent would
        observe a phantom enemy frozen at the death spot.  Detection only
        arms on a death, so live enemies are never hidden.
        """
        nes = self._nes
        tr = getattr(self, "_activity", None)
        if tr is None:
            tr = self._activity = R.SlotActivityTracker()
        tr.update(nes, [s for s in range(1, N_SLOTS_SCANNED)
                        if nes[R.OBJ.id + R.OBJ.STRIDE * s]])

    def _is_zombie(self, slot: int, oid: int) -> bool:
        tr = getattr(self, "_activity", None)
        return (tr is not None and oid in R.ENEMY_IDS and tr.is_stale(slot))

    def _active_enemies(self) -> list[tuple[int, int, int, int]]:
        """Return (dx, dy, spd_x, obj_id) for on-screen non-player objects."""
        nes = self._nes
        p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
        p_y = R.read_obj_world_y(nes, R.PLAYER_SLOT)
        out = []
        for s in range(1, N_SLOTS_SCANNED):
            oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
            if oid == 0 or self._is_zombie(s, oid):
                continue
            o_x = R.read_obj_world_x(nes, s)
            o_y = R.read_obj_world_y(nes, s)
            dx = R.px(o_x - p_x)
            dy = R.px(o_y - p_y)
            if abs(dx) > 320 or not (-64 < dy < 256):   # off-screen
                continue
            spd = nes[R.OBJ.spd_x + R.OBJ.STRIDE * s]
            spd = spd - 256 if spd > 127 else spd
            out.append((dx, dy, spd / 64.0, oid))
        out.sort(key=lambda e: abs(e[0]))
        return out[:K_NEAREST_ENEMIES]

    def _nearest_of_kind(self, ids: tuple[int, ...] | int
                         ) -> tuple[float, float, float]:
        """(dx, dy, present) of the closest object whose id is in `ids`.

        Fruits keep the stamina bar alive, so the agent needs to see them
        explicitly instead of only through the generic nearest slots.
        """
        if isinstance(ids, int):
            ids = (ids,)
        nes = self._nes
        p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
        p_y = R.read_obj_world_y(nes, R.PLAYER_SLOT)
        best = None
        for s in range(1, N_SLOTS_SCANNED):
            oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
            if oid not in ids or self._is_zombie(s, oid):
                continue
            dx = R.px(R.read_obj_world_x(nes, s) - p_x)
            dy = R.px(R.read_obj_world_y(nes, s) - p_y)
            if abs(dx) > 320 or not (-64 < dy < 256):
                continue
            if best is None or abs(dx) < abs(best[0]):
                best = (dx, dy)
        if best is None:
            return 0.0, 0.0, 0.0
        return best[0], best[1], 1.0

    def _build_obs(self) -> np.ndarray:
        nes = self._nes
        self._track_slots()
        p_x = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        p_y = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        spd = nes[R.OBJ.spd_x]
        spd = spd - 256 if spd > 127 else spd
        grav = nes[R.OBJ.gravity]
        grav = grav - 256 if grav > 127 else grav
        progress = self._cam.progress(nes)

        player = np.array([
            p_x / 256.0,                   # screen x (0..1)
            p_y / 256.0,
            spd / 64.0,
            grav / 64.0,
            nes[R.STAMINA] / 16.0,
            float(nes[R.WEAPON] > 0),      # holding the axe?
            nes[R.SKATEBOARD] / 255.0,
            float(nes[R.BTN_A_FLAG] > 0),
        ], dtype=np.float32)

        enemies = np.zeros(K_NEAREST_ENEMIES * 4, dtype=np.float32)
        for i, (dx, dy, s, oid) in enumerate(self._active_enemies()):
            enemies[4 * i:4 * i + 4] = (
                np.clip(dx / 256.0, -2, 2), dy / 256.0, s, oid / 255.0)

        e_dx, e_dy, e_seen = self._nearest_of_kind(R.ENEMY_IDS)
        f_dx, f_dy, f_seen = self._nearest_of_kind(R.FRUIT_IDS)
        # id 28 = power-up egg: axe, skateboard, heart ...  The agent can
        # only go and fetch the axe if it knows where the egg is.
        i_dx, i_dy, i_seen = self._nearest_of_kind(R.ITEM_IDS)

        glob = np.array([
            (self._nes[R.LIVES] - self._start_lives) / 4.0,
            nes[R.AREA] / 8.0 + nes[R.ROUND] / 32.0,
            float(nes[R.STAGE_COMPLETE] > 0),
            progress / 2048.0,
        ], dtype=np.float32)

        kind = np.array([
            np.clip(e_dx / 256.0, -2, 2), e_dy / 256.0, e_seen,
            np.clip(f_dx / 256.0, -2, 2), f_dy / 256.0, f_seen,
            np.clip(i_dx / 256.0, -2, 2), i_dy / 256.0, i_seen,
        ], dtype=np.float32)

        return np.concatenate([player, enemies, glob, kind])

    def _near_hits(self) -> int:
        """How often an axe of ours is flying right through an enemy.

        Reward shaping for aiming: killing needs a lucky hit, and random
        throwing almost never lands one (102 k checkpoint: 32 % of steps
        had an axe in the air, 0 kills in 6 episodes).  Paying for "the
        axe passed the snail" instead of "the snail died" turns a signal
        that never fires into one that fires often.

        Scored once per (axe slot, enemy slot) pair: while the pair keeps
        existing it stays in ``_hit_pairs`` and pays no further reward,
        and once it is gone (axe expired / enemy dead) it can score again.
        """
        nes = self._nes
        p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
        p_y = R.read_obj_world_y(nes, R.PLAYER_SLOT)
        shots: list[tuple[int, float, float]] = []
        foes: list[tuple[int, float, float]] = []
        for s in range(1, N_SLOTS_SCANNED):
            oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
            if oid == 0:
                continue
            if oid == R.ID_PROJECTILE:
                shots.append((s, R.px(R.read_obj_world_x(nes, s) - p_x),
                              R.px(R.read_obj_world_y(nes, s) - p_y)))
            elif oid in R.ENEMY_IDS and not self._is_zombie(s, oid):
                foes.append((s, R.px(R.read_obj_world_x(nes, s) - p_x),
                             R.px(R.read_obj_world_y(nes, s) - p_y)))

        live: set[tuple[int, int]] = set()
        hits = 0
        for ss, sx, sy in shots:
            for fs, fx, fy in foes:
                if abs(sx - fx) < NEAR_HIT_X and abs(sy - fy) < NEAR_HIT_Y:
                    key = (ss, fs)
                    live.add(key)
                    if key not in self._hit_pairs:
                        hits += 1
        self._hit_pairs = live
        return min(hits, 2)          # never pay for more than 2 per step

    def _count_kills(self) -> int:
        """Enemy slots that were next to us last step and are gone now.

        Two conditions must both hold:
        * the slot was within 96 px when last seen (enemies are only
          despawned far off screen, so a close slot that vanishes was
          killed);
        * one of our own axes (id 66) was in flight during the last
          SHOT_MEMORY steps -- the axe is the ONLY way to kill anything
          in this game, so a disappearance without a shot in the air is
          most likely slot bookkeeping, not a kill.

        The game itself pays NO score for a kill (tools/probe_score.py:
        5 kills, 0 score events -- score only moves when fruit is eaten),
        which is why this detector, not the score, drives the reward.
        """
        nes = self._nes
        p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
        cur: dict[int, float] = {}
        shot = False
        for s in range(1, N_SLOTS_SCANNED):
            oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
            if oid == 0:
                continue
            if oid in R.ENEMY_IDS and not self._is_zombie(s, oid):
                cur[s] = R.px(R.read_obj_world_x(nes, s) - p_x)
            elif oid == R.ID_PROJECTILE:
                shot = True
        # remember the axe for a few decision steps: it can hit and vanish
        # between two observations
        self._shot_steps = SHOT_MEMORY if shot else max(
            0, self._shot_steps - 1)
        kills = sum(1 for s, dx in self._prev_enemies.items()
                    if s not in cur and abs(dx) < 96
                    and self._shot_steps > 0)
        self._prev_enemies = cur
        return kills

    def _compute_reward(self):
        nes = self._nes
        score = self._read_score()
        sta = nes[R.STAMINA]
        lives = nes[R.LIVES]
        weapon = nes[R.WEAPON]
        skate = nes[R.SKATEBOARD]

        # A new area / round / checkpoint or a death resets the player, and
        # with it the weapon flag.  Only credit pickups the agent really
        # walked into, never the automatic resets.
        area_changed = (nes[R.AREA] != self._last_area
                        or nes[R.ROUND] != self._last_round
                        or nes[R.CHECKPOINT] != self._last_checkpoint)

        # True world progress: camera travel + on-screen movement.
        progress = self._cam.progress(nes)
        dp = progress - self._last_progress
        if abs(dp) < 64:                     # ignore respawn teleports
            self._x_progress += dp
            progress_r = float(np.clip(dp, -32, 32)) * 0.02
            # Walking BACKWARDS must be clearly worse than standing still,
            # otherwise fleeing left is cheaper than facing the snail ahead
            # (measured 2026-10-13: the ppo_v3 policy learned to hold
            # left+jump against the left screen edge for 1000+ frames
            # instead of advancing -- see tools/probe_window_traj.py).
            if dp < 0:
                progress_r *= BACKWARD_PENALTY
        else:
            progress_r = 0.0

        score_r = np.clip(score - self._last_score, 0, 20) * SCORE_SCALE
        stamina_r = -0.01 if sta < self._last_stamina else 0.0

        # --- instant death detection ---------------------------------
        # $0072 fires at the death instant (1) and stays non-zero (255)
        # through the respawn animation, while lives only drops 236
        # frames later.  Ending the episode on $0072 instead of on the
        # lives counter removes ~59 wasted decisions (236 / frame_skip 4)
        # per death and -- more importantly -- never lets the agent play
        # on after a respawn, which is exactly when it has lost the axe
        # and has to re-learn the pickup.
        # Stage clear is not a death, so it is excluded.
        dying = nes[R.DEATH] != 0 and nes[R.STAGE_COMPLETE] == 0
        dead = dying or lives < self._start_lives
        death_r = -5.0 if dead else 0.0

        # --- stall detection -----------------------------------------
        # The second way the policy dodged the snails was to simply stop:
        # hold left+jump against the screen edge and burn the clock.  Left
        # alone that is a stable strategy (only -0.001/step), so an episode
        # with no new best progress for STALL_STEPS decisions is ended with
        # a penalty.  "Standing still" is then never better than trying.
        if progress > self._best_progress:
            self._best_progress = progress
            self._best_step = self._steps
        stalled = (self._steps - self._best_step) >= STALL_STEPS
        stall_r = -STALL_PENALTY if stalled else 0.0

        # --- the important part: picking up the axe -------------------
        weapon_r = 0.0
        if weapon > self._last_weapon and not area_changed and not dead:
            weapon_r = WEAPON_GAIN
        skate_r = 0.0
        if skate > self._last_skate and not area_changed and not dead:
            skate_r = SKATE_GAIN

        # Killed something?  Paid once per dead enemy slot.
        kills = 0 if dead else self._count_kills()
        if dead:
            self._prev_enemies = {}
            self._shot_steps = 0
            self._hit_pairs = set()
        kill_r = KILL_GAIN * kills
        # Aiming shaping: an axe that flew through an enemy.
        near = 0 if dead else self._near_hits()
        near_r = NEAR_HIT_GAIN * near
        # ...and: pulling the trigger while an enemy is ahead at all.
        aim_r = 0.0
        self._aim_cd = max(0, self._aim_cd - 1)
        if (not dead and self._aim_cd == 0
                and (ACTIONS[self._action][0] & NES_INPUT_B)):
            ahead = min((dx for dx in self._prev_enemies.values() if dx > 0),
                        default=None)
            if ahead is not None and THROW_AIM_MIN < ahead < THROW_AIM_MAX:
                aim_r = THROW_AIM_GAIN
                self._aim_cd = THROW_AIM_CD

        reward = (progress_r + score_r + stamina_r + death_r
                  + weapon_r + skate_r + kill_r + near_r + aim_r
                  + stall_r - 0.001)

        self._last_progress = progress
        self._last_score = score
        self._last_stamina = sta
        self._last_weapon = weapon
        self._last_skate = skate
        self._last_area = nes[R.AREA]
        self._last_round = nes[R.ROUND]
        self._last_checkpoint = nes[R.CHECKPOINT]

        info = {"x_progress": self._x_progress, "score": score,
                "stamina": sta, "lives": lives, "death": bool(dead),
                "weapon": weapon, "got_axe": bool(weapon_r),
                "got_skate": bool(skate_r), "stalled": bool(stalled),
                "kills": kills, "near": near,
                "best_progress": self._best_progress}
        return reward, info, dead or stalled


if __name__ == "__main__":
    import time

    env = AdventureIslandEnv()
    obs = env.reset()
    print(f"obs keys: {list(obs)}  ram={obs['ram'].shape} img={obs['img'].shape}")
    print(f"example ram: {np.round(obs['ram'], 2)}")

    t0, total, steps = time.perf_counter(), 0.0, 0
    for i in range(600):
        a = 1 if (i // 30) % 3 else 4          # mostly run right, hop sometimes
        obs, r, done, info = env.step(a)
        total += r
        steps += 1
        if done:
            print(f"episode over after {steps} steps: {info}")
            obs = env.reset()
            steps = 0
    dt = time.perf_counter() - t0
    print(f"600 env steps ({600 * env.frame_skip} frames) in {dt:.2f}s "
          f"-> {600 * env.frame_skip / dt:.0f} FPS, total reward {total:.2f}")
    print("final info:", info)
    env.close()
