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

# --- multi-stage curriculum ---------------------------------------------
# One policy for all 32 stages (8 areas x 4 rounds).  The observation
# carries a one-hot "which stage am I in" tail so the shared trunk can
# specialise per stage; see plan.md section 2 decision 1.
N_STAGES = 32
WARMUP_RETRIES = 3          # re-roll the warm-up if it kills us
# Clearing the flag is the goal of a stage (plan.md phase 1), not a
# by-product: symmetric with the -5.0 death penalty.
CLEAR_GAIN = 5.0

OBS_DIM = (9                             # player (incl. fireball bit)
           + K_NEAREST_ENEMIES * 4       # nearest object slots
           + 4                           # global
           + 6                           # nearest enemy / fruit dx,dy,seen
           + 3                           # nearest item (axe!) dx,dy,seen
           + N_STAGES                    # one-hot: which stage is this
           + 1)                          # invincible (fairy puff, id 199)

# --- reward weights ----------------------------------------------------
# The axe is the single most important pickup in the game: without it the
# player cannot kill anything and the later rounds are close to impossible.
# $006D flips 0 -> 1 the moment the axe is picked up (verified with
# tools/probe_weapon.py -- event "f=453 WEAPON 0 -> 1" right next to an
# id-28 item), so we pay a big one-off bonus for it.  The skateboard is
# worth less but still absorbs one hit.
WEAPON_GAIN = 3.0           # picked up the axe
SKATE_GAIN = 1.5            # picked up the skateboard
# A 1UP is worth real progress in a game this lethal: the observation
# carries (lives - start)/4, but until 2026-10-17 the reward only paid
# for LOSING a life (-5), never for gaining one -- so a hidden-egg 1UP
# was invisible to the optimiser.  Paid once on the rising edge.
LIFE_GAIN = 2.0             # gained an extra life
# $006D is a weapon TYPE: 1 = axe, 2 = FIREBALL (measured 2026-10-17,
# tools/probe_fireball.py: same object id 66, but reach 270 px vs 186 px,
# and the wiki says it is the only weapon that breaks rocks).  It only
# comes from HIDDEN power-up eggs, so it is rare -- pay more than the axe.
FIRE_GAIN = 4.0             # upgraded axe -> fireball
# 2026-10-17: object id 199 is the white fairy puff that trails the
# player while INVINCIBLE (tools/probe_fairy_inv.py: an enemy pinned
# onto the player does NOT set the death flag while 199 is on stage,
# 2/2; the same poke kills without it, 2/2).  The puff only exists
# after the fairy pickup has been grabbed, so pay on its rising edge.
INV_GAIN = 3.0              # grabbed the invincibility fairy
FAIRY_ID = 199
# 2026-10-17: the skateboard absorbs ONE hit -- verified with
# tools/probe_skateboard.py: walking into an enemy kills you at frame 504
# with $006E = 0, but holding $006E = 1 you walk through the very same
# enemy and survive.  It does NOT save you from a fall: dropping into the
# sea on 1-2 kills you at frame 238 bare and 177 with the board held.
# So "board absorbed a hit" is worth almost as much as a life -- pay for it
# explicitly, otherwise the agent only ever sees the board *appearing*
# (SKATE_GAIN) and never learns what it was good for.
SKATE_SAVE = 2.0            # the board took a hit that would have killed us
SCORE_SCALE = 0.01          # per score point (fruit, kills, ...)
KILL_GAIN = 2.0             # an enemy slot died right next to us
SHOT_MEMORY = 3             # decision steps an axe stays "in flight"
# 2026-10-13: "kill" alone is too sparse to learn from.  Measured on the
# 102 k checkpoint: the agent throws plenty (an axe is airborne 32 % of
# the time) yet scored 0 kills in 6 episodes -- random throwing almost
# never connects, so the kill reward never fires and there is no gradient
# pointing at "aim at the snail".  Pay a smaller amount for simply flying
# an axe past an enemy, which is a far denser signal.
NEAR_HIT_GAIN = 0.3         # an axe flew right through an enemy
# --- the 1-4 boss: hit counter, NOT $0011 ------------------------------
# 2026-10-18, measured with tools/boss_fight_bot.py + outputs/boss_win.npy.
#
#   * $0011 IS A DECOY.  It reads 8 / 7 / 6 in various snapshots (which
#     looks exactly like "8 hits for the area-1 boss") but it never moves
#     when you hit anything: 3308 frames of jump-throwing that killed the
#     boss left it pinned at 6 the whole time.  Do NOT pay for it.
#   * The real signal is field block 16 ($0664 + slot) of the slot holding
#     the HEAD (id 198, slot 14 in every snapshot we have).  It counts UP
#     one step per successful hit: 0 -> 1 -> 2 ... -> 8, and at 8 the boss
#     "changes head and escapes": the body id goes 67 -> 63, the head rises
#     off the top of the screen and $007A (stage complete) flips to 1.
#     Eight matches the published area-1 requirement exactly.
#   * A/B with the same movement script (tools/boss_fight_bot.py):
#       jump + throw -> boss dies at frame 3308
#       throw only  -> never      jump only -> never      neither -> never
#     so the reward cannot be farmed by standing around.
#   * It is slot-local: the body's own block 16 ($0673) oscillates 0..3
#     with its walk animation, so the counter MUST be read from the head
#     slot and only while id 198 is really there.
BOSS_HEAD_HIT_BLOCK = 0x0664      # block 16 of the object table
BOSS_HIT_GAIN = 1.0               # per confirmed head hit
BOSS_KILL_GAIN = 6.0              # boss defeated (body 67 -> 63)
# One-shot bonus for the frame the boss first appears on screen.  See the
# comment at its use site: without it the policy prefers to idle just
# short of the arena trigger.
BOSS_ARRIVE_GAIN = 2.0
# 2026-10-18: the flat window was reward hacking.  Measured
# (tools/snail_test.py on 1-1, and the same probe on 1-4):
#   * a kill is REAL and detectable -- the snail slot goes 14 -> 69 at the
#     moment the axe connects, so KILL_GAIN can fire;
#   * the axe flies on an ARC: dx 44 -> dy 101, dx 94 -> dy 139,
#     dx 102 -> dy 147, and it only reaches a ground enemy (dy ~159) at the
#     END of the arc, dx >= ~100.  Thrown at dx 70 it passes ~60 px above
#     the target.
# With NEAR_HIT_Y = 40 that miss still scored a full 0.3, so the policy had
# no reason to ever lift the throw point: 209 "hits" in 4 episodes of 1-4
# and 0 kills.  The window is now tighter AND graded -- the closer the axe
# passes to the enemy's centre, the more it pays, with 0 at the rim.
NEAR_HIT_X = 48             # px (an axe travels ~32 px per decision step,
NEAR_HIT_Y = 40             # so this window cannot be stepped over)
# The window keeps its old reach on purpose: kills DO happen (5 in one
# episode of 1-4), so this is a dense and useful signal.  Only the shape
# changed -- graded, so a centre pass pays ~1.0 and a rim pass ~0.
# ...and one step earlier still: just pulling the trigger while an enemy
# is in front.  The agent currently dies on the FIRST snail it meets, so
# a whole episode only offers one or two chances to score a near hit.
THROW_AIM_GAIN = 0.1        # threw while an enemy was ahead of us
THROW_AIM_MIN = 16          # px - closer than this and the axe cannot
THROW_AIM_MAX = 200         # reach it before we bump into it anyway
THROW_AIM_CD = 8            # steps between two such rewards (anti-farm)

# --- boss aim shaping ---------------------------------------------------
# The 1-4 head is a tiny, high, moving target and the whole fight is worth
# exactly 8 hits, so "+1 per hit" alone is far too sparse to learn from.
# Measured geometry (2026-10-18):
#   * a standing throw peaks at y=136 and a standing jump throw at y=107,
#     while the head sits at y=93 -- only a RUNNING jump gets an axe up
#     there, and the axe only travels ~57 px before it falls;
#   * touching the boss kills outright, so the usable band is ~25..70 px.
# So: pay a little for releasing a throw while airborne with the head in
# that band.  Cooldown keeps it from being farmed by spamming B.
BOSS_AIM_GAIN = 0.2
BOSS_AIM_MIN = 20
BOSS_AIM_MAX = 72
BOSS_AIM_CD = 12

# ...and drifting the other way must cost something BEFORE the fatal
# contact does.  Death is -5 but it arrives too late to shape anything:
# a greedy 1-4 run (2026-10-18) walked from dx=131 straight into the boss
# at dx=22 and never threw once, because "keep going right" is exactly
# what the travelling half of the policy is rewarded for.
BOSS_TOO_CLOSE = 24         # px - inside this the boss kills on contact
BOSS_CLOSE_PENALTY = 0.05   # per decision, ~1/4 of a landed hit per second
# ...and sitting in the throwing band is worth something on its own, every
# step, not just on the throw cooldown.  Measured 2026-10-18: from 91 px
# away the policy kills the boss 10/10, from 120 px it scores 8,0,3,0,0,2
# -- the missing skill is BRAKING, and before this there was no reason to
# stop walking right once the approach bonus ran out at 72 px.
BOSS_HOLD_GAIN = 0.03       # per decision inside (BOSS_AIM_MIN, BOSS_AIM_MAX)

# Closing in from outside the band: paid once per NEW CLOSEST APPROACH
# (per pixel gained, capped), so pacing back and forth farms nothing.
BOSS_APPROACH_GAIN = 0.02    # per px closer, same order as progress reward
BOSS_APPROACH_MAX = 170      # px - beyond this the head is not the target

# The arena never advances the camera, so the normal "no new progress"
# stall timer would cut every boss fight off at 250 steps -- and the
# scripted bot needs ~825 steps (3300 frames) for the 8 head hits.  While
# the boss is on screen the clock instead measures INACTIVITY: every
# useful action (a throw inside the band, or a landed hit) resets it.
BOSS_STALL_STEPS = 600

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


def control_live(nes, frames: int = 30) -> bool:
    """Is the player under our control right now?  (state is restored)

    Idle first, then RIGHT: right after a stage clear the game walks
    Higgins right by itself, so "RIGHT moved him" alone is a false
    positive there.
    """
    st = nes.save()
    x0 = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    nes.controller = 0
    for _ in range(frames):
        nes.step()
    x_idle = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    nes.load(st)
    nes.controller = NES_INPUT_RIGHT
    for _ in range(frames):
        nes.step()
    nes.controller = 0
    x_right = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    nes.load(st)
    return abs(x_idle - x0) < 2.0 and (x_right - x0) > 5.0


def wait_for_picture(nes, max_frames: int = 400) -> None:
    """Step until the stage has finished fading in.

    Right after a stage load the PPU is still off: the frame is PURE BLACK
    for ~180 frames while the game logic already runs and the player
    already answers the controller.  A state saved in there would start
    every episode with ~45 black observations -- and would look like
    "still loading" in a screenshot.
    """
    for _ in range(max_frames):
        nes.controller = 0
        if float(np.asarray(nes.step()).mean()) > 30.0:
            return


def boot_to_round(nes, round_value: int) -> None:
    """Title -> START with $0038 forced, then wait for a playable stage.

    The game reads the round register while the stage is being built, so
    hammering it through the init loads that stage's layout (spawn y:
    152 for round 0, 80 for round 1, 192 for round 2).
    """
    nes.step(frames=300)                       # title screen appears
    nes.controller = NES_INPUT_START
    for _ in range(90):
        nes[R.ROUND] = round_value
        nes.step()
    nes.controller = 0
    for _ in range(60):                        # up to 600 frames
        if control_live(nes):
            break
        for _ in range(10):
            nes.controller = 0
            nes.step()
    wait_for_picture(nes)                      # out of the black fade-in


def load_state_file(path: str) -> np.ndarray:
    """Read a dumped cynes state.  ``load`` wants a *writeable* array."""
    with open(path, "rb") as fh:
        return np.frombuffer(fh.read(), dtype=np.uint8).copy()


def stage_sort_key(name: str) -> tuple[int, int, str]:
    """Canonical sort key for a stage-pool entry.

    The real game has 32 stages: 1-1..1-4, 2-1..2-4, ... 8-4.  There is NO
    1-5/1-6/1-7 -- those names used to be invented for practice snapshots
    and read like real stages (2026-10-18).  A practice snapshot of a stage
    is now named <stage><suffix>, e.g. "1-4boss" (boss arena) or "1-4late"
    (a mid-stage restart), and sorts immediately after the stage it belongs
    to: "1-4" < "1-4boss" < "1-4late" < "2-1".
    """
    area, _, rest = name.partition("-")
    digits = ""
    while rest and rest[0].isdigit():
        digits += rest[0]
        rest = rest[1:]
    return (int(area) if area.isdigit() else 99,
            int(digits) if digits else 0,
            rest)


def parse_stage_pool(spec: str, root: Optional[str] = None
                     ) -> tuple[list[str], list[float], list[str], list[int]]:
    """'"1-4:0.4,1-4boss:0.3@3"' -> (paths, weights, names, one-hot idx).

    The one-hot index a stage gets is its POSITION in this list, so every
    tool (train.py, visualize.py, tools/eval_boss.py) must call this exact
    function with the exact same string, or the policy is fed a stage bit
    that does not match the screen.

    "@N" overrides that index.  It exists because several pool entries can
    be different SNAPSHOTS OF THE SAME STAGE (1-4 from the start, 1-4late
    from ~3500 px, 1-4boss in the arena).  Left alone they would get three
    different one-hot bits and the policy would treat "the last stretch of
    1-4" and "the boss fight of 1-4" as unrelated games, so skills learned
    in one would not transfer to the other.  Measured 2026-10-18: the very
    same arena picture scored 8/8 hits at index 4 and 0 hits at index 3.
    """
    import os as _os
    root = root or _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 "saves", "stages")
    named: list[tuple[str, float, Optional[int]]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, rest = item.partition(":")
        weight, _, idx = rest.partition("@")
        named.append((name.strip(), float(weight) if weight else 1.0,
                      int(idx) if idx.strip() else None))
    named.sort(key=lambda nw: stage_sort_key(nw[0]))
    paths, weights, names = [], [], []
    for name, w, _ in named:
        paths.append(_os.path.join(root, f"stage_{name}.state"))
        weights.append(w)
        names.append(name)
    s = sum(weights) or 1.0
    indices = [idx if idx is not None else i
               for i, (_, _, idx) in enumerate(named)]
    return paths, [w / s for w in weights], names, indices


class AdventureIslandEnv:
    """Headless cynes environment with a RAM feature vector observation."""

    metadata = {"render_modes": []}

    def __init__(self, rom_path: str = ROM, frame_skip: int = FRAME_SKIP,
                 seed: Optional[int] = None,
                 start_with_axe: bool = True,
                 start_round: int = 0,
                 start_state: Optional[str] = None,
                 stage_pool: Optional[list[str]] = None,
                 stage_weights: Optional[list[float]] = None,
                 stage_index: Optional[int] = None,
                 stage_indices: Optional[list[int]] = None,
                 clear_ends_episode: bool = False) -> None:
        self.rom_path = rom_path
        self.frame_skip = frame_skip
        self.rng = np.random.default_rng(seed)
        self.start_with_axe = start_with_axe
        self.clear_ends_episode = clear_ends_episode
        # Which stage an episode starts in.  $0038 is 0-based: 0 is stage
        # 1-1, 1 is stage 1-2 (第二关) -- the game bumps it to 1 the moment
        # 1-1 is cleared (tools/make_round2_state.py).  Either give the
        # register value here (the boot warps into it) or hand over a state
        # file dumped by that tool.
        self.start_round = start_round
        self._start_state = None
        if start_state is not None:
            # cynes wants a *writeable* uint8 array, not raw bytes
            with open(start_state, "rb") as fh:
                self._start_state = np.frombuffer(fh.read(),
                                                  dtype=np.uint8).copy()

        # --- multi-stage curriculum -----------------------------------
        # A pool of per-stage save states; reset() draws one by weight.
        # 32 states x ~5 KB is nothing, so they all stay in memory.
        self.stage_pool: list[np.ndarray] = []
        self.stage_labels: list[str] = []
        if stage_pool:
            for path in stage_pool:
                self.stage_pool.append(load_state_file(path))
                self.stage_labels.append(
                    os.path.splitext(os.path.basename(path))[0])
            w = np.asarray(stage_weights or [1.0] * len(stage_pool),
                           dtype=np.float64)
            self.stage_weights = (w / w.sum()).tolist()
        else:
            self.stage_weights = []
        # Optional per-entry override of the one-hot bit (see
        # parse_stage_pool).  Several snapshots of one stage share a bit so
        # the skill transfers between them; pool position alone would split
        # them.  None -> fall back to "index = position in the pool".
        self.stage_indices: Optional[list[int]] = (
            list(stage_indices) if stage_indices is not None else None)
        # Which stage the CURRENT episode is in -- the 32-way one-hot at
        # the end of the observation is built from this.
        self._stage_idx = stage_index if stage_index is not None else 0

        self._nes = NES(rom_path)
        self._save_state = None

        self.action_space_n = N_ACTIONS
        self.observation_space_shape = (OBS_DIM,)

        self._reset_internal_state()

    # ------------------------------------------------------------------
    # bootstrapping
    # ------------------------------------------------------------------
    def _control_live(self, frames: int = 30) -> bool:
        """Is the player under our control?  (state is restored afterwards)

        Idle first, then RIGHT: right after a stage clear the game walks
        Higgins right by itself, so "RIGHT moved him" alone is a false
        positive.  See tools/make_round2_state.py.
        """
        nes = self._nes
        st = nes.save()
        x0 = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        nes.controller = 0
        for _ in range(frames):
            nes.step()
        x_idle = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        nes.load(st)
        nes.controller = NES_INPUT_RIGHT
        for _ in range(frames):
            nes.step()
        nes.controller = 0
        x_right = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        nes.load(st)
        return abs(x_idle - x0) < 2.0 and (x_right - x0) > 5.0

    def _wait_for_picture(self, max_frames: int = 400) -> None:
        """Step until the stage has finished fading in.

        Right after a stage load the PPU is still off: the frame is PURE
        BLACK for ~180 frames while the game logic already runs and the
        player already answers the controller.  A state saved in there
        would start every episode with ~45 black observations -- and would
        look like "still loading" in a screenshot.
        """
        nes = self._nes
        for _ in range(max_frames):
            nes.controller = 0
            if float(np.asarray(nes.step()).mean()) > 30.0:
                return

    def _boot_to_round(self, round_value: int) -> None:
        """Title -> START with $0038 forced, then wait for real control.

        The game reads the round register while the stage is being built,
        so hammering it through the init loads that stage's layout
        (spawn y: 152 for round 0, 80 for round 1, 192 for round 2).
        """
        nes = self._nes
        nes.step(frames=300)                       # title screen appears
        nes.controller = NES_INPUT_START
        for _ in range(90):
            nes[R.ROUND] = round_value
            nes.step()
        nes.controller = 0
        for _ in range(60):                        # up to 600 frames
            if self._control_live():
                break
            for _ in range(10):
                nes.controller = 0
                nes.step()
        self._wait_for_picture()                   # out of the black fade-in

    def _reset_internal_state(self) -> None:
        """Boot the game once and snapshot the state for fast resets."""
        nes = self._nes
        if self._start_state is not None:
            nes.load(self._start_state)
        elif self.start_round > 0:
            self._boot_to_round(self.start_round)
        else:
            nes.step(frames=300)                       # title screen appears
            nes.controller = NES_INPUT_START
            nes.step(frames=60)
            nes.controller = 0
            nes.step(frames=60)                        # let gameplay settle
        self._start_lives = nes[R.LIVES]
        self._last_lives = self._start_lives
        self._save_state = nes.save()
        self._img = np.zeros((IMG_STACK, IMG_H, IMG_W), dtype=np.uint8)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def _died(self) -> bool:
        """Death flag, excluding the flag being set during a stage clear."""
        nes = self._nes
        return nes[R.DEATH] != 0 and nes[R.STAGE_COMPLETE] == 0

    def _warmup(self) -> None:
        """Play a few random frames so episodes do not start identically.

        Only harmless inputs (right / right+jump / nothing) are used: the
        agent still starts at the beginning of the stage, just at a
        slightly different point in the run-up.
        """
        # 2026-10-18: the boss arena has no room for a run-up -- the boss
        # walks toward Higgins while the warm-up runs him right, so a
        # 90-frame warm-up turned a safe 91 px into a lethal 6 px and the
        # practice episodes were dead on the first decision.  Once the boss
        # is on screen the warm-up only idles or jumps: still enough to
        # de-synchronise the parallel worlds, but it does not close in.
        # 2026-10-19: with idle/jump only, an arena episode could never
        # stumble onto a hit during warm-up, and -- much worse -- the
        # policy had no reason to ever throw there: measured on every
        # checkpoint we have, it scores 8/8 hits in the hand-made practice
        # snapshot and exactly 0 in every REAL arrival frame.  So the
        # warm-up now also throws (never RIGHT: closing in is what kills).
        n = int(self.rng.integers(0, WARMUP_MAX + 1))
        for _ in range(n):
            if self._boss_on_screen():
                self._nes.controller = int(self.rng.choice(
                    (0x00, NES_INPUT_A, NES_INPUT_B,
                     NES_INPUT_A | NES_INPUT_B)))
            else:
                self._nes.controller = int(self.rng.choice(
                    (0x00, NES_INPUT_RIGHT, NES_INPUT_RIGHT | NES_INPUT_A)))
            self._nes.step()
        self._nes.controller = 0

    def reset(self, seed: Optional[int] = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # --- pick the stage for this episode --------------------------
        if self.stage_pool:
            i = int(self.rng.choice(len(self.stage_pool),
                                    p=self.stage_weights))
            self._stage_idx = (i if self.stage_indices is None
                               else self.stage_indices[i])
            state = self.stage_pool[i]
        else:
            state = self._save_state

        # --- warm up, re-rolling if it kills us -----------------------
        # tools/probe_all_stages.py: on 20 of the 32 stages, simply
        # holding RIGHT for 4 s is fatal.  The warm-up is the ONLY thing
        # breaking the game's determinism (without it all 8 parallel
        # worlds run byte-identical trajectories), so instead of setting
        # its length to 0 we re-roll it: a warm-up death is discarded and
        # never becomes an episode.
        for attempt in range(WARMUP_RETRIES + 1):
            self._nes.load(state)
            if getattr(self, "_activity", None) is not None:
                self._activity.reset()      # new episode: disarm zombies
            self._warmup()
            if not self._died():
                break
            if attempt == WARMUP_RETRIES:
                self._nes.load(state)       # last resort: clean start
                if getattr(self, "_activity", None) is not None:
                    self._activity.reset()
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
        self._last_inv = any(
            int(self._nes[R.OBJ.id + R.OBJ.STRIDE * s]) == FAIRY_ID
            for s in range(0, 16))
        self._last_area = self._nes[R.AREA]
        self._last_round = self._nes[R.ROUND]
        self._last_checkpoint = self._nes[R.CHECKPOINT]
        boss_hits, boss_beaten = self._boss_hit_count()
        self._last_boss_hits = boss_hits
        self._boss_beaten = boss_beaten
        self._boss_aim_cd = 0
        self._steps = 0
        self._boss_fight = False
        self._prev_boss_fight = False     # to catch the arrival frame
        self._boss_act_step = 0      # for the boss-specific stall clock
        self._boss_dx_best: float | None = None   # None = not seen yet
        self._x_progress = 0.0
        self._best_progress = self._last_progress
        self._best_step = 0
        self._prev_enemies = {}
        self._hit_pairs: set[tuple[int, int]] = set()
        self._aim_cd = 0
        self._action = 0
        self._kills = 0
        self._shot_steps = 0
        self._prev_stage_complete = self._nes[R.STAGE_COMPLETE]
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

    def _boss_on_screen(self) -> bool:
        """Is the 1-4 boss (id 67 body / 198 head) loaded in any slot?

        The user identified it from a screenshot (2026-10-18): one creature,
        two slots.  They also tested by hand that axes at the BODY do
        nothing while hits to the HEAD hurt it.
        """
        return self._boss_head_slot() is not None

    def _boss_head_slot(self):
        """Slot holding the boss HEAD (id 198), or None.

        Everything boss-related is read relative to this slot: the head is
        the only part that can be damaged, and its field block 16 is the
        hit counter (see BOSS_HEAD_HIT_BLOCK).
        """
        nes = self._nes
        for s in range(1, N_SLOTS_SCANNED):
            if int(nes[R.OBJ.id + R.OBJ.STRIDE * s]) == R.BOSS_HEAD_ID:
                return s
        return None

    def _boss_head_dx(self):
        """Screen-space dx from the player to the boss head, or None.

        Object x is SCREEN space (ram_map docstring), so a plain difference
        is the on-screen distance -- exactly what the axe has to cover.
        """
        s = self._boss_head_slot()
        if s is None:
            return None
        nes = self._nes
        dx = int(nes[R.OBJ.x_lo + R.OBJ.STRIDE * s]) - int(nes[R.OBJ.x_lo])
        # wrap into (-128, 128]
        if dx > 128:
            dx -= 256
        elif dx < -128:
            dx += 256
        return dx

    def _player_airborne(self) -> bool:
        """True when Higgins is off the ground (screen y smaller = higher).

        Measured: standing on the floor y_lo = 144, a standing jump bottoms
        out at ~110, a running jump at ~70.
        """
        return int(self._nes[R.OBJ.y_lo]) < 138

    def _boss_hit_count(self) -> tuple[int, bool]:
        """(hits landed on the head so far, is the boss already beaten).

        Returns (-1, False) when there is no boss on screen at all, so a
        stage without a boss can never pay the reward.
        """
        s = self._boss_head_slot()
        if s is None:
            return -1, False
        nes = self._nes
        hits = int(nes[BOSS_HEAD_HIT_BLOCK + s])
        beaten = int(nes[R.OBJ.id + R.OBJ.STRIDE * s]) != R.BOSS_HEAD_ID \
            or any(int(nes[R.OBJ.id + R.OBJ.STRIDE * t]) == R.BOSS_BEATEN_ID
                   for t in range(1, N_SLOTS_SCANNED))
        return hits, beaten

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
            float(nes[R.WEAPON] > 0),      # holding any weapon?
            # $006D is a weapon TYPE (0 none / 1 axe / 2 fireball) and both
            # projectiles share object id 66 -- without this bit the policy
            # cannot tell a fireball from an axe.  The fireball reaches
            # ~270 px vs ~186 px and is the only thing that breaks rocks.
            float(nes[R.WEAPON] >= 2),
            # The skateboard is a 0/1 flag (verified: $006E only ever
            # reads 0 or 1 -- tools/probe_skateboard.py).  Scaling it by
            # 1/255 turned it into 0.0 / 0.0039, i.e. a feature too small
            # for the net to ever notice, which hid the fact that the flag
            # means "one free hit".  Feed it as a clean 0/1 instead.
            float(nes[R.SKATEBOARD] > 0),
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

        # Which stage is this?  Appended LAST so that widening an old
        # checkpoint is a plain concat of zeros (tools/widen_net.py).
        stage = np.zeros(N_STAGES, dtype=np.float32)
        stage[int(self._stage_idx) % N_STAGES] = 1.0

        # 2026-10-17: object id 199 is the white "fairy puff" that trails
        # the player while INVINCIBLE.  Decisive test
        # (tools/probe_fairy_inv.py): with 199 on stage an enemy pinned
        # onto the player does NOT set the death flag (2/2); without it
        # the same poke kills (2/2).  So "slot 199 present" IS the
        # invincibility state -- no timer byte exists in RAM.
        inv = any(int(nes[R.OBJ.id + R.OBJ.STRIDE * s]) == 199
                  for s in range(0, 16))
        return np.concatenate(
            [player, enemies, glob, kind, stage,
             np.array([float(inv)], dtype=np.float32)])

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
        score = 0.0
        for ss, sx, sy in shots:
            for fs, fx, fy in foes:
                dx, dy = abs(sx - fx), abs(sy - fy)
                if dx < NEAR_HIT_X and dy < NEAR_HIT_Y:
                    key = (ss, fs)
                    live.add(key)
                    if key not in self._hit_pairs:
                        # 1.0 dead centre, 0 at the rim of the window --
                        # a near miss now pays strictly less than a hit.
                        score += 1.0 - max(dx / NEAR_HIT_X, dy / NEAR_HIT_Y)
        self._hit_pairs = live
        return min(score, 2.0)       # never pay for more than 2 per step

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
        #
        # PAY FOR NEW GROUND ONLY (2026-10-18).  The old rule banked every
        # small forward delta and skipped any |dp| >= 64 jump ("respawn
        # teleport"), which made the reward HACKABLE: on 1-4 the policy
        # bounces ~100 px forward and straight back (tools/diag_14_deaths.py,
        # measured: screen x pinned at 96, camera swinging 60<->80, 13
        # cycles in one life).  The outbound leg is a big negative jump and
        # was therefore free, while the walk back was paid for -- 89 px of
        # reward per cycle, standing still.  It inflated the reported
        # x_progress 2.4x (1911 px claimed on a run whose true maximum was
        # 744 px) and, worse, the agent was being trained to farm it.
        #
        # Now: a step only pays when it sets a new furthest point, so
        # oscillating pays nothing at all; and going backwards is charged
        # on EVERY pixel (clamped, so a respawn teleport cannot produce an
        # enormous negative spike).
        progress = self._cam.progress(nes)
        dp = progress - self._last_progress
        self._last_progress = progress
        if progress > self._best_progress:
            new_ground = progress - self._best_progress
            self._best_progress = progress
            self._best_step = self._steps
            progress_r = float(np.clip(new_ground, 0, 32)) * 0.02
        elif dp < 0:
            # Walking BACKWARDS must be clearly worse than standing still,
            # otherwise fleeing left is cheaper than facing the snail ahead
            # (measured 2026-10-13: the ppo_v3 policy learned to hold
            # left+jump against the left screen edge for 1000+ frames
            # instead of advancing -- see tools/probe_window_traj.py).
            progress_r = float(np.clip(dp, -32, 32)) * 0.02 * BACKWARD_PENALTY
        else:
            progress_r = 0.0
        # --- boss arena: "walking right" must stop paying --------------
        # 2026-10-18, straight from the user: once the boss is on screen,
        # paying for ground gained teaches the runner to walk straight
        # into the boss and die.  Progress is still MEASURED honestly
        # (best_progress keeps updating, so eval stays truthful) but it
        # pays nothing while 67/198 are on screen -- from here the only
        # way to score is to actually hurt the boss (and to stay alive).
        self._boss_fight = self._boss_on_screen()
        boss_arrived = self._boss_fight and not self._prev_boss_fight
        if boss_arrived:
            # First frame with 67/198 on screen.  The boss stall clock must
            # start NOW: it is still 0 from reset() (or from the last
            # arena), so without this the very first boss frame reads
            # "idle = steps - 0 >= 600" and the episode is cut the instant
            # the boss appears -- which is exactly why a full 1-4 run
            # reached 5414 px and then ended with stalled=True, 0 hits.
            self._boss_act_step = self._steps
        if self._boss_fight:
            progress_r = 0.0
        # x_progress is now the honest furthest point, not a sum of legs.
        self._x_progress = self._best_progress

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

        # --- stage clear ---------------------------------------------
        # plan.md phase 1: clearing the flag IS the training target, not a
        # side effect.  Fires on the 0 -> non-zero edge of $003C so it is
        # paid exactly once.
        sc = nes[R.STAGE_COMPLETE]
        cleared = sc != 0 and self._prev_stage_complete == 0
        clear_r = CLEAR_GAIN if cleared else 0.0

        # --- stall detection -----------------------------------------
        # The second way the policy dodged the snails was to simply stop:
        # hold left+jump against the screen edge and burn the clock.  Left
        # alone that is a stable strategy (only -0.001/step), so an episode
        # with no new best progress for STALL_STEPS decisions is ended with
        # a penalty.  "Standing still" is then never better than trying.
        # _best_progress / _best_step are already maintained by the
        # "new ground only" block above (2026-10-18); this only reads them.
        # During a boss fight the camera is locked, so "no new progress" is
        # the normal state, not stalling (the arena would otherwise end
        # after 250 steps, well before the 8 hits are possible) -- there the
        # clock measures inactivity since the last useful action instead.
        if self._boss_fight:
            idle = self._steps - self._boss_act_step
            stalled = idle >= BOSS_STALL_STEPS
        else:
            stalled = (self._steps - self._best_step) >= STALL_STEPS
        stall_r = -STALL_PENALTY if stalled else 0.0

        # --- the important part: picking up the axe -------------------
        weapon_r = 0.0
        fire_gained = False
        if weapon > self._last_weapon and not area_changed and not dead:
            weapon_r = WEAPON_GAIN
        # axe -> FIREBALL upgrade ($006D 1 -> 2).  Same object id 66 in the
        # air, but longer reach and the only thing that breaks rocks.
        if (self._last_weapon < 2 <= weapon
                and not area_changed and not dead):
            weapon_r += FIRE_GAIN
            fire_gained = True
        skate_r = 0.0
        skate_gained = skate_lost = False
        if not area_changed and not dead:
            if skate > self._last_skate:
                skate_r = SKATE_GAIN
                skate_gained = True
            elif skate < self._last_skate:
                # The board is gone but we are still alive: it ate a hit
                # for us.  This is the only signal that teaches what the
                # board is for -- without it the flag only ever *appears*.
                skate_r = SKATE_SAVE
                skate_lost = True

        # --- extra life (1UP), see LIFE_GAIN above --------------------
        life_r = 0.0
        life_gained = False
        if lives > self._last_lives and not area_changed and not dead:
            life_r = LIFE_GAIN
            life_gained = True

        # --- invincibility (fairy puff 199), see INV_GAIN above -------
        inv = any(int(nes[R.OBJ.id + R.OBJ.STRIDE * s]) == FAIRY_ID
                  for s in range(0, 16))
        inv_r = 0.0
        inv_gained = False
        if inv and not self._last_inv and not area_changed and not dead:
            inv_r = INV_GAIN
            inv_gained = True

        # Killed something?  Paid once per dead enemy slot.
        kills = 0 if dead else self._count_kills()
        if dead:
            self._prev_enemies = {}
            self._shot_steps = 0
            self._hit_pairs = set()
        kill_r = KILL_GAIN * kills

        # --- boss damage: head hit counter (see BOSS_HEAD_HIT_BLOCK) ---
        # Only an INCREASE of the head's block-16 counter pays, and only
        # while the head is really on screen.  A reload (area_changed) or a
        # death resets the object table, which must not pay.
        boss_hits_now, boss_beaten = self._boss_hit_count()
        boss_hits = 0
        if (not area_changed and not dead and boss_hits_now >= 0
                and self._last_boss_hits >= 0
                and boss_hits_now > self._last_boss_hits):
            boss_hits = boss_hits_now - self._last_boss_hits
        boss_r = BOSS_HIT_GAIN * boss_hits
        if boss_arrived and not dead:
            # Reaching the arena has to be worth something ON ITS OWN.
            # While the stall-clock bug was in place (see BOSS_STALL_STEPS)
            # every arrival was punished with -1.0, so the policy learned
            # to stop ~9 camera units short and stand there: it measured
            # 2026-10-18 that the same snapshot advances cam 222 -> 231 in
            # 145 frames when simply told to run right and jump.  One-shot
            # per episode, so it cannot be farmed.
            boss_r += BOSS_ARRIVE_GAIN
        boss_killed = False
        if (not area_changed and boss_beaten and not self._boss_beaten):
            boss_killed = True
            boss_r += BOSS_KILL_GAIN
        boss_hp = boss_hits_now

        # --- spacing: walk into the band where an axe can reach ----------
        # Outside the band there is no reward at all, so "get closer" used
        # to have zero gradient.  Pay once per NEW CLOSEST APPROACH (like
        # the progress reward pays only for new ground), so shuffling back
        # and forth cannot farm it.
        boss_approach_r = 0.0
        if not dead and self._boss_fight:
            dxh = self._boss_head_dx()
            if dxh is not None and dxh > BOSS_AIM_MAX \
                    and dxh < BOSS_APPROACH_MAX:
                if self._boss_dx_best is None:
                    self._boss_dx_best = dxh       # first sighting: no pay
                else:
                    gain = self._boss_dx_best - dxh
                    if gain > 0:
                        boss_approach_r = BOSS_APPROACH_GAIN * min(gain, 8.0)
                        self._boss_dx_best = dxh

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

        # Boss fight shaping, graded so there IS a gradient all the way
        # from "runs into the boss and dies" to "lands a head hit":
        #   1. spacing  - stop in the band where an axe can reach (the axe
        #                 only flies ~57 px and touching the boss is fatal)
        #   2. airborne - be off the ground there (a ground throw peaks at
        #                 y=136, the head is at y=93, so it can never hit)
        #   3. release  - actually throw while airborne in the band
        # All three share one cooldown, so camping at the right distance
        # pays ~0.003/step -- negligible next to real progress.
        self._boss_aim_cd = max(0, self._boss_aim_cd - 1)
        boss_close_r = 0.0
        if not dead and self._boss_fight:
            dxh = self._boss_head_dx()
            if dxh is None:
                pass
            elif dxh < BOSS_TOO_CLOSE:
                boss_close_r = -BOSS_CLOSE_PENALTY
            elif dxh < BOSS_AIM_MAX:
                boss_close_r = BOSS_HOLD_GAIN
        if (not dead and self._boss_fight and self._boss_aim_cd == 0):
            dxh = self._boss_head_dx()
            if dxh is not None and BOSS_AIM_MIN < dxh < BOSS_AIM_MAX:
                threw = bool(ACTIONS[self._action][0] & NES_INPUT_B)
                if threw and self._player_airborne():
                    aim_r += BOSS_AIM_GAIN          # the real thing
                elif threw:
                    aim_r += 0.15 * BOSS_AIM_GAIN   # right place, too low
                else:
                    aim_r += 0.05 * BOSS_AIM_GAIN   # just holding the range
                self._boss_aim_cd = BOSS_AIM_CD
                self._boss_act_step = self._steps

        # Any real damage also feeds the boss clock.
        if boss_hits:
            self._boss_act_step = self._steps

        reward = (progress_r + score_r + stamina_r + death_r
                  + weapon_r + skate_r + life_r + inv_r + kill_r + near_r
                  + aim_r + stall_r + clear_r + boss_r + boss_approach_r
                  + boss_close_r - 0.001)
        self._last_progress = progress
        self._last_score = score
        self._last_stamina = sta
        self._last_weapon = weapon
        self._last_skate = skate
        self._last_lives = lives
        self._last_inv = inv
        self._last_area = nes[R.AREA]
        self._last_round = nes[R.ROUND]
        self._last_checkpoint = nes[R.CHECKPOINT]
        self._last_boss_hits = boss_hits_now
        self._boss_beaten = boss_beaten
        self._prev_boss_fight = self._boss_fight
        self._prev_stage_complete = sc

        info = {"x_progress": self._x_progress, "score": score,
                "stamina": sta, "lives": lives, "death": bool(dead),
                "weapon": weapon, "got_axe": bool(weapon_r),
                "got_fire": fire_gained, "fire": int(weapon >= 2),
                "got_skate": skate_gained,
                "skate": int(skate > 0),
                "skate_saved": skate_lost,
                "got_life": life_gained,
                "got_inv": inv_gained, "invincible": int(inv),
                "stalled": bool(stalled),
                "kills": kills, "near": near, "cleared": bool(cleared),
                "boss_hp": boss_hp, "boss_hits": boss_hits,
                "boss_killed": boss_killed,
                "boss_fight": self._boss_fight,
                "stage": (self.stage_labels[self._stage_idx]
                          if self.stage_pool else None),
                "best_progress": self._best_progress}
        return reward, info, dead or stalled or (
            cleared and self.clear_ends_episode)


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
