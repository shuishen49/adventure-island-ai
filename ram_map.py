"""Reverse-engineered RAM map for Adventure Island (Chinese hack, MMC1).

Object slots (verified 2026-10-13, 16 slots, STRIDE = 1):

The disassembly uses a struct-of-arrays layout: each field is a 16-byte
array whose elements are CONSECUTIVE (stride 1), and the arrays start
0x10 bytes apart:

  $0564 + n   obj_id          (player = 64, e.g. 28 = flying snail)
  $0574 + n   pos_X fraction
  $0584 + n   pos_X low
  $0594 + n   pos_X high
  $05A4 + n   spd_X           signed; 47-51 running right, -3 left
  $05B4 + n   pos_Y fraction
  $05C4 + n   pos_Y low
  $05D4 + n   gravity
  $0624 + n   anim timer

  further field arrays: $0604 $0614 $0634 $0644 $0654 $0664

Slot 15 is a NORMAL object slot (corrected 2026-10-13).  Its x_lo byte
($0593) was long believed to be the scroll register because it DECREASES
by the player's per-frame advance -- but that is just what happens to any
world-fixed object's screen x while the camera scrolls.  Measured with
tools/probe_snailslot.py: whenever slot 15's 3-byte decode lands in
0..255 it sits on a real ground snail 99.3% of the time (5782/5823
frames, 41 empty).  $0593 stays constant while the player is frozen, and
slot 15's y_lo tracks real objects (160 on the ground, 88 for a flying
fruit), so the "camera object" story was a misidentification.  $0593 is
still read as a progress proxy (CameraTracker) -- it works because the
object in slot 15 is usually world-fixed; do not change it mid-training.

Static addresses (all verified):
  $0037  area        $0038 round      $0039 checkpoint
  $003F  lives (starts at 3, decrements on death)
  $0072  death type  (1 at death instant, 255 during respawn anim)
  $0076  stamina/health bar (starts 10, drains over time)
  $006D  weapon      $006E skateboard
         (verified 2026-10-13 with tools/probe_weapon.py: $006D flips
          0 -> 1 at the exact frame an id-28 egg is touched and stays 1
          afterwards -- that IS the axe.  It goes back to 0 on death.)
  $069A  score (BCD-ish, increments when eating fruit)
  $007A  stage complete flag

IMPORTANT (verified 2026-10-12):
  obj_pos_X ($0594:$0584:$0574) is SCREEN-space, not world-space.  While
  running, the camera keeps the player parked at screen x = 96, so the
  "world X" barely moves.  The real scroll register is $0593 (8-bit,
  DECREASING by the player's per-frame advance, wraps every 256 px).
  World progress = accumulated (-camera delta) + (screen_x - screen_x0).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjSlot:
    id: int = 0x0564
    x_fr: int = 0x0574
    x_lo: int = 0x0584
    x_hi: int = 0x0594
    spd_x: int = 0x05A4
    y_fr: int = 0x05B4
    y_lo: int = 0x05C4
    gravity: int = 0x05D4

    STRIDE: int = 1

    CAMERA_SLOT: int = 15       # MISNOMER: slot 15 is a normal object slot
                                # (see module docstring); this name predates
                                # the 2026-10-13 correction


OBJ = ObjSlot()

N_OBJ_SLOTS = 16          # $0564 .. $0694 region

AREA = 0x0037
ROUND = 0x0038
CHECKPOINT = 0x0039
LIVES = 0x003F
DEATH = 0x0072
STAMINA = 0x0076
# $006D is a weapon TYPE, not a boolean (romdetectives RAM map + measured
# 2026-10-17 with tools/probe_fireball.py): 0 = unarmed, 1 = stone axe,
# 2 = magical fireball.  Both projectiles share object id 66, so the ONLY
# way to tell them apart is this byte.  Fireball flies ~270 px vs the
# axe's ~186 px and is the only weapon that destroys rocks / boulders.
WEAPON = 0x006D
SKATEBOARD = 0x006E
BTN_A_FLAG = 0x0071
BTN_B_FLAG = 0x0073
STAGE_COMPLETE = 0x007A
SCORE = 0x069A
HISCORE = 0x06A3

CAMERA = 0x0593           # 8-bit scroll, decreasing; see module docstring

PLAYER_SLOT = 0
PLAYER_ID = 64

# Object id semantics.
#   - 14 / 16 / 17 / 28 came from the user.
#   - the rest were classified automatically by tools/profile_ids.py, which
#     played ~320k frames of random input and recorded, for every id:
#     how often it appears, whether it moves, where it sits relative to
#     the player, and what happens on contact (death / damage / score).
# ---------------------------------------------------------------------------
# Object id semantics.
#
# Sources, in order of trustworthiness:
#   1. the user (id 14 = enemy, 16/17 = fruit, 28 = item, static "obj" =
#      obstacles that cost stamina but do not kill)
#   2. a trajectory study (tools/probe_66b.py, 132k frames): every object
#      slot's lifetime was logged -- where it was BORN relative to the
#      player, which way it moved, and how long it lived.  Things that are
#      born at the player (dx ~ 0) are yours; things born at the screen
#      edge (dx ~ +223) and walking left are hostile.
#
# NOTE: an earlier pass labelled id 66 an enemy because "stamina dropped
# after contact".  That signal is worthless -- stamina drains on its own
# and id 66 sits on top of you, so it always scored a hit.  It is in fact
# your own axe: born at dx=+2, dy=-8, constant speed +8, weapon=1.
# ---------------------------------------------------------------------------
ID_ENEMY = 14             # snail / crawler: born off-screen, walks left
ID_FRUIT_A = 16           # fruit: eat it before it vanishes = score
ID_FRUIT_B = 17
ID_ITEM = 28              # power-up egg: axe ($006D), skateboard, heart ...
ID_PROJECTILE = 66        # YOUR OWN axe -- born on the player, flies right

FRUIT_IDS = (ID_FRUIT_A, ID_FRUIT_B)
ITEM_IDS = (ID_ITEM,)

# Born at the screen edge and moving towards the player.
# (4 was flagged by the flawed stamina heuristic and has not been seen
#  since -- kept, but marked doubtful in the overlay.)
# 35/37/41: the octopus family of the water stages (user-identified from
# the overlay screenshots, 2026-10-17).  41 takes two axe hits to die.
#
# 7 / 10 / 32 / 56 -- the stage 1-4 (forest) cast, named by the user from
# the annotated frames on 2026-10-18:
#     7  FROG     hops along the ground
#    10  CROW     the red bird that flies in on the 1600-2000 px stretch
#    32  CHARGER  spawns BEHIND the player and runs him down
#    56  SPIDER   drops on a thread and bobs up and down
# All four were invisible to the policy before this: the death census
# (tools/diag_14_deaths.py, 16 episodes) found id 10 next to the corpse in
# 7 of 16 deaths and id 7 in 4, yet neither was in this tuple -- so the
# "nearest enemy" feature never pointed at them and the axe-aiming reward
# never fired on them.  Killing one is worth a throw, so they go in.
# 5 -- the end-of-stage killer of 1-4 (2026-10-18).  Found by the blind
# death census (tools/diag_killer_id.py on ppo_14c, 12 episodes: 6 of 12
# deaths ended overlapping id 5, all of them on the 4385..4872 px stretch)
# and then nailed down with tools/probe_id5.py:
#   * contact is LETHAL: walking right into it with the invulnerability
#     cheat switched off kills on frame 43;
#   * 900 frames of jump-throw spam leave it on screen -> not (yet) killed;
#   * it drifts vertically ~32 px and slides horizontally, and shows up
#     only in the last ~1200 px of the stage.
# It was NOT in ENEMY_IDS, so the policy could not see it and never aimed
# at it -- the same blind spot that cost us 7/10/32/56.  Added 2026-10-18.
# Not yet named from a picture; ask the user before trusting the label.
# --- id 36 : 2-1 killer, found 2026-10-19 -------------------------------
# Same blind-spot story as 5/7/10/32/56.  Found with
# tools/diag_killer_id.py --start-round 4 --stage-idx 4 on runs/ppo_21b:
# 6 of 12 deaths had id 36 hugging the player ON THE DEATH FRAME
# (dx 15..22, dy 0..16) at ~4770..4822 px, with nothing else lethal nearby
# (the other slots there are 16 = fruit and 59 = platform, both harmless).
# The other 6 deaths were id 14 (snail, already known) at ~4380..4415 px.
# NOT yet named from a picture -- ask the user before trusting any label.
ENEMY_IDS = (ID_ENEMY, 9, 4, 35, 37, 41, 7, 10, 32, 56, 5, 36)

# --- the 1-4 boss -------------------------------------------------------
# Named by the user from a screenshot (2026-10-18): the end-of-stage boss
# is ONE creature drawn with TWO object slots --
#     67  = the body (the tall thing on the ground)
#     198 = the head (floats above the body)
# Contact is lethal (tools/probe_boss_pair.py: standing next to them kills
# on frame 33).  The user tested by hand: throwing axes at the BODY does
# nothing, hitting the HEAD is what hurts it.
# They only appear in the last stretch of 1-4 (first seen at ~5173 px).
BOSS_BODY_ID = 67
BOSS_HEAD_ID = 198
BOSS_IDS = (BOSS_BODY_ID, BOSS_HEAD_ID)
# id 63 = the boss AFTER it is beaten: the body turns into 63 the frame the
# fight ends (measured 2026-10-18, outputs/boss_win.npy frame 3308 -- same
# frame $007A stage-complete flips to 1 and the head floats off the top).
BOSS_BEATEN_ID = 63

# Lethal on contact: touching one of these costs a LIFE.  Measured with
# tools/probe_id4.py (agent-driven survey, 6 episodes): id 2 is completely
# static (vx = 0), sits exactly on the ground line (dy = +7, same as the
# player), and contact kills -- the user confirmed the campfire is lethal,
# it is NOT the "costs stamina" kind of obstacle.  Keep it out of
# ENEMY_IDS: an axe cannot destroy it, so treating it as a target would
# just make the agent waste throws on it.
LETHAL_STATIC_IDS = (2,)

# Static things that only drain stamina on contact without killing.
# (18 and 65 were dropped: the catalogue shows 18 as a floating heart and
#  65 spawns on a strict 768px grid -> checkpoint sign, not an obstacle.)
OBSTACLE_IDS = (1, 2, 19, 20, 25, 48)

# SELF: born on the player -- your own shot, its muzzle flash, shadow, ...
# id 69 (measured 2026-10-17, tools session: model playthrough of 1-1):
# spawns near the player's feet ONLY after he stands still a few frames,
# drifts LEFT ~20 px over ~16 game frames at ground level (y 156-163),
# harmless: no collision effect observed.  Matches the "stand still and a
# cloud appears" bonus-stage mechanic (hidden warp cloud), so labelled
# CLOUD? -- it is NOT part of the player.  Keep it out of ENEMY_IDS.
SELF_IDS = (43, 69, ID_PROJECTILE)

# Harmless decoration / never observed in the trajectory study.
STATIC_IDS = (3, 42, 54, 57, 59, 199)

# kind name -> (label, RGB) used by the annotation overlay
OBJ_KINDS = {
    ID_ENEMY: ("ENEMY", (255, 70, 70)),
    9: ("ENEMY", (255, 70, 70)),
    4: ("ENEMY?", (255, 110, 90)),      # doubtful, see note above
    # id 35 -- the OCTOPUS.  Named by the user from the picture (2026-10-17)
    # and confirmed by motion: it cycles vertically between y = 115 and
    # y = 240 (a 124 px sweep, measured over 300 frames of stage 1-2), which
    # no scenery or pickup does.  Contact is lethal per the game's own rules
    # (octopus = one of the "everything kills in one hit" enemies).  It was
    # NOT in ENEMY_IDS before, which meant the policy literally could not
    # see it -- no nearest-enemy feature, no throw target -- on the water
    # stages where it is the main obstacle.  Added 2026-10-17.
    35: ("OCTOPUS?", (255, 90, 160)),
    # ids 37 / 41 -- more octopuses, same family (user-identified from the
    # overlay screenshots, 2026-10-17: "?37 也是敌人，?41 也是章鱼，要打
    # 两次才死").  Added to ENEMY_IDS at the same time as 35.
    37: ("OCTOPUS?", (255, 90, 160)),
    41: ("OCTO2HP?", (200, 90, 255)),       # two hits to kill, per the user
    ID_FRUIT_A: ("FRUIT", (255, 230, 60)),
    ID_FRUIT_B: ("FRUIT", (255, 230, 60)),
    # --- the 1-4 forest cast (user-named 2026-10-18) ------------------
    # Colours are picked to be separable from the player's green and from
    # the existing ENEMY red, so an annotated frame can be read at a glance.
    7: ("FROG", (120, 230, 140)),        # 青蛙: hops on the ground
    10: ("CROW", (150, 130, 255)),       # 乌鸦: the red bird of 1600-2000 px
    32: ("CHARGER", (255, 60, 120)),     # 从背后冲过来的敌人
    56: ("SPIDER", (255, 150, 60)),      # 蜘蛛: 上下移动
    # id 5 -- the lethal, apparently unkillable thing that guards the last
    # ~1200 px of 1-4.  Name TBD (waiting on a picture from the user); the
    # colour is picked to stand out from every other kind.
    5: ("END5?", (255, 40, 255)),
    # user identified from a screenshot (2026-10-18): 67 = the tall boss's
    # BODY, 198 = its floating HEAD -- two slots, one creature.  Contact is
    # lethal (probe_boss_pair.py: standing still next to them = death f33).
    67: ("BOSS_BODY", (255, 90, 90)),
    198: ("BOSS_HEAD", (255, 220, 90)),
    ID_ITEM: ("ITEM", (180, 120, 255)),
    PLAYER_ID: ("PLAYER", (60, 255, 60)),
    ID_PROJECTILE: ("SHOT", (90, 220, 255)),
    43: ("SCORE?", (255, 255, 180)),    # catalogue shows "+50" popups
    69: ("SELF", (150, 230, 255)),
    1: ("BLOCK", (255, 160, 40)),
    # id 2 = THE CAMPFIRE.  Measured (tools/probe_id4.py): vx = 0 (static),
    # dy = +7 (on the ground line, same as the player), and contact costs a
    # LIFE -- not a "drains some stamina" obstacle.  Drawn in fire red so it
    # cannot be confused with the harmless blocks.  The "?" stays until the
    # user confirms it from the picture (outputs/campfire.png).
    2: ("FIRE?", (255, 80, 30)),
    # id 13 -- found in the 9M-step death census (tools/death_cause.py,
    # 8 episodes: 2 of 8 deaths).  Red grid-like sprite, sits on the ground
    # line (y = 160, dy ~ +30 as seen from a jumping player), drifts with
    # the scenery and contact kills.  Deliberately NOT added to ENEMY_IDS
    # until the user names it: if it is another fire the axe is useless
    # against it and it must stay out of the throw-target set.
    13: ("FIRE2?", (255, 130, 20)),
    19: ("BLOCK?", (255, 160, 40)),     # never visible in catalogue crops
    20: ("BLOCK", (255, 160, 40)),
    25: ("SCORE?", (255, 255, 180)),    # catalogue shows "+50" popups
    48: ("BLOCK", (255, 160, 40)),
    18: ("ITEM?", (200, 140, 255)),     # catalogue shows a floating heart
    65: ("CHECK", (255, 255, 255)),     # spawns every 768px = checkpoint
    3: ("DECO", (170, 170, 170)),
    42: ("DECO", (170, 170, 170)),
    54: ("DECO", (170, 170, 170)),
    57: ("DECO", (170, 170, 170)),
    # id 59 -- the CLOUD PLATFORM (named from the user's screenshot,
    # 2026-10-17: Higgins standing on one over the sea in 1-2).  NOT mere
    # decoration: measured over 400 frames of 1-2, several slots drift
    # LEFTWARD from x=320 at DIFFERENT speeds, each at a fixed height
    # (y = 136 / 152 / 168).  They are the rideable moving clouds of the
    # water stages.  Harmless to touch, so it stays out of the enemy and
    # obstacle sets; the CNN sees the cloud pixels, so no extra obs
    # feature was added for it (yet).
    59: ("PLAT?", (120, 220, 255)),
    # id 58 -- the second moving platform (user-identified from the overlay
    # screenshot, 2026-10-17: "?58 是一个左右移动的平台").  Same family as 59
    # but patrols LEFT-RIGHT instead of drifting left.  Rideable, harmless:
    # stays out of enemy / obstacle sets; the CNN sees its pixels, so no obs
    # feature (yet -- a nearest-platform feature is the real upgrade, not a
    # label).
    58: ("PLAT?", (120, 220, 255)),
    # id 199 -- the white "fairy puff" that trails the player while
    # INVINCIBLE.  Decisive test 2026-10-17 (tools/probe_fairy_inv.py):
    # with 199 on stage an enemy pinned onto the player does NOT set the
    # death flag (2/2); without it the same poke kills (2/2).  Only seen
    # on 1-3.  The puff cannot be "eaten" -- it is the active effect.
    199: ("FAIRY", (255, 255, 120)),
}
UNKNOWN_KIND = ("?", (200, 200, 200))


def kind_of(oid: int) -> tuple[str, tuple[int, int, int]]:
    """Return (label, colour) for an object id."""
    return OBJ_KINDS.get(oid, UNKNOWN_KIND)

# Y value of the player while standing on the ground (16.16 -> ~2432 raw)
GROUND_Y_RAW = 2432


def read_obj_world_x(nes, slot: int) -> int:
    hi = nes[OBJ.x_hi + OBJ.STRIDE * slot]
    lo = nes[OBJ.x_lo + OBJ.STRIDE * slot]
    fr = nes[OBJ.x_fr + OBJ.STRIDE * slot]
    return (hi << 16) | (lo << 8) | fr


def read_obj_world_y(nes, slot: int) -> int:
    lo = nes[OBJ.y_lo + OBJ.STRIDE * slot]
    fr = nes[OBJ.y_fr + OBJ.STRIDE * slot]
    return (lo << 8) | fr


class CameraTracker:
    """Tracks world progress via the 8-bit scroll register at $0593.

    The camera register DECREASES by the player's per-frame advance and
    wraps every 256 px, so we accumulate (-delta) with wrap detection.
    Total progress = accumulated camera travel + (screen_x - screen_x0);
    screen_x handles the initial stretch before the camera starts
    following (player screen x moves 80 -> 96 while camera is still 0).
    """

    def __init__(self, nes) -> None:
        self.reset(nes)

    def reset(self, nes) -> None:
        self._cam_prev = nes[CAMERA]
        self._travel = 0.0
        self._sx0 = nes[OBJ.x_lo]

    def progress(self, nes) -> float:
        cam = nes[CAMERA]
        d = (self._cam_prev - cam) % 256
        if d > 128:                      # reverse wrap (respawn etc.)
            d -= 256
        self._travel += d
        self._cam_prev = cam
        return self._travel + (nes[OBJ.x_lo] - self._sx0)


def px(raw: int) -> float:
    """Convert 16.8 fixed point raw value to pixels.

    Verified: player spawns at raw 0x5000 -> 80 px, ground Y raw ~38912
    -> 152 px, which matches its on-screen pixel position (camera is at 0
    at the start of a stage, so world px == screen px).
    """
    return raw / 256.0


# --- zombie ("ghost") object slots -------------------------------------
# Verified 2026-10-13 with tools/probe_death_freeze.py + probe_active_flag.py:
# after a death/respawn the game KEEPS stale slot data forever -- every
# field byte freezes (including the anim timer $0624) and the game stops
# rendering the slot, but nothing is cleared.  A full 2 KB RAM diff found
# no dedicated active flag (probe_active_flag.py), so we detect zombies
# behaviourally: a live object always ticks (anim timer advances, the
# sub-pixel position accumulates), while a zombie's watched bytes stay
# bit-identical for dozens of frames.
SLOT_FIELD_BASES = (OBJ.id, OBJ.x_fr, OBJ.x_lo, OBJ.x_hi, OBJ.spd_x,
                    OBJ.y_fr, OBJ.y_lo, OBJ.gravity,
                    0x0604, 0x0614, 0x0624, 0x0634, 0x0644, 0x0654, 0x0664)


class SlotActivityTracker:
    """Detect object slots the game has stopped updating ("zombies").

    Zombies are ONLY produced by a death: the game freezes every slot at
    the death instant and never clears nor renders the stale ones again
    (verified with tools/probe_death_freeze.py).  Therefore detection is
    ARMED by a death (lives decrement or a non-zero $0072), never before:

      * before any death: nothing is ever hidden -- live enemies that
        merely stand still (a crawler can pause for minutes, verified)
        are never misjudged;
      * after a death: every present slot's signature is baselined; a
        slot whose signature NEVER changed since the baseline is a
        zombie.  Any change (even one frame of movement) clears it for
        good, so a live enemy that pauses and then moves stays visible.

    Call ``update(nes, slots)`` once per observation with the currently
    present slot indices; ``is_stale(s)`` then reports zombie status.
    Only apply it to enemy-kind objects: fruits/items can legitimately
    sit perfectly still while rendered.
    """

    def __init__(self) -> None:
        self._base: dict[int, tuple] = {}
        self._stale: set[int] = set()
        self._armed = False
        self._lives: int | None = None

    def reset(self) -> None:
        self._base.clear()
        self._stale.clear()
        self._armed = False
        self._lives = None

    def update(self, nes, slots) -> None:
        lives = nes[LIVES]
        if self._lives is None:
            self._lives = lives
        if lives < self._lives or nes[DEATH] != 0:
            self._armed = True
        self._lives = min(self._lives, lives)

        seen: set[int] = set()
        for s in slots:
            seen.add(s)
            sig = tuple(nes[b + OBJ.STRIDE * s] for b in SLOT_FIELD_BASES)
            if not self._armed:
                self._base[s] = sig
                continue
            if sig != self._base.get(s):
                self._base[s] = sig          # moved since baseline -> live
                self._stale.discard(s)
            else:
                self._stale.add(s)
        for s in set(self._base) - seen:
            self._base.pop(s, None)
            self._stale.discard(s)

    def is_stale(self, slot: int) -> bool:
        return slot in self._stale
