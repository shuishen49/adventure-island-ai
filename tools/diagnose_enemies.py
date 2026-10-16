"""诊断：敌人为什么没被标注。

列出所有对象槽的 id/x/y，并与画面中实际可见的精灵位置对照。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402
from env import ROM  # noqa: E402
from visualize import boot  # noqa: E402


def dump(nes, label):
    print(f"\n=== {label} ===")
    print("slot  id    x_raw      y_raw     x_px      y_px    spd  grav")
    for s in range(R.N_OBJ_SLOTS):
        if s == R.OBJ.CAMERA_SLOT:   # slot 15 = scroll object, no sprite
            continue
        oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
        xr = R.read_obj_world_x(nes, s)
        yr = R.read_obj_world_y(nes, s)
        spd = nes[R.OBJ.spd_x + R.OBJ.STRIDE * s]
        spd = spd - 256 if spd > 127 else spd
        grav = nes[R.OBJ.gravity + R.OBJ.STRIDE * s]
        x, y = R.px(xr), R.px(yr)
        flag = ""
        if oid == 0:
            flag = "  <empty>"
        elif 0 <= x < 256 and 0 <= y < 240:
            flag = "  <ON-SCREEN>"
        else:
            flag = "  <off-screen>"
        print(f"{s:4d} {oid:4d} {xr:9d} {yr:9d} {x:8.1f} {y:8.1f} "
              f"{spd:5d} {grav:5d}{flag}")


def main() -> None:
    nes = NES(ROM)
    boot(nes)
    dump(nes, "boot (0 frames)")

    # 跑一段，让敌人出现
    nes.controller = NES_INPUT_RIGHT | NES_INPUT_A
    for i in range(400):
        nes.step()
    nes.controller = 0
    nes.step()
    dump(nes, "after 400 frames of running right")

    buf = np.asarray(nes.step(), dtype=np.uint8)
    del nes

    # 找出画面里显眼的非背景色块（粗略定位可见精灵）
    bg = np.median(buf.reshape(-1, 3), axis=0)
    diff = np.abs(buf.astype(int) - bg).sum(axis=2)
    mask = diff > 120
    print("\n画面中显著色块的列范围（疑似精灵位置）:")
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    print(f"  x: {cols.min()}..{cols.max()}   y: {rows.min()}..{rows.max()}")
    # 按行带统计（敌人多在底部平台 y≈130-190）
    for y0 in range(100, 240, 20):
        band = mask[y0:y0 + 20]
        cs = np.where(band.any(axis=0))[0]
        if len(cs):
            groups = []
            start = cs[0]
            prev = cs[0]
            for c in cs[1:]:
                if c - prev > 6:
                    groups.append((start, prev))
                    start = c
                prev = c
            groups.append((start, prev))
            print(f"  y {y0:3d}-{y0 + 20:3d}: " +
                  ", ".join(f"{a}-{b}" for a, b in groups if b - a >= 3))


if __name__ == "__main__":
    main()
