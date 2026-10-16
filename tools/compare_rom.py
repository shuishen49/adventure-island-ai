"""对比测试：原版 ROM vs 中文 hack 版，角色能否正常前进。

关键验证：按住右跑，世界 X 是否持续增长（hack 版硬卡在 97.0）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402

ROMS = {
    "hack(中文)": r"D:\source_code\cynes\冒险岛1.nes",
    "orig(USA)": r"D:\source_code\adventure-island-ai\roms\Adventure Island (USA).nes",
}


def boot(nes, warmup=300):
    nes.step(frames=warmup)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def run(rom, label, strategy, nframes=900):
    nes = NES(rom)
    boot(nes)
    x0 = R.px(R.read_obj_world_x(nes, 0))
    trace = []
    max_x = x0
    for i in range(nframes):
        nes.controller = strategy(i)
        nes.step()
        if i % 100 == 99:
            x = R.px(R.read_obj_world_x(nes, 0))
            y = R.px(R.read_obj_world_y(nes, 0))
            max_x = max(max_x, x)
            trace.append((i + 1, x, y))
    x_end = R.px(R.read_obj_world_x(nes, 0))
    lives = nes[R.LIVES]
    del nes
    print(f"\n--- {label} / {strategy.__name__} ---")
    print(f"  start x={x0:.1f}  end x={x_end:.1f}  max x={max_x:.1f}  "
          f"advance={max_x - x0:.1f} px  lives={lives}")
    print("  轨迹 (frame, x, y):")
    for i, x, y in trace:
        print(f"    f{i:4d}  x={x:7.1f}  y={y:6.1f}")
    return max_x - x0


def hold_right(i):
    return NES_INPUT_RIGHT


def right_and_jump(i):
    return NES_INPUT_RIGHT | (NES_INPUT_A if (i // 20) % 2 == 0 else 0)


def run_then_jump(i):
    """跑 40 帧后一直跳（越过 x=97 障碍的标准打法）。"""
    return NES_INPUT_RIGHT | (NES_INPUT_A if i >= 40 else 0)


if __name__ == "__main__":
    for label, rom in ROMS.items():
        try:
            run(rom, label, hold_right)
        except Exception as e:
            print(f"\n--- {label} / hold_right --- FAILED: {e}")
        try:
            run(rom, label, run_then_jump)
        except Exception as e:
            print(f"\n--- {label} / run_then_jump --- FAILED: {e}")
