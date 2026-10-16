"""Verify death detection: force stamina to 0 via RAM write and watch
lives / death flag, to define episode-termination logic for the RL env."""

from cynes import NES, NES_INPUT_RIGHT, NES_INPUT_START

ROM = r"D:\source_code\cynes\冒险岛1.nes"


def boot(nes):
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=180)


def st(nes, tag):
    print(f"[{tag:>10}] death={nes[0x72]} lives={nes[0x3F]} sta={nes[0x76]} "
          f"death_anim={nes[0x71]} complete={nes[0x7A]}")


nes = NES(ROM)
boot(nes)
st(nes, "boot")

print("\n--- force stamina=0 (starvation death) ---")
nes[0x76] = 0
for i in range(300):
    nes.step()
    if i % 50 == 0:
        st(nes, f"f{i}")
    if nes[0x72] or nes[0x3F] < 3:
        st(nes, f"DEATH at f{i}")
        break

print("\n--- wait for respawn ---")
for i in range(300):
    nes.step()
    if i % 100 == 0:
        st(nes, f"respawn f{i}")
st(nes, "final")
del nes
