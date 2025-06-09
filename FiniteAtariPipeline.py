# ==========================================================================
#
#   Finite Atari unified pipeline (4 KB, reset @ $F000) - FIXED VERSION
#
#   This script generates random Atari 2600 ROMs in the GPU via CUDA,
#   Filters them on-GPU with a heuristic, and if interesting, boots
#   them in head-less MAME for 2 s to check if they have dynamic video.
#
#   It is designed to run on a CUDA-capable GPU with MAME installed in PATH.
#
#   The output is saved in the "finite_atari_roms" directory.
#
# ===========================================================================

from __future__ import annotations
import cupy as cp, numpy as np, hashlib, subprocess, tempfile, time, textwrap
from pathlib import Path
from PIL import Image

# ─── 1.  Global constants ────────────────────────────────────────────────────
ROM_SIZE       = 4096
PAYLOAD_BYTES  = ROM_SIZE - 2
RESET_VECTOR   = (0x00, 0xF0)         # little-endian $F000
BATCH_SIZE     = 1024 * 256           # ~256 k ROMs per GPU batch
STATUS_EVERY   = 10                   # batches between status prints
OUTPUT_DIR     = Path("finite_atari_roms"); OUTPUT_DIR.mkdir(exist_ok=True)

# Video thresholds
BLACK_LEVEL          = 15             # 0-255 grey; ≤ this is "black"
NONBLACK_THRESHOLD   = 0.005          # ≥ 0.5 % pixels brighter ⇒ video
DYNAMIC_THRESHOLD    = 0.01           # ≥ 1 % hashed pixels differ ⇒ motion

# Heuristic thresholds (same as your earlier Stella/MAME scripts)
OPCODE_THRESHOLD    = 0.58
TIA_THRESHOLD       = 50
RIOT_THRESHOLD      = 13
BRANCH_THRESHOLD    = 150
JUMP_THRESHOLD      = 37
INSTRUCTION_VARIETY = 100
MIN_SCORE           = 0.52

# ─── 2.  Opcode lookup tables ────────────────────────────────────────────────
# Valid 6502 opcodes for 2600 home-brew context
VALID_OPCODES = np.array([
    0x00,0x01,0x05,0x06,0x08,0x09,0x0A,0x0D,0x0E,0x10,0x11,0x15,0x16,0x18,
    0x19,0x1D,0x1E,0x20,0x21,0x24,0x25,0x26,0x28,0x29,0x2A,0x2C,0x2D,0x2E,
    0x30,0x31,0x35,0x36,0x38,0x39,0x3D,0x3E,0x40,0x41,0x45,0x46,0x48,0x49,
    0x4A,0x4C,0x4D,0x4E,0x50,0x51,0x55,0x56,0x58,0x59,0x5D,0x5E,0x60,0x61,
    0x65,0x66,0x68,0x69,0x6A,0x6C,0x6D,0x6E,0x70,0x71,0x75,0x76,0x78,0x79,
    0x7D,0x7E,0x81,0x84,0x85,0x86,0x88,0x8A,0x8C,0x8D,0x8E,0x90,0x91,0x94,
    0x95,0x96,0x98,0x99,0x9A,0x9D,0xA0,0xA1,0xA2,0xA4,0xA5,0xA6,0xA8,0xA9,
    0xAA,0xAC,0xAD,0xAE,0xB0,0xB1,0xB4,0xB5,0xB6,0xB8,0xB9,0xBA,0xBC,0xBD,
    0xBE,0xC0,0xC1,0xC4,0xC5,0xC6,0xC8,0xC9,0xCA,0xCC,0xCD,0xCE,0xD0,0xD1,
    0xD5,0xD6,0xD8,0xD9,0xDD,0xDE,0xE0,0xE1,0xE4,0xE5,0xE6,0xE8,0xE9,0xEA,
    0xEC,0xED,0xEE,0xF0,0xF1,0xF5,0xF6,0xF8,0xF9,0xFD,0xFE], dtype=np.uint8)

BRANCH_OPCODES = np.array([0x10,0x30,0x50,0x70,0x90,0xB0,0xD0,0xF0], dtype=np.uint8)
JUMP_OPCODES   = np.array([0x4C,0x6C,0x20], dtype=np.uint8)

def create_luts():
    """Return a dict of 256-entry boolean lookup tables (cupy)."""
    lut = {}
    lut["valid"]  = cp.zeros(256, cp.bool_); lut["valid"][VALID_OPCODES]  = True
    lut["branch"] = cp.zeros(256, cp.bool_); lut["branch"][BRANCH_OPCODES] = True
    lut["jump"]   = cp.zeros(256, cp.bool_); lut["jump"][JUMP_OPCODES]     = True

    # 2600 addressing quirks for TIA/RIOT access detection
    lut["tia_store"] = cp.zeros(256, cp.bool_)
    lut["tia_store"][[0x84,0x85,0x86, 0x94,0x95,0x96]] = True   # STY/STA/STX (zp & zp,x)
    lut["tia_load"]  = cp.zeros(256, cp.bool_)
    lut["tia_load" ][[0xA4,0xA5,0xA6, 0xB4,0xB5,0xB6]] = True   # LDY/LDA/LDX (zp & zp,x)
    lut["tia_abs"]   = cp.zeros(256, cp.bool_)
    lut["tia_abs"  ][[0x8C,0x8D,0x8E, 0xAC,0xAD,0xAE]] = True   # abs versions

    lut["riot_acc"]  = cp.zeros(256, cp.bool_)
    lut["riot_acc"][[0x84,0x85,0x86, 0xA4,0xA5,0xA6]] = True

    addr = cp.arange(256, dtype=cp.uint8)
    lut["tia_range"] = addr <= 0x2F
    lut["riot_tmr"]  = (addr >= 0x80) & (addr <= 0x87)
    lut["riot_io"]   = (addr >= 0x94) & (addr <= 0x97)
    return lut

# ─── 3.  GPU heuristic filter ────────────────────────────────────────────────
def analyse_batch(roms: cp.ndarray, lut) -> tuple[np.ndarray, cp.ndarray]:
    """
    Return (interesting_mask, scores) for a 2-D uint8 array of ROMs.
    Each row = one ROM.
    """
    valid_cnt    = cp.sum(lut["valid"][roms], axis=1)
    opcode_ratio = valid_cnt.astype(cp.float32) / ROM_SIZE
    branch_cnt   = cp.sum(lut["branch"][roms], axis=1)
    jump_cnt     = cp.sum(lut["jump"  ][roms], axis=1)

    # --- TIA accesses --------------------------------------------------------
    tia_acc  = cp.sum((lut["tia_store"][roms[:,:-1]] | lut["tia_load"][roms[:,:-1]])
                      & lut["tia_range"][roms[:,1:]], axis=1)
    tia_acc += cp.sum(lut["tia_abs"][roms[:,:-2]]
                      & lut["tia_range"][roms[:,1:-1]]
                      & (roms[:,2:] == 0x00), axis=1)

    # --- RIOT accesses -------------------------------------------------------
    riot_acc  = cp.sum(lut["riot_acc"][roms[:,:-1]] & lut["riot_tmr"][roms[:,1:]], axis=1)
    riot_acc += cp.sum(lut["riot_acc"][roms[:,:-1]] & lut["riot_io" ][roms[:,1:]], axis=1)

    # --- Opcode diversity in first 1 KB --------------------------------------
    uniq = cp.zeros(roms.shape[0], dtype=cp.int32)
    first_kb = roms[:, :1024]
    for op in VALID_OPCODES:
        uniq += cp.any(first_kb == op, axis=1)

    scores = (opcode_ratio * 0.25 +
              cp.minimum(tia_acc / 150.0, 1.0) * 0.30 +
              cp.minimum(riot_acc / 50.0, 1.0)  * 0.20 +
              cp.minimum(branch_cnt / 200.0, 1.0) * 0.15 +
              cp.minimum(jump_cnt / 40.0, 1.0)   * 0.10)

    interesting = ((opcode_ratio >= OPCODE_THRESHOLD) &
                   (tia_acc      >= TIA_THRESHOLD) &
                   (riot_acc     >= RIOT_THRESHOLD) &
                   (branch_cnt   >= BRANCH_THRESHOLD) &
                   (jump_cnt     >= JUMP_THRESHOLD) &
                   (uniq         >= INSTRUCTION_VARIETY) &
                   (scores       >= MIN_SCORE))

    return interesting, scores

# ─── 4.  Lua helper script (snapshot two frames) - FIXED ────────────────────
SNAPSHOT_LUA = textwrap.dedent("""
    local s = manager.machine.screens[":screen"]
    local frame_count = 0
    emu.register_frame_done(function ()
        frame_count = frame_count + 1
        if     frame_count == 1  then s:snapshot("first.png")
        elseif frame_count == 60 then s:snapshot("second.png"); manager.machine:exit() end
    end, "snapper")
""")

# ─── 5.  Video analysis helpers ──────────────────────────────────────────────
def _hash16(img: Path) -> str:
    with Image.open(img) as im:
        im = im.convert("L").resize((16,16), Image.NEAREST)
        return hashlib.sha1(im.tobytes()).hexdigest()

def _frame_is_nonblack(img: Path) -> bool:
    with Image.open(img) as im:
        g = np.asarray(im.convert("L"))
    return (g > BLACK_LEVEL).mean() >= NONBLACK_THRESHOLD

def rom_video_flags(rom: bytes, *, mame="mame", seconds=2.0) -> tuple[bool,bool]:
    """
    Returns (has_video, is_dynamic).
    • has_video  → first frame not black
    • is_dynamic → ≥ 1 % hashed pixels differ between frame 1 and 60
    """
    with tempfile.TemporaryDirectory() as td_s:
        td = Path(td_s)
        (td / "test.bin").write_bytes(rom)
        (td / "snapshot.lua").write_text(SNAPSHOT_LUA)

        base = [mame, "a2600", "-cart", "test.bin",
                "-seconds_to_run", str(seconds),
                "-nothrottle", "-window", "-sound", "none", "-skip_gameinfo"]

        for flag in ("-autoboot_script", "-script"):
            try:
                subprocess.run(base + [flag, "snapshot.lua"],
                               cwd=td, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=seconds*5,
                               check=True)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if flag == "-autoboot_script":
                    continue
                return (False, False)

        # Check both root directory and snap subdirectory for frames
        f1, f2 = td / "first.png", td / "second.png"
        snap_dir = td / "snap"
        if not f1.exists() and snap_dir.exists():
            snap_f1 = snap_dir / "first.png"
            snap_f2 = snap_dir / "second.png"
            if snap_f1.exists():
                f1 = snap_f1
            if snap_f2.exists():
                f2 = snap_f2

        if not f1.exists():
            return (False, False)

        nonblack = _frame_is_nonblack(f1)
        if not nonblack or not f2.exists():
            return (nonblack, False)

        diff_bits = bin(int(_hash16(f1),16) ^ int(_hash16(f2),16)).count("1")
        dynamic = diff_bits / 256.0 >= DYNAMIC_THRESHOLD
        return (nonblack, dynamic)

# ─── 6.  ROM generator ───────────────────────────────────────────────────────
def generate_batch(n: int) -> np.ndarray:
    payload = np.random.randint(0, 256, size=(n, PAYLOAD_BYTES), dtype=np.uint8)
    reset   = np.tile(np.array(RESET_VECTOR, dtype=np.uint8), (n,1))
    return np.hstack((payload, reset))

# ─── 7.  Main loop ───────────────────────────────────────────────────────────
def main():
    lut = create_luts()
    tot_gen = tot_int = tot_vid = tot_dyn = 0
    batch_idx = 0
    start = time.perf_counter()

    try:
        while True:
            roms_cpu = generate_batch(BATCH_SIZE); tot_gen += BATCH_SIZE

            roms_gpu = cp.asarray(roms_cpu)
            keep, _ = analyse_batch(roms_gpu, lut)
            keep = keep.get(); del roms_gpu
            interesting = roms_cpu[keep]
            tot_int += len(interesting)

            for rom in interesting:
                has_vid, is_dyn = rom_video_flags(rom.tobytes())

                # Save if EITHER condition is true
                if has_vid or is_dyn:
                    sha = hashlib.sha1(rom).hexdigest()[:12]
                    (OUTPUT_DIR / f"{sha}.bin").write_bytes(rom.tobytes())

                # Separate bookkeeping
                if has_vid:
                    tot_vid += 1           # first frame not black
                if is_dyn:
                    tot_dyn += 1           # animation detected

            batch_idx += 1
            if batch_idx % STATUS_EVERY == 0:
                elapsed = time.perf_counter() - start
                rate = int(tot_gen / elapsed) if elapsed else 0
                print(f"{tot_gen:,d} generated | "
                      f"{tot_int:,d} interesting | {tot_vid:,d} with video | "
                      f"{tot_dyn:,d} dynamic | {rate:,d} ROM/s", flush=True)

    except KeyboardInterrupt:
        pass  # graceful exit

    elapsed = time.perf_counter() - start
    rate = int(tot_gen / elapsed) if elapsed else 0
    print("─"*72)
    print(f"TOTAL: {tot_gen:,d} generated | {tot_int:,d} interesting | "
          f"{tot_vid:,d} with video | {tot_dyn:,d} dynamic | {rate:,d} ROM/s")

if __name__ == "__main__":
    main()