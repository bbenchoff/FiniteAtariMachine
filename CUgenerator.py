#!/usr/bin/env python3
"""
CUDA Atari ROM Generator using CuPy
Much more reliable than raw CUDA on Windows
"""

import cupy as cp
import numpy as np
import time
import os
from pathlib import Path

# Constants
ROM_SIZE = 4094
BATCH_SIZE = 1024 * 256  # 256K ROMs per batch
OPCODE_THRESHOLD = 0.66
TIA_THRESHOLD = 51
RIOT_THRESHOLD = 1
BRANCH_THRESHOLD = 177
JUMP_THRESHOLD = 37
INSTRUCTION_VARIETY = 125
MIN_SCORE = 0.80

# Valid opcodes
VALID_OPCODES = np.array([
    0x00, 0x01, 0x05, 0x06, 0x08, 0x09, 0x0A, 0x0D, 0x0E, 0x10, 0x11, 0x15, 0x16, 0x18,
    0x19, 0x1D, 0x1E, 0x20, 0x21, 0x24, 0x25, 0x26, 0x28, 0x29, 0x2A, 0x2C, 0x2D, 0x2E,
    0x30, 0x31, 0x35, 0x36, 0x38, 0x39, 0x3D, 0x3E, 0x40, 0x41, 0x45, 0x46, 0x48, 0x49,
    0x4A, 0x4C, 0x4D, 0x4E, 0x50, 0x51, 0x55, 0x56, 0x58, 0x59, 0x5D, 0x5E, 0x60, 0x61,
    0x65, 0x66, 0x68, 0x69, 0x6A, 0x6C, 0x6D, 0x6E, 0x70, 0x71, 0x75, 0x76, 0x78, 0x79,
    0x7D, 0x7E, 0x81, 0x84, 0x85, 0x86, 0x88, 0x8A, 0x8C, 0x8D, 0x8E, 0x90, 0x91, 0x94,
    0x95, 0x96, 0x98, 0x99, 0x9A, 0x9D, 0xA0, 0xA1, 0xA2, 0xA4, 0xA5, 0xA6, 0xA8, 0xA9,
    0xAA, 0xAC, 0xAD, 0xAE, 0xB0, 0xB1, 0xB4, 0xB5, 0xB6, 0xB8, 0xB9, 0xBA, 0xBC, 0xBD,
    0xBE, 0xC0, 0xC1, 0xC4, 0xC5, 0xC6, 0xC8, 0xC9, 0xCA, 0xCC, 0xCD, 0xCE, 0xD0, 0xD1,
    0xD5, 0xD6, 0xD8, 0xD9, 0xDD, 0xDE, 0xE0, 0xE1, 0xE4, 0xE5, 0xE6, 0xE8, 0xE9, 0xEA,
    0xEC, 0xED, 0xEE, 0xF0, 0xF1, 0xF5, 0xF6, 0xF8, 0xF9, 0xFD, 0xFE
], dtype=np.uint8)

BRANCH_OPCODES = np.array([0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0], dtype=np.uint8)
JUMP_OPCODES = np.array([0x4C, 0x6C, 0x20], dtype=np.uint8)
RIOT_REGISTERS = np.array([0x280, 0x281, 0x282, 0x283, 0x284, 0x285, 0x286, 0x287, 
                          0x294, 0x295, 0x296, 0x297], dtype=np.uint16)

def create_lookup_tables():
    """Create GPU lookup tables for fast analysis"""
    # Valid opcodes lookup
    valid_lut = cp.zeros(256, dtype=cp.bool_)
    valid_lut[VALID_OPCODES] = True
    
    # TIA registers (0x00-0x2F)
    tia_lut = cp.zeros(256, dtype=cp.bool_)
    tia_lut[0:0x30] = True
    
    # Branch opcodes lookup
    branch_lut = cp.zeros(256, dtype=cp.bool_)
    branch_lut[BRANCH_OPCODES] = True
    
    # Jump opcodes lookup
    jump_lut = cp.zeros(256, dtype=cp.bool_)
    jump_lut[JUMP_OPCODES] = True
    
    return valid_lut, tia_lut, branch_lut, jump_lut

def analyze_roms_gpu(roms, valid_lut, tia_lut, branch_lut, jump_lut):
    """Analyze a batch of ROMs on GPU using CuPy"""
    batch_size = roms.shape[0]
    
    # Count valid opcodes per ROM
    valid_opcodes = cp.sum(valid_lut[roms], axis=1)
    opcode_ratio = valid_opcodes / ROM_SIZE
    
    # Count TIA accesses (STA instructions to TIA range)
    # Look for STA absolute (0x8D) followed by TIA addresses
    sta_abs_mask = (roms[:, :-2] == 0x8D)
    sta_zp_mask = (roms[:, :-1] == 0x85)
    
    # For STA absolute, check if low byte is in TIA range
    tia_abs_accesses = cp.sum(sta_abs_mask & tia_lut[roms[:, 1:-1]], axis=1)
    
    # For STA zero page, check if address is in TIA range  
    tia_zp_accesses = cp.sum(sta_zp_mask & tia_lut[roms[:, 1:]], axis=1)
    
    tia_accesses = tia_abs_accesses + tia_zp_accesses
    
    # Count RIOT accesses (simplified - just look for common patterns)
    # This is a simplified version - real RIOT detection would be more complex
    riot_accesses = cp.zeros(batch_size, dtype=cp.int32)
    for riot_addr in RIOT_REGISTERS:
        low_byte = riot_addr & 0xFF
        high_byte = (riot_addr >> 8) & 0xFF
        # Look for STA absolute to this address
        addr_matches = sta_abs_mask & (roms[:, 1:-1] == low_byte) & (roms[:, 2:] == high_byte)
        riot_accesses += cp.sum(addr_matches, axis=1)
    
    # Count branches and jumps
    branch_count = cp.sum(branch_lut[roms], axis=1)
    jump_count = cp.sum(jump_lut[roms], axis=1)
    
    # Count unique opcodes per ROM
    unique_opcodes = cp.zeros(batch_size, dtype=cp.int32)
    for i in range(batch_size):
        unique_opcodes[i] = len(cp.unique(roms[i][valid_lut[roms[i]]]))
    
    # Calculate composite score
    scores = (opcode_ratio * 0.25 + 
              cp.minimum(tia_accesses / 20.0, 1.0) * 0.20 +
              cp.minimum(riot_accesses / 10.0, 1.0) * 0.15 +
              cp.minimum(branch_count / 15.0, 1.0) * 0.15 +
              cp.minimum(jump_count / 8.0, 1.0) * 0.10 +
              cp.minimum(unique_opcodes / 30.0, 1.0) * 0.10)
    
    # Check if promising
    promising = ((opcode_ratio >= OPCODE_THRESHOLD) &
                (tia_accesses >= TIA_THRESHOLD) &
                (riot_accesses >= RIOT_THRESHOLD) &
                (branch_count >= BRANCH_THRESHOLD) &
                (jump_count >= JUMP_THRESHOLD) &
                (unique_opcodes >= INSTRUCTION_VARIETY) &
                (scores >= MIN_SCORE))
    
    return {
        'scores': scores,
        'opcode_ratio': opcode_ratio,
        'tia_accesses': tia_accesses,
        'riot_accesses': riot_accesses,
        'branch_count': branch_count,
        'jump_count': jump_count,
        'unique_opcodes': unique_opcodes,
        'promising': promising
    }

def save_promising_rom(rom_data, analysis, rom_id, output_dir):
    """Save a promising ROM to disk"""
    score = float(analysis['scores'])
    filename = f"base_{rom_id:06d}_score_{score:.3f}_cupy.bin"
    filepath = output_dir / filename
    
    # Save ROM
    with open(filepath, 'wb') as f:
        f.write(rom_data.tobytes())
    
    # Save metadata
    meta_filename = f"base_{rom_id:06d}_score_{score:.3f}_cupy.txt"
    meta_filepath = output_dir / meta_filename
    
    with open(meta_filepath, 'w') as f:
        f.write("ROM Base Analysis\n")
        f.write("=================\n")
        f.write("Generated with CuPy\n")
        f.write(f"Overall Score: {score:.4f}\n")
        f.write(f"Valid Opcodes: {float(analysis['opcode_ratio']):.3f} ({float(analysis['opcode_ratio'])*100:.1f}%)\n")
        f.write(f"TIA Accesses: {int(analysis['tia_accesses'])}\n")
        f.write(f"RIOT Accesses: {int(analysis['riot_accesses'])}\n")
        f.write(f"Branch Instructions: {int(analysis['branch_count'])}\n")
        f.write(f"Jump Instructions: {int(analysis['jump_count'])}\n")
        f.write(f"Unique Opcodes: {int(analysis['unique_opcodes'])}\n")
    
    print(f"💎 SAVED: {filename}")
    print(f"    Score: {score:.4f} | Opcodes: {float(analysis['opcode_ratio']):.3f} | "
          f"TIA: {int(analysis['tia_accesses'])} | RIOT: {int(analysis['riot_accesses'])}")

def main():
    print("🕹️  Finite Atari Machine - CuPy Edition")
    print("=" * 80)
    print(f"Generating {BATCH_SIZE:,} ROMs per batch on GPU")
    print(f"ROM size: {ROM_SIZE} bytes")
    print("🟢 EASY Thresholds (5th percentile - bottom 5% of real games)")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    # Check GPU
    print(f"🎮 GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"🎮 Memory: {cp.cuda.runtime.memGetInfo()[1] // 1024**2} MB")
    
    # Create output directory
    output_dir = Path("promising_roms")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize lookup tables on GPU
    print("\n🔨 Initializing lookup tables...")
    valid_lut, tia_lut, branch_lut, jump_lut = create_lookup_tables()
    
    total_generated = 0
    promising_found = 0
    start_time = time.time()
    last_report = start_time
    
    try:
        while True:
            # Generate random ROMs on GPU
            batch_start = time.time()
            roms = cp.random.randint(0, 256, size=(BATCH_SIZE, ROM_SIZE), dtype=cp.uint8)
            
            # Analyze on GPU
            analysis = analyze_roms_gpu(roms, valid_lut, tia_lut, branch_lut, jump_lut)
            
            # Find promising ROMs
            promising_indices = cp.where(analysis['promising'])[0]
            
            if len(promising_indices) > 0:
                # Copy promising ROMs back to CPU for saving
                promising_roms = cp.asnumpy(roms[promising_indices])
                promising_analysis = {k: cp.asnumpy(v[promising_indices]) for k, v in analysis.items()}
                
                # Save each promising ROM
                for i in range(len(promising_indices)):
                    rom_analysis = {k: v[i] for k, v in promising_analysis.items()}
                    save_promising_rom(promising_roms[i], rom_analysis, promising_found, output_dir)
                    promising_found += 1
            
            total_generated += BATCH_SIZE
            batch_time = time.time() - batch_start
            
            # Report progress every 5 seconds
            current_time = time.time()
            if current_time - last_report >= 5:
                elapsed = current_time - start_time
                rate = total_generated / elapsed
                success_rate = promising_found / total_generated * 100
                
                print(f"\r🔍 Generated: {total_generated:,} | Promising: {promising_found} | "
                      f"Success: {success_rate:.6f}% | Rate: {rate:,.0f}/sec | "
                      f"Batch: {batch_time:.1f}s", end="", flush=True)
                
                last_report = current_time
    
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n🛑 Stopped after {elapsed:.1f} seconds")
        print(f"📊 Final stats: {total_generated:,} generated, {promising_found} promising found")
        print("💾 Check 'promising_roms/' directory for results")

if __name__ == "__main__":
    main()