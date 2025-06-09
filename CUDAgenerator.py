#!/usr/bin/env python3
"""
CUDA Atari ROM Generator - FIXED VERSION
"""

import cupy as cp
import numpy as np
import time
from pathlib import Path

# Constants
ROM_SIZE = 4094
BATCH_SIZE = 1024 * 256

# Discovery thresholds based on observed patterns
OPCODE_THRESHOLD = 0.58
TIA_THRESHOLD = 50
RIOT_THRESHOLD = 13
BRANCH_THRESHOLD = 150
JUMP_THRESHOLD = 37
INSTRUCTION_VARIETY = 100
MIN_SCORE = 0.52

# Valid 6502 opcodes (151 total)
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

# Control flow opcodes
BRANCH_OPCODES = np.array([0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0], dtype=np.uint8)
JUMP_OPCODES = np.array([0x4C, 0x6C, 0x20], dtype=np.uint8)

def create_lookup_tables():
    """Create GPU lookup tables for ROM analysis"""
    valid_lut = cp.zeros(256, dtype=cp.bool_)
    valid_lut[VALID_OPCODES] = True
    
    branch_lut = cp.zeros(256, dtype=cp.bool_)
    branch_lut[BRANCH_OPCODES] = True
    
    jump_lut = cp.zeros(256, dtype=cp.bool_)
    jump_lut[JUMP_OPCODES] = True
    
    # TIA instruction lookups
    tia_store_lut = cp.zeros(256, dtype=cp.bool_)
    tia_store_lut[[0x85, 0x86, 0x84, 0x95, 0x96, 0x94]] = True
    
    tia_load_lut = cp.zeros(256, dtype=cp.bool_)
    tia_load_lut[[0xA5, 0xA6, 0xA4, 0xB5, 0xB6, 0xB4]] = True
    
    tia_abs_lut = cp.zeros(256, dtype=cp.bool_)
    tia_abs_lut[[0x8D, 0x8E, 0x8C, 0xAD, 0xAE, 0xAC]] = True
    
    # RIOT instruction lookups
    riot_access_lut = cp.zeros(256, dtype=cp.bool_)
    riot_access_lut[[0x85, 0x86, 0x84, 0xA5, 0xA6, 0xA4]] = True
    
    # Address range masks
    tia_range_mask = cp.arange(256, dtype=cp.uint8) <= 0x2F
    riot_timer_mask = (cp.arange(256, dtype=cp.uint8) >= 0x80) & (cp.arange(256, dtype=cp.uint8) <= 0x87)
    riot_io_mask = (cp.arange(256, dtype=cp.uint8) >= 0x94) & (cp.arange(256, dtype=cp.uint8) <= 0x97)
    
    return {
        'valid': valid_lut,
        'branch': branch_lut,
        'jump': jump_lut,
        'tia_store': tia_store_lut,
        'tia_load': tia_load_lut,
        'tia_abs': tia_abs_lut,
        'riot_access': riot_access_lut,
        'tia_range': tia_range_mask,
        'riot_timer': riot_timer_mask,
        'riot_io': riot_io_mask
    }

def analyze_roms(roms, lut):
    """Analyze ROMs for game-like patterns"""
    batch_size = roms.shape[0]
    
    # Opcode analysis
    valid_opcodes_count = cp.sum(lut['valid'][roms], axis=1)
    opcode_ratio = valid_opcodes_count.astype(cp.float32) / ROM_SIZE
    
    # Control flow analysis
    branch_count = cp.sum(lut['branch'][roms], axis=1)
    jump_count = cp.sum(lut['jump'][roms], axis=1)
    
    # TIA analysis
    tia_accesses = cp.zeros(batch_size, dtype=cp.int32)
    
    # Zero page addressing
    tia_store_zp = lut['tia_store'][roms[:, :-1]] & lut['tia_range'][roms[:, 1:]]
    tia_load_zp = lut['tia_load'][roms[:, :-1]] & lut['tia_range'][roms[:, 1:]]
    tia_zp_total = cp.sum(tia_store_zp | tia_load_zp, axis=1)
    tia_accesses += tia_zp_total
    
    # Absolute addressing (any high byte due to mirroring)
    tia_abs_positions = lut['tia_abs'][roms[:, :-2]]
    tia_abs_targets = lut['tia_range'][roms[:, 1:-1]]  # Only check low byte for TIA range
    tia_abs_total = cp.sum(tia_abs_positions & tia_abs_targets, axis=1)
    tia_accesses += tia_abs_total
    
    # RIOT analysis
    riot_accesses = cp.zeros(batch_size, dtype=cp.int32)
    
    # Timer access
    riot_timer_positions = lut['riot_access'][roms[:, :-1]]
    riot_timer_targets = lut['riot_timer'][roms[:, 1:]]
    riot_timer_hits = cp.sum(riot_timer_positions & riot_timer_targets, axis=1)
    riot_accesses += riot_timer_hits
    
    # I/O access
    riot_io_positions = lut['riot_access'][roms[:, :-1]]
    riot_io_targets = lut['riot_io'][roms[:, 1:]]
    riot_io_hits = cp.sum(riot_io_positions & riot_io_targets, axis=1)
    riot_accesses += riot_io_hits
    
    # FIXED: Unique opcode counting in first 1KB (code section)
    unique_opcodes = cp.zeros(batch_size, dtype=cp.int32)
    first_kb = roms[:, :1024]  # First 1KB where code typically resides
    
    # Count unique valid opcodes in the code section (FIXED - no duplication)
    for opcode in VALID_OPCODES:
        has_opcode = cp.any(first_kb == opcode, axis=1)
        unique_opcodes += has_opcode.astype(cp.int32)
    
    # Composite score
    scores = (
        opcode_ratio * 0.25 + 
        cp.minimum(tia_accesses / 150.0, 1.0) * 0.30 +
        cp.minimum(riot_accesses / 50.0, 1.0) * 0.20 +
        cp.minimum(branch_count / 200.0, 1.0) * 0.15 +
        cp.minimum(jump_count / 40.0, 1.0) * 0.10
    )
    
    # Promising ROM detection
    promising = (
        (opcode_ratio >= OPCODE_THRESHOLD) &
        (tia_accesses >= TIA_THRESHOLD) &
        (riot_accesses >= RIOT_THRESHOLD) &
        (branch_count >= BRANCH_THRESHOLD) &
        (jump_count >= JUMP_THRESHOLD) &
        (unique_opcodes >= INSTRUCTION_VARIETY) &
        (scores >= MIN_SCORE)
    )
    
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

def save_promising_rom(rom_data, score, rom_id, output_dir):
    """Save promising ROM with simple filename format: number_score_timestamp.bin"""
    timestamp = int(time.time())
    filename = f"{rom_id:06d}_{score:.3f}_{timestamp}.bin"
    filepath = output_dir / filename
    
    with open(filepath, 'wb') as f:
        f.write(rom_data.tobytes())
    
    return filename

def main():
    print("Finite Atari Machine - CUDA Generator")
    print("=" * 60)
    print(f"Batch size: {BATCH_SIZE:,} ROMs per batch")
    print(f"ROM size: {ROM_SIZE:,} bytes")
    print()
    print("Thresholds:")
    print(f"  Opcodes: {OPCODE_THRESHOLD:.1%}")
    print(f"  TIA: {TIA_THRESHOLD}+")
    print(f"  RIOT: {RIOT_THRESHOLD}+")
    print(f"  Branches: {BRANCH_THRESHOLD}+")
    print(f"  Jumps: {JUMP_THRESHOLD}+")
    print(f"  Unique opcodes: {INSTRUCTION_VARIETY}+")
    print(f"  Min score: {MIN_SCORE:.2f}")
    print()
    
    # GPU info
    try:
        gpu_props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = gpu_props['name'].decode()
        total_mem = cp.cuda.runtime.memGetInfo()[1] // 1024**2
        print(f"GPU: {gpu_name}")
        print(f"Memory: {total_mem:,} MB")
    except Exception:
        print("GPU: CuPy device detected")
    
    print("\nInitializing lookup tables...")
    
    # Setup
    output_dir = Path("possible_roms")
    output_dir.mkdir(exist_ok=True)
    
    lookup_tables = create_lookup_tables()
    
    # Statistics
    total_generated = 0
    promising_found = 0
    start_time = time.time()
    last_report = start_time
    best_score_ever = 0.0
    
    print("Starting ROM generation...")
    print("=" * 60)
    
    try:
        while True:
            batch_start = time.time()
            
            # Generate batch of ROMs
            roms = cp.random.randint(0, 256, size=(BATCH_SIZE, ROM_SIZE), dtype=cp.uint8)
            
            # Analyze ROMs
            analysis = analyze_roms(roms, lookup_tables)
            
            # Track best score
            current_best = float(cp.max(analysis['scores']))
            if current_best > best_score_ever:
                best_score_ever = current_best
            
            # Check for promising ROMs
            promising_indices = cp.where(analysis['promising'])[0]
            
            if len(promising_indices) > 0:
                # Save promising ROMs
                promising_roms = cp.asnumpy(roms[promising_indices])
                promising_scores = cp.asnumpy(analysis['scores'][promising_indices])
                
                for i in range(len(promising_indices)):
                    filename = save_promising_rom(
                        promising_roms[i], promising_scores[i], promising_found, output_dir
                    )
                    promising_found += 1
            
            total_generated += BATCH_SIZE
            batch_time = time.time() - batch_start
            
            # Progress reporting
            current_time = time.time()
            if current_time - last_report >= 4:
                # Get best ROM stats for this batch
                scores = cp.asnumpy(analysis['scores'])
                best_idx = np.argmax(scores)
                best_opcodes = float(analysis['opcode_ratio'][best_idx])
                best_tia = int(analysis['tia_accesses'][best_idx])
                best_riot = int(analysis['riot_accesses'][best_idx])
                best_branches = int(analysis['branch_count'][best_idx])
                best_jumps = int(analysis['jump_count'][best_idx])
                
                elapsed = current_time - start_time
                rate = total_generated / elapsed
                success_rate = promising_found / total_generated * 100 if total_generated > 0 else 0
                
                print(f"\rGenerated: {total_generated:,} | Found: {promising_found} | "
                      f"Success: {success_rate:.8f}% | Rate: {rate:,.0f}/sec | "
                      f"Best: {best_score_ever:.3f} | "
                      f"Op:{best_opcodes:.1%} TIA:{best_tia} RIOT:{best_riot} Br:{best_branches} Jmp:{best_jumps}", 
                      end="", flush=True)
                
                last_report = current_time
    
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        rate = total_generated / elapsed
        success_rate = promising_found / total_generated * 100 if total_generated > 0 else 0
        
        print(f"\n\nStopped after {elapsed:.1f} seconds")
        print(f"Total ROMs generated: {total_generated:,}")
        print(f"Promising ROMs found: {promising_found}")
        print(f"Success rate: {success_rate:.8f}%")
        print(f"Average rate: {rate:,.0f} ROMs/second")
        print(f"Best score achieved: {best_score_ever:.4f}")
        print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()