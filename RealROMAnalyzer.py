#!/usr/bin/env python3
"""
Fixed Atari ROM Collection Analyzer
Proper memory mapping analysis for TIA and RIOT
"""

import os
import struct
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import statistics
from collections import Counter, defaultdict

# 6502 instruction analysis
VALID_OPCODES = {
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
}

# TIA registers (only the low byte matters due to mirroring)
TIA_REGISTERS = {
    # Write registers
    0x00: 'VSYNC', 0x01: 'VBLANK', 0x02: 'WSYNC', 0x03: 'RSYNC',
    0x04: 'NUSIZ0', 0x05: 'NUSIZ1', 0x06: 'COLUP0', 0x07: 'COLUP1',
    0x08: 'COLUPF', 0x09: 'COLUBK', 0x0A: 'CTRLPF', 0x0B: 'REFP0',
    0x0C: 'REFP1', 0x0D: 'PF0', 0x0E: 'PF1', 0x0F: 'PF2',
    0x10: 'RESP0', 0x11: 'RESP1', 0x12: 'RESM0', 0x13: 'RESM1',
    0x14: 'RESBL', 0x15: 'AUDC0', 0x16: 'AUDC1', 0x17: 'AUDF0',
    0x18: 'AUDF1', 0x19: 'AUDV0', 0x1A: 'AUDV1', 0x1B: 'GRP0',
    0x1C: 'GRP1', 0x1D: 'ENAM0', 0x1E: 'ENAM1', 0x1F: 'ENABL',
    0x20: 'HMP0', 0x21: 'HMP1', 0x22: 'HMM0', 0x23: 'HMM1',
    0x24: 'HMBL', 0x25: 'VDELP0', 0x26: 'VDELP1', 0x27: 'VDELBL',
    0x28: 'RESMP0', 0x29: 'RESMP1', 0x2A: 'HMOVE', 0x2B: 'HMCLR',
    0x2C: 'CXCLR', 0x2D: 'CXCLR', 0x2E: 'CXCLR', 0x2F: 'CXCLR'
}

# RIOT register ranges (low byte only due to mirroring)
# Timer registers: $x80-$x87 (where x can be 0,1,2,3...)  
# I/O registers: $x94-$x97
RIOT_TIMER_RANGE = range(0x80, 0x88)      # 0x80-0x87
RIOT_IO_RANGE = range(0x94, 0x98)         # 0x94-0x97

# All instructions that can read/write memory
MEMORY_INSTRUCTIONS = {
    # Store instructions
    0x85: ('STA', 'zp', 2),      # STA zero page
    0x8D: ('STA', 'abs', 3),     # STA absolute  
    0x95: ('STA', 'zp,X', 2),    # STA zero page,X
    0x9D: ('STA', 'abs,X', 3),   # STA absolute,X
    0x99: ('STA', 'abs,Y', 3),   # STA absolute,Y
    0x81: ('STA', '(zp,X)', 2),  # STA (zero page,X)
    0x91: ('STA', '(zp),Y', 2),  # STA (zero page),Y
    
    0x86: ('STX', 'zp', 2),      # STX zero page
    0x8E: ('STX', 'abs', 3),     # STX absolute
    0x96: ('STX', 'zp,Y', 2),    # STX zero page,Y
    
    0x84: ('STY', 'zp', 2),      # STY zero page  
    0x8C: ('STY', 'abs', 3),     # STY absolute
    0x94: ('STY', 'zp,X', 2),    # STY zero page,X
    
    # Load instructions  
    0xA5: ('LDA', 'zp', 2),      # LDA zero page
    0xAD: ('LDA', 'abs', 3),     # LDA absolute
    0xB5: ('LDA', 'zp,X', 2),    # LDA zero page,X
    0xBD: ('LDA', 'abs,X', 3),   # LDA absolute,X
    0xB9: ('LDA', 'abs,Y', 3),   # LDA absolute,Y
    0xA1: ('LDA', '(zp,X)', 2),  # LDA (zero page,X)
    0xB1: ('LDA', '(zp),Y', 2),  # LDA (zero page),Y
    
    0xA6: ('LDX', 'zp', 2),      # LDX zero page
    0xAE: ('LDX', 'abs', 3),     # LDX absolute  
    0xB6: ('LDX', 'zp,Y', 2),    # LDX zero page,Y
    0xBE: ('LDX', 'abs,Y', 3),   # LDX absolute,Y
    
    0xA4: ('LDY', 'zp', 2),      # LDY zero page
    0xAC: ('LDY', 'abs', 3),     # LDY absolute
    0xB4: ('LDY', 'zp,X', 2),    # LDY zero page,X  
    0xBC: ('LDY', 'abs,X', 3),   # LDY absolute,X
}

def is_tia_address(addr: int) -> bool:
    """Check if address maps to TIA (considering mirroring)"""
    # TIA appears at $00-$2F in zero page, and mirrors throughout address space
    return (addr & 0xFF) <= 0x2F

def is_riot_address(addr: int) -> bool:
    """Check if address maps to RIOT (considering mirroring)"""
    low_byte = addr & 0xFF
    return low_byte in RIOT_TIMER_RANGE or low_byte in RIOT_IO_RANGE

def get_riot_type(addr: int) -> str:
    """Determine if RIOT access is timer or I/O"""
    low_byte = addr & 0xFF
    if low_byte in RIOT_TIMER_RANGE:
        return "TIMER"
    elif low_byte in RIOT_IO_RANGE:
        return "IO"
    return "UNKNOWN"

@dataclass
class MemoryAccess:
    """Memory access information"""
    pc: int
    instruction: str
    addressing_mode: str
    target_address: int
    access_type: str  # 'TIA', 'RIOT_TIMER', 'RIOT_IO', 'OTHER'
    is_read: bool
    context_bytes: bytes

@dataclass
class ROMAnalysis:
    filename: str
    size: int
    valid_opcodes_ratio: float
    tia_accesses: List[MemoryAccess]
    riot_accesses: List[MemoryAccess]
    tia_access_count: int
    riot_access_count: int
    riot_timer_count: int
    riot_io_count: int
    branch_instructions: int
    jump_instructions: int
    unique_opcodes: int
    score: float

def analyze_memory_accesses(rom_data: bytes) -> Tuple[List[MemoryAccess], List[MemoryAccess]]:
    """Analyze all memory accesses in ROM"""
    tia_accesses = []
    riot_accesses = []
    
    i = 0
    while i < len(rom_data):
        if rom_data[i] in MEMORY_INSTRUCTIONS:
            inst_name, addr_mode, inst_len = MEMORY_INSTRUCTIONS[rom_data[i]]
            
            if i + inst_len <= len(rom_data):
                target_addr = None
                
                # Decode target address based on addressing mode
                if addr_mode == 'zp':
                    target_addr = rom_data[i + 1]
                elif addr_mode == 'abs':
                    target_addr = struct.unpack('<H', rom_data[i+1:i+3])[0]
                elif addr_mode in ['zp,X', 'zp,Y']:
                    # For indexed zero page, base address could hit TIA/RIOT with index
                    base_addr = rom_data[i + 1] 
                    target_addr = base_addr  # We'll check if base could hit target
                elif addr_mode in ['abs,X', 'abs,Y']:
                    base_addr = struct.unpack('<H', rom_data[i+1:i+3])[0]
                    target_addr = base_addr
                # Skip indirect addressing for now (complex to analyze)
                
                if target_addr is not None:
                    # Get context
                    context_start = max(0, i - 3)
                    context_end = min(len(rom_data), i + inst_len + 3)
                    context = rom_data[context_start:context_end]
                    
                    is_read = inst_name.startswith('LD')
                    
                    access = MemoryAccess(
                        pc=i,
                        instruction=inst_name,
                        addressing_mode=addr_mode,
                        target_address=target_addr,
                        access_type='OTHER',
                        is_read=is_read,
                        context_bytes=context
                    )
                    
                    # Classify the access
                    if is_tia_address(target_addr):
                        access.access_type = 'TIA'
                        tia_accesses.append(access)
                    elif is_riot_address(target_addr):
                        riot_type = get_riot_type(target_addr)
                        access.access_type = f'RIOT_{riot_type}'
                        riot_accesses.append(access)
        
        i += 1
    
    return tia_accesses, riot_accesses

def analyze_rom_file_fixed(filepath: Path) -> ROMAnalysis:
    """Fixed analysis of ROM with proper memory mapping"""
    try:
        with open(filepath, 'rb') as f:
            rom_data = f.read()
        
        if len(rom_data) < 2:
            return None
            
        size = len(rom_data)
        
        # Basic metrics
        opcode_ratio = sum(1 for byte in rom_data if byte in VALID_OPCODES) / len(rom_data)
        
        # Memory access analysis
        tia_accesses, riot_accesses = analyze_memory_accesses(rom_data)
        
        # Classify RIOT accesses
        riot_timer_count = sum(1 for acc in riot_accesses if acc.access_type == 'RIOT_TIMER')
        riot_io_count = sum(1 for acc in riot_accesses if acc.access_type == 'RIOT_IO')
        
        # Other metrics
        branch_count = sum(1 for byte in rom_data if byte in {0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0})
        jump_count = sum(1 for byte in rom_data if byte in {0x4C, 0x6C, 0x20})
        unique_ops = len(set(byte for byte in rom_data if byte in VALID_OPCODES))
        
        # Calculate score
        score = (
            opcode_ratio * 0.25 +
            min(len(tia_accesses) / 50.0, 1.0) * 0.30 +
            min(len(riot_accesses) / 10.0, 1.0) * 0.15 +
            min(branch_count / 100.0, 1.0) * 0.15 +
            min(jump_count / 20.0, 1.0) * 0.10 +
            min(unique_ops / 50.0, 1.0) * 0.05
        )
        
        return ROMAnalysis(
            filename=filepath.name,
            size=size,
            valid_opcodes_ratio=opcode_ratio,
            tia_accesses=tia_accesses,
            riot_accesses=riot_accesses,
            tia_access_count=len(tia_accesses),
            riot_access_count=len(riot_accesses),
            riot_timer_count=riot_timer_count,
            riot_io_count=riot_io_count,
            branch_instructions=branch_count,
            jump_instructions=jump_count,
            unique_opcodes=unique_ops,
            score=score
        )
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

def print_fixed_analysis(analyses: List[ROMAnalysis]):
    """Print corrected analysis results"""
    
    print(f"\n🔧 FIXED ATARI 2600 MEMORY ANALYSIS")
    print("=" * 80)
    print("Now properly handling mirrored memory mapping!")
    
    # TIA Analysis
    all_tia = []
    for rom in analyses:
        all_tia.extend(rom.tia_accesses)
    
    print(f"\n🎮 TIA Analysis:")
    print(f"  Total TIA accesses: {len(all_tia):,}")
    
    # TIA by instruction
    tia_by_inst = Counter(acc.instruction for acc in all_tia)
    print(f"  Instructions:")
    for inst, count in tia_by_inst.most_common():
        pct = count / len(all_tia) * 100
        print(f"    {inst}: {count:,} ({pct:.1f}%)")
    
    # TIA by addressing mode
    tia_by_mode = Counter(acc.addressing_mode for acc in all_tia)
    print(f"  Addressing modes:")
    for mode, count in tia_by_mode.most_common():
        pct = count / len(all_tia) * 100
        print(f"    {mode}: {count:,} ({pct:.1f}%)")
    
    # RIOT Analysis  
    all_riot = []
    for rom in analyses:
        all_riot.extend(rom.riot_accesses)
    
    print(f"\n🕹️ RIOT Analysis:")
    print(f"  Total RIOT accesses: {len(all_riot):,}")
    
    # RIOT by type
    riot_by_type = Counter(acc.access_type for acc in all_riot)
    print(f"  Access types:")
    for riot_type, count in riot_by_type.most_common():
        pct = count / len(all_riot) * 100 if all_riot else 0
        print(f"    {riot_type}: {count:,} ({pct:.1f}%)")
    
    # RIOT by instruction
    riot_by_inst = Counter(acc.instruction for acc in all_riot)
    print(f"  Instructions:")
    for inst, count in riot_by_inst.most_common():
        pct = count / len(all_riot) * 100 if all_riot else 0
        print(f"    {inst}: {count:,} ({pct:.1f}%)")
    
    # Statistics
    tia_counts = [rom.tia_access_count for rom in analyses]
    riot_counts = [rom.riot_access_count for rom in analyses]
    
    print(f"\n📊 Statistics:")
    print(f"  TIA accesses per ROM:")
    print(f"    Min: {min(tia_counts)}, Max: {max(tia_counts)}")
    print(f"    Mean: {statistics.mean(tia_counts):.1f}, Median: {statistics.median(tia_counts):.1f}")
    
    print(f"  RIOT accesses per ROM:")
    print(f"    Min: {min(riot_counts)}, Max: {max(riot_counts)}")
    print(f"    Mean: {statistics.mean(riot_counts):.1f}, Median: {statistics.median(riot_counts):.1f}")
    
    # Thresholds
    sorted_tia = sorted(tia_counts)
    sorted_riot = sorted(riot_counts)
    n = len(analyses)
    
    print(f"\n🎯 CORRECTED Thresholds:")
    print(f"  TIA Accesses:")
    print(f"    5th percentile: {sorted_tia[int(n * 0.05)]} (EASY)")
    print(f"    10th percentile: {sorted_tia[int(n * 0.10)]} (MEDIUM)")
    print(f"    25th percentile: {sorted_tia[int(n * 0.25)]} (HARD)")
    
    print(f"  RIOT Accesses:")
    print(f"    5th percentile: {sorted_riot[int(n * 0.05)]} (EASY)")
    print(f"    10th percentile: {sorted_riot[int(n * 0.10)]} (MEDIUM)")  
    print(f"    25th percentile: {sorted_riot[int(n * 0.25)]} (HARD)")
    
    # Show some examples
    print(f"\n🔍 Examples of games with high RIOT usage:")
    high_riot_games = sorted(analyses, key=lambda x: x.riot_access_count, reverse=True)[:5]
    for rom in high_riot_games:
        print(f"  {rom.filename}: {rom.riot_access_count} RIOT accesses "
              f"({rom.riot_timer_count} timer, {rom.riot_io_count} I/O)")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python3 fixed_analyzer.py <directory_with_roms>")
        sys.exit(1)
    
    rom_dir = Path(sys.argv[1])
    
    if not rom_dir.exists():
        print(f"Directory {rom_dir} does not exist!")
        sys.exit(1)
    
    rom_files = list(rom_dir.glob("*.bin")) + list(rom_dir.glob("*.BIN"))
    
    if not rom_files:
        print(f"No .bin files found in {rom_dir}")
        sys.exit(1)
    
    print(f"🔍 Analyzing {len(rom_files)} ROM files with FIXED memory mapping...")
    
    analyses = []
    for i, rom_file in enumerate(rom_files):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(rom_files)} ROMs...")
        
        analysis = analyze_rom_file_fixed(rom_file)
        if analysis:
            analyses.append(analysis)
    
    print(f"✅ Successfully analyzed {len(analyses)} ROMs")
    
    print_fixed_analysis(analyses)
    
    # Save corrected results
    with open("fixed_memory_analysis.csv", "w") as f:
        f.write("ROM,TIA_Count,RIOT_Count,RIOT_Timer,RIOT_IO,Branches,Jumps,Opcodes,Score\n")
        for rom in analyses:
            f.write(f"{rom.filename},{rom.tia_access_count},{rom.riot_access_count},"
                   f"{rom.riot_timer_count},{rom.riot_io_count},{rom.branch_instructions},"
                   f"{rom.jump_instructions},{rom.unique_opcodes},{rom.score:.4f}\n")
    
    print(f"\n💾 Fixed analysis saved to 'fixed_memory_analysis.csv'")

if __name__ == "__main__":
    main()