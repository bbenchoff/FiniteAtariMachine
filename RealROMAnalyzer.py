#!/usr/bin/env python3
"""
Atari ROM Collection Analyzer
Analyze all real Atari 2600 ROMs to calibrate our heuristics.
"""

import os
import struct
import glob
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import statistics

# Copy the same constants and functions from the generator
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

BRANCH_OPCODES = {0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0}
JUMP_OPCODES = {0x4C, 0x6C, 0x20}

TIA_REGISTERS = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
    0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23,
    0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F
}

RIOT_REGISTERS = {0x280, 0x281, 0x282, 0x283, 0x284, 0x285, 0x286, 0x287,
                  0x294, 0x295, 0x296, 0x297}

@dataclass
class ROMAnalysis:
    filename: str
    size: int
    valid_opcodes_ratio: float
    tia_accesses: int
    riot_accesses: int
    branch_instructions: int
    jump_instructions: int
    unique_opcodes: int
    has_loop_patterns: bool
    has_valid_reset_vector: bool
    score: float

def analyze_rom_file(filepath: Path) -> ROMAnalysis:
    """Analyze a single ROM file"""
    try:
        with open(filepath, 'rb') as f:
            rom_data = f.read()
        
        # Handle different ROM sizes (some are 2K, 4K, 8K, etc.)
        if len(rom_data) < 2:
            return None
            
        size = len(rom_data)
        
        # Calculate metrics
        opcode_ratio = check_opcode_sanity(rom_data)
        tia_count = count_tia_accesses(rom_data)
        riot_count = count_riot_accesses(rom_data)
        branch_count = count_branches(rom_data)
        jump_count = count_jumps(rom_data)
        unique_ops = count_unique_opcodes(rom_data)
        has_loops = detect_loop_patterns(rom_data)
        valid_reset = check_reset_vector(rom_data)
        
        # Use same scoring as generator
        score = (
            opcode_ratio * 0.25 +
            min(tia_count / 20.0, 1.0) * 0.20 +
            min(riot_count / 10.0, 1.0) * 0.15 +
            min(branch_count / 15.0, 1.0) * 0.15 +
            min(jump_count / 8.0, 1.0) * 0.10 +
            min(unique_ops / 30.0, 1.0) * 0.10 +
            (0.05 if has_loops else 0.0) +
            (0.05 if valid_reset else 0.0)
        )
        
        return ROMAnalysis(
            filename=filepath.name,
            size=size,
            valid_opcodes_ratio=opcode_ratio,
            tia_accesses=tia_count,
            riot_accesses=riot_count,
            branch_instructions=branch_count,
            jump_instructions=jump_count,
            unique_opcodes=unique_ops,
            has_loop_patterns=has_loops,
            has_valid_reset_vector=valid_reset,
            score=score
        )
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

def check_opcode_sanity(rom_data: bytes) -> float:
    """Check what percentage of bytes are valid 6502 opcodes"""
    if not rom_data:
        return 0.0
    valid_count = sum(1 for byte in rom_data if byte in VALID_OPCODES)
    return valid_count / len(rom_data)

def count_tia_accesses(rom_data: bytes) -> int:
    """Count potential TIA register accesses"""
    count = 0
    for i in range(len(rom_data) - 2):
        # STA absolute
        if rom_data[i] == 0x8D and i + 2 < len(rom_data):
            addr = struct.unpack('<H', rom_data[i+1:i+3])[0]
            if (addr & 0xFF) in TIA_REGISTERS:
                count += 1
        # STA zero page
        elif rom_data[i] == 0x85 and i + 1 < len(rom_data):
            if rom_data[i+1] in TIA_REGISTERS:
                count += 1
    return count

def count_riot_accesses(rom_data: bytes) -> int:
    """Count potential RIOT register accesses"""
    count = 0
    for i in range(len(rom_data) - 2):
        if rom_data[i] == 0x8D and i + 2 < len(rom_data):
            addr = struct.unpack('<H', rom_data[i+1:i+3])[0]
            if addr in RIOT_REGISTERS:
                count += 1
    return count

def count_branches(rom_data: bytes) -> int:
    """Count branch instructions"""
    return sum(1 for byte in rom_data if byte in BRANCH_OPCODES)

def count_jumps(rom_data: bytes) -> int:
    """Count jump/call instructions"""
    return sum(1 for byte in rom_data if byte in JUMP_OPCODES)

def count_unique_opcodes(rom_data: bytes) -> int:
    """Count number of unique valid opcodes used"""
    used_opcodes = set()
    for byte in rom_data:
        if byte in VALID_OPCODES:
            used_opcodes.add(byte)
    return len(used_opcodes)

def detect_loop_patterns(rom_data: bytes) -> bool:
    """Look for potential loop patterns (branch backwards)"""
    for i in range(len(rom_data) - 1):
        if rom_data[i] in BRANCH_OPCODES and i + 1 < len(rom_data):
            offset = struct.unpack('b', bytes([rom_data[i+1]]))[0]
            if offset < 0:
                return True
    return False

def check_reset_vector(rom_data: bytes) -> bool:
    """Check if the reset vector points to valid ROM space"""
    if len(rom_data) < 2:
        return False
    
    # Different ROM sizes have different address spaces
    reset_addr = struct.unpack('<H', rom_data[-2:])[0]
    
    # Common Atari 2600 ROM mappings
    if len(rom_data) == 4096:  # 4K ROM
        return 0xF000 <= reset_addr <= 0xFFFF
    elif len(rom_data) == 2048:  # 2K ROM  
        return 0xF800 <= reset_addr <= 0xFFFF
    elif len(rom_data) == 8192:  # 8K ROM
        return 0xE000 <= reset_addr <= 0xFFFF
    else:
        # Generic check - high byte should be reasonable
        return reset_addr >= 0xE000

def print_statistics(analyses: List[ROMAnalysis]):
    """Print detailed statistics about the ROM collection"""
    
    if not analyses:
        print("No valid ROMs analyzed!")
        return
    
    print(f"\n📊 Analysis of {len(analyses)} Atari 2600 ROMs")
    print("=" * 60)
    
    # Size distribution
    sizes = [rom.size for rom in analyses]
    print(f"\n📏 ROM Sizes:")
    print(f"  Min: {min(sizes):,} bytes")
    print(f"  Max: {max(sizes):,} bytes") 
    print(f"  Most common: {statistics.mode(sizes):,} bytes")
    
    # Helper function to print percentiles
    def print_metric_stats(name: str, values: list, is_percentage: bool = False):
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        p5 = sorted_vals[int(n * 0.05)]
        p10 = sorted_vals[int(n * 0.10)]
        p25 = sorted_vals[int(n * 0.25)]
        p50 = statistics.median(values)
        p75 = sorted_vals[int(n * 0.75)]
        p90 = sorted_vals[int(n * 0.90)]
        p95 = sorted_vals[int(n * 0.95)]
        
        print(f"\n{name}:")
        print(f"  Min: {min(values):.3f}" + (f" ({min(values)*100:.1f}%)" if is_percentage else ""))
        print(f"  5th percentile: {p5:.3f}" + (f" ({p5*100:.1f}%)" if is_percentage else ""))
        print(f"  10th percentile: {p10:.3f}" + (f" ({p10*100:.1f}%)" if is_percentage else ""))
        print(f"  25th percentile: {p25:.3f}" + (f" ({p25*100:.1f}%)" if is_percentage else ""))
        print(f"  Median (50th): {p50:.3f}" + (f" ({p50*100:.1f}%)" if is_percentage else ""))
        print(f"  75th percentile: {p75:.3f}" + (f" ({p75*100:.1f}%)" if is_percentage else ""))
        print(f"  90th percentile: {p90:.3f}" + (f" ({p90*100:.1f}%)" if is_percentage else ""))
        print(f"  95th percentile: {p95:.3f}" + (f" ({p95*100:.1f}%)" if is_percentage else ""))
        print(f"  Max: {max(values):.3f}" + (f" ({max(values)*100:.1f}%)" if is_percentage else ""))
        print(f"  Mean: {statistics.mean(values):.3f}" + (f" ({statistics.mean(values)*100:.1f}%)" if is_percentage else ""))
    
    # Detailed percentile analysis for each metric
    opcodes = [rom.valid_opcodes_ratio for rom in analyses]
    print_metric_stats("🔧 Valid Opcode Ratios", opcodes, is_percentage=True)
    
    tia_counts = [rom.tia_accesses for rom in analyses]
    print_metric_stats("🎮 TIA Accesses", tia_counts)
    
    riot_counts = [rom.riot_accesses for rom in analyses]
    print_metric_stats("🕹️ RIOT Accesses", riot_counts)
    
    branches = [rom.branch_instructions for rom in analyses]
    print_metric_stats("🔀 Branch Instructions", branches)
    
    jumps = [rom.jump_instructions for rom in analyses]
    print_metric_stats("🦘 Jump Instructions", jumps)
    
    unique = [rom.unique_opcodes for rom in analyses]
    print_metric_stats("🎯 Unique Opcodes", unique)
    
    scores = [rom.score for rom in analyses]
    print_metric_stats("🏆 Overall Scores", scores)
    
    # Loop patterns
    loops = sum(1 for rom in analyses if rom.has_loop_patterns)
    print(f"\n🔄 Loop Patterns: {loops}/{len(analyses)} ({loops/len(analyses)*100:.1f}%)")
    
    # Reset vectors
    vectors = sum(1 for rom in analyses if rom.has_valid_reset_vector)
    print(f"\n🎯 Valid Reset Vectors: {vectors}/{len(analyses)} ({vectors/len(analyses)*100:.1f}%)")
    
    # SUGGESTED THRESHOLDS FOR RANDOM ROM GENERATION
    print(f"\n" + "="*80)
    print(f"💡 SUGGESTED THRESHOLDS FOR RANDOM ROM GENERATION")
    print(f"="*80)
    
    # Use 5th-10th percentiles as thresholds (bottom 5-10% of real games)
    sorted_opcodes = sorted(opcodes)
    sorted_tia = sorted(tia_counts)
    sorted_riot = sorted(riot_counts)
    sorted_branches = sorted(branches)
    sorted_jumps = sorted(jumps)
    sorted_unique = sorted(unique)
    sorted_scores = sorted(scores)
    
    n = len(analyses)
    
    # 5th percentile thresholds (very achievable)
    print(f"\n🟢 EASY Thresholds (5th percentile - bottom 5% of real games):")
    print(f"  OPCODE_THRESHOLD = {sorted_opcodes[int(n * 0.05)]:.2f}")
    print(f"  TIA_THRESHOLD = {sorted_tia[int(n * 0.05)]}")
    print(f"  RIOT_THRESHOLD = {sorted_riot[int(n * 0.05)]}")
    print(f"  BRANCH_THRESHOLD = {sorted_branches[int(n * 0.05)]}")
    print(f"  JUMP_THRESHOLD = {sorted_jumps[int(n * 0.05)]}")
    print(f"  INSTRUCTION_VARIETY = {sorted_unique[int(n * 0.05)]}")
    print(f"  MIN_SCORE = {sorted_scores[int(n * 0.05)]:.2f}")
    
    # 10th percentile thresholds (still achievable)
    print(f"\n🟡 MEDIUM Thresholds (10th percentile - bottom 10% of real games):")
    print(f"  OPCODE_THRESHOLD = {sorted_opcodes[int(n * 0.10)]:.2f}")
    print(f"  TIA_THRESHOLD = {sorted_tia[int(n * 0.10)]}")
    print(f"  RIOT_THRESHOLD = {sorted_riot[int(n * 0.10)]}")
    print(f"  BRANCH_THRESHOLD = {sorted_branches[int(n * 0.10)]}")
    print(f"  JUMP_THRESHOLD = {sorted_jumps[int(n * 0.10)]}")
    print(f"  INSTRUCTION_VARIETY = {sorted_unique[int(n * 0.10)]}")
    print(f"  MIN_SCORE = {sorted_scores[int(n * 0.10)]:.2f}")
    
    # 25th percentile thresholds (more selective)
    print(f"\n🟠 HARD Thresholds (25th percentile - bottom 25% of real games):")
    print(f"  OPCODE_THRESHOLD = {sorted_opcodes[int(n * 0.25)]:.2f}")
    print(f"  TIA_THRESHOLD = {sorted_tia[int(n * 0.25)]}")
    print(f"  RIOT_THRESHOLD = {sorted_riot[int(n * 0.25)]}")
    print(f"  BRANCH_THRESHOLD = {sorted_branches[int(n * 0.25)]}")
    print(f"  JUMP_THRESHOLD = {sorted_jumps[int(n * 0.25)]}")
    print(f"  INSTRUCTION_VARIETY = {sorted_unique[int(n * 0.25)]}")
    print(f"  MIN_SCORE = {sorted_scores[int(n * 0.25)]:.2f}")
    
    print(f"\n💭 Recommendation: Start with EASY thresholds, then move up if you find too many.")
    
    # Current thresholds analysis (keep this)
    print(f"\n🚨 Current Generator Threshold Analysis:")
    current_thresholds = {
        'opcodes': 0.70,
        'tia': 50,
        'riot': 1, 
        'branches': 100,
        'jumps': 25,
        'unique': 50,
        'score': 0.75
    }
    
    passing_opcodes = sum(1 for rom in analyses if rom.valid_opcodes_ratio >= current_thresholds['opcodes'])
    passing_tia = sum(1 for rom in analyses if rom.tia_accesses >= current_thresholds['tia'])
    passing_riot = sum(1 for rom in analyses if rom.riot_accesses >= current_thresholds['riot'])
    passing_branches = sum(1 for rom in analyses if rom.branch_instructions >= current_thresholds['branches'])
    passing_jumps = sum(1 for rom in analyses if rom.jump_instructions >= current_thresholds['jumps'])
    passing_unique = sum(1 for rom in analyses if rom.unique_opcodes >= current_thresholds['unique'])
    passing_score = sum(1 for rom in analyses if rom.score >= current_thresholds['score'])
    
    print(f"  Opcodes ≥70%: {passing_opcodes}/{len(analyses)} ({passing_opcodes/len(analyses)*100:.1f}%)")
    print(f"  TIA ≥50: {passing_tia}/{len(analyses)} ({passing_tia/len(analyses)*100:.1f}%)")
    print(f"  RIOT ≥1: {passing_riot}/{len(analyses)} ({passing_riot/len(analyses)*100:.1f}%)")
    print(f"  Branches ≥100: {passing_branches}/{len(analyses)} ({passing_branches/len(analyses)*100:.1f}%)")
    print(f"  Jumps ≥25: {passing_jumps}/{len(analyses)} ({passing_jumps/len(analyses)*100:.1f}%)")
    print(f"  Unique ≥50: {passing_unique}/{len(analyses)} ({passing_unique/len(analyses)*100:.1f}%)")
    print(f"  Score ≥0.75: {passing_score}/{len(analyses)} ({passing_score/len(analyses)*100:.1f}%)")
    
    # ALL thresholds
    passing_all = sum(1 for rom in analyses if (
        rom.valid_opcodes_ratio >= current_thresholds['opcodes'] and
        rom.tia_accesses >= current_thresholds['tia'] and
        rom.riot_accesses >= current_thresholds['riot'] and
        rom.branch_instructions >= current_thresholds['branches'] and
        rom.jump_instructions >= current_thresholds['jumps'] and
        rom.unique_opcodes >= current_thresholds['unique'] and
        rom.score >= current_thresholds['score']
    ))
    
    print(f"\n💎 ROMs passing ALL current thresholds: {passing_all}/{len(analyses)} ({passing_all/len(analyses)*100:.3f}%)")
    
    if passing_all < len(analyses) * 0.05:  # Less than 5% pass
        print("❌ CURRENT THRESHOLDS ARE TOO HARSH!")
        print("   Consider using the EASY or MEDIUM suggested thresholds above.")
    elif passing_all > len(analyses) * 0.5:  # More than 50% pass
        print("⚠️  CURRENT THRESHOLDS MIGHT BE TOO EASY!")
        print("   Consider using HARD thresholds for more selectivity.")
    else:
        print("✅ Current thresholds look reasonable!")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python3 rom_calibrator.py <directory_with_roms>")
        print("Example: python3 rom_calibrator.py ./atari_roms/")
        sys.exit(1)
    
    rom_dir = Path(sys.argv[1])
    
    if not rom_dir.exists():
        print(f"Directory {rom_dir} does not exist!")
        sys.exit(1)
    
    # Find all .bin files
    rom_files = list(rom_dir.glob("*.bin")) + list(rom_dir.glob("*.BIN"))
    
    if not rom_files:
        print(f"No .bin files found in {rom_dir}")
        sys.exit(1)
    
    print(f"🔍 Found {len(rom_files)} ROM files")
    print("Analyzing...")
    
    analyses = []
    for i, rom_file in enumerate(rom_files):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(rom_files)} ROMs...")
        
        analysis = analyze_rom_file(rom_file)
        if analysis:
            analyses.append(analysis)
    
    print(f"✅ Successfully analyzed {len(analyses)} ROMs")
    
    print_statistics(analyses)
    
    # Save detailed results
    with open("rom_analysis_results.txt", "w") as f:
        f.write("Filename,Size,Opcodes,TIA,RIOT,Branches,Jumps,Unique,Loops,ResetVector,Score\n")
        for rom in analyses:
            f.write(f"{rom.filename},{rom.size},{rom.valid_opcodes_ratio:.4f},"
                   f"{rom.tia_accesses},{rom.riot_accesses},{rom.branch_instructions},"
                   f"{rom.jump_instructions},{rom.unique_opcodes},{rom.has_loop_patterns},"
                   f"{rom.has_valid_reset_vector},{rom.score:.4f}\n")
    
    print(f"\n💾 Detailed results saved to 'rom_analysis_results.txt'")

if __name__ == "__main__":
    main()