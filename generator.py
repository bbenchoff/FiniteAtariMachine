#!/usr/bin/env python3
"""
Finite Atari Machine - Multicore ROM Generator (Harsh Edition)
Generates Atari 2600 ROM bases continuously with much stricter heuristics.
Uses all CPU cores for maximum ROM generation throughput.
"""

import os
import random
import struct
import time
import signal
import sys
import multiprocessing as mp
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from pathlib import Path
import queue
import threading

# 6502/6507 valid opcodes (151 of them)
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

# Branch instructions for flow control detection
BRANCH_OPCODES = {0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0}

# Jump instructions  
JUMP_OPCODES = {0x4C, 0x6C, 0x20}  # JMP abs, JMP ind, JSR

# Atari 2600 memory map constants
ROM_SIZE = 4094  # 4KB minus 2 bytes for reset vector
TIA_REGISTERS = {
    # Write registers
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
    0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23,
    0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F
}

RIOT_REGISTERS = {0x280, 0x281, 0x282, 0x283, 0x284, 0x285, 0x286, 0x287,
                  0x294, 0x295, 0x296, 0x297}

# EASY THRESHOLDS (5th percentile - bottom 5% of real games)
OPCODE_THRESHOLD = 0.66        # 66% valid opcodes (real games 5th percentile)
TIA_THRESHOLD = 51             # At least 51 TIA accesses (real games 5th percentile)
RIOT_THRESHOLD = 1             # At least 1 RIOT access (real games 5th percentile)
BRANCH_THRESHOLD = 177         # At least 177 branch instructions (real games 5th percentile)
JUMP_THRESHOLD = 37            # At least 37 jumps/calls (real games 5th percentile)
INSTRUCTION_VARIETY = 125      # At least 125 different opcodes (real games 5th percentile)
MIN_SCORE = 0.80               # Minimum composite score (real games 5th percentile)

@dataclass
class HeuristicResult:
    """Results from applying heuristics to a ROM"""
    valid_opcodes_ratio: float
    tia_accesses: int
    riot_accesses: int
    branch_instructions: int
    jump_instructions: int
    unique_opcodes: int
    has_loop_patterns: bool
    has_interrupt_vectors: bool
    score: float
    
    def is_promising(self) -> bool:
        """Much stricter criteria for what we consider worth saving"""
        return (self.valid_opcodes_ratio >= OPCODE_THRESHOLD and 
                self.tia_accesses >= TIA_THRESHOLD and 
                self.riot_accesses >= RIOT_THRESHOLD and
                self.branch_instructions >= BRANCH_THRESHOLD and
                self.jump_instructions >= JUMP_THRESHOLD and
                self.unique_opcodes >= INSTRUCTION_VARIETY and
                self.score >= MIN_SCORE)

@dataclass
class PromisingROM:
    """Container for a promising ROM and its analysis"""
    rom_data: bytes
    result: HeuristicResult
    worker_id: int
    generation_number: int

def worker_process(worker_id: int, result_queue: mp.Queue, stats_queue: mp.Queue, stop_event: mp.Event):
    """Worker process that generates and analyzes ROMs"""
    # Seed each worker differently
    random.seed(int(time.time() * 1000) + worker_id)
    
    local_generated = 0
    local_promising = 0
    
    while not stop_event.is_set():
        try:
            # Generate MUCH larger batches to reduce queue overhead
            batch_size = 10000  # 100x bigger batches
            for batch_i in range(batch_size):
                if stop_event.is_set():
                    break
                    
                # Generate ROM base
                rom_base = bytes(random.randint(0, 255) for _ in range(ROM_SIZE))
                local_generated += 1
                
                # Analyze it
                result = analyze_rom_base(rom_base)
                
                # If promising, send to main process
                if result.is_promising():
                    local_promising += 1
                    promising_rom = PromisingROM(
                        rom_data=rom_base,
                        result=result,
                        worker_id=worker_id,
                        generation_number=local_generated
                    )
                    result_queue.put(promising_rom)
            
            # Send stats update much less frequently
            if local_generated % 10000 == 0:  # Only every 10k ROMs
                stats_queue.put((worker_id, local_generated, local_promising))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            break

def analyze_rom_base(rom_base: bytes) -> HeuristicResult:
    """Apply heuristics to ROM base - moved to module level for multiprocessing"""
    opcode_ratio = check_opcode_sanity(rom_base)
    tia_count = count_tia_accesses(rom_base)
    riot_count = count_riot_accesses(rom_base)
    branch_count = count_branches(rom_base)
    jump_count = count_jumps(rom_base)
    unique_ops = count_unique_opcodes(rom_base)
    has_loops = detect_loop_patterns(rom_base)
    has_vectors = check_interrupt_vectors(rom_base)
    
    # Weighted scoring system - much higher standards
    score = (
        opcode_ratio * 0.25 +                           # Basic validity
        min(tia_count / 20.0, 1.0) * 0.20 +            # Graphics capability
        min(riot_count / 10.0, 1.0) * 0.15 +           # I/O capability  
        min(branch_count / 15.0, 1.0) * 0.15 +         # Control flow
        min(jump_count / 8.0, 1.0) * 0.10 +            # Subroutines/structure
        min(unique_ops / 30.0, 1.0) * 0.10 +           # Instruction variety
        (0.05 if has_loops else 0.0) +                 # Loop structures
        (0.05 if has_vectors else 0.0)                 # Proper vectors
    )
    
    return HeuristicResult(
        valid_opcodes_ratio=opcode_ratio,
        tia_accesses=tia_count,
        riot_accesses=riot_count,
        branch_instructions=branch_count,
        jump_instructions=jump_count,
        unique_opcodes=unique_ops,
        has_loop_patterns=has_loops,
        has_interrupt_vectors=has_vectors,
        score=score
    )

# Heuristic functions moved to module level for multiprocessing
def check_opcode_sanity(rom_data: bytes) -> float:
    """Check what percentage of bytes are valid 6502 opcodes"""
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
    """Count branch instructions (indicates control flow)"""
    return sum(1 for byte in rom_data if byte in BRANCH_OPCODES)

def count_jumps(rom_data: bytes) -> int:
    """Count jump/call instructions (indicates structure)"""
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
            # Branch offset (signed byte)
            offset = struct.unpack('b', bytes([rom_data[i+1]]))[0]
            if offset < 0:  # Backward branch = potential loop
                return True
    return False

def check_interrupt_vectors(rom_data: bytes) -> bool:
    """Check if ROM has reasonable interrupt vector area"""
    if len(rom_data) < 16:
        return False
    
    vector_area = rom_data[-16:]
    # Count how many bytes look like high bytes of ROM addresses (0xF0-0xFF)
    high_bytes = sum(1 for i in range(1, len(vector_area), 2) 
                    if vector_area[i] >= 0xF0)
    
    return high_bytes >= 3  # At least 3 vectors pointing to ROM

class MulticoreAtariGenerator:
    def __init__(self, num_workers=None):
        if num_workers is None:
            self.num_workers = 8  # Default to 8 cores instead of all cores
        else:
            self.num_workers = num_workers
            
        self.total_generated = 0
        self.promising_found = 0
        self.worker_stats = {}
        self.start_time = time.time()
        self.output_dir = Path("promising_roms")
        self.output_dir.mkdir(exist_ok=True)
        
        # Multiprocessing objects
        self.result_queue = mp.Queue()
        self.stats_queue = mp.Queue() 
        self.stop_event = mp.Event()
        self.workers = []
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        print(f"\n\n🛑 Shutting down workers...")
        self.stop_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2)
            if worker.is_alive():
                worker.terminate()
        
        print(f"\n📊 Final Statistics:")
        print(f"Workers used: {self.num_workers}")
        print(f"Total base ROMs generated: {self.total_generated:,}")
        print(f"Promising ROMs found: {self.promising_found}")
        print(f"Success rate: {self.promising_found/max(self.total_generated,1)*100:.6f}%")
        runtime = time.time() - self.start_time
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Rate: {self.total_generated/runtime:.1f} ROMs/sec")
        print(f"Effective Rate: {self.total_generated/runtime*self.num_workers:.1f} ROMs/sec (multicore)")
        
        print(f"\nPer-worker stats:")
        for worker_id, (generated, promising) in self.worker_stats.items():
            print(f"  Worker {worker_id}: {generated:,} generated, {promising} promising")
        
        print("\n👋 Exiting gracefully...")
        sys.exit(0)
    
    def save_promising_rom(self, promising_rom: PromisingROM):
        """Save a promising ROM base for later reset vector testing"""
        result = promising_rom.result
        filename = f"base_{self.promising_found:06d}_score_{result.score:.3f}_w{promising_rom.worker_id}.bin"
        filepath = self.output_dir / filename
        
        with open(filepath, "wb") as f:
            f.write(promising_rom.rom_data)
        
        # Also save metadata
        meta_filename = f"base_{self.promising_found:06d}_score_{result.score:.3f}_w{promising_rom.worker_id}.txt"
        meta_filepath = self.output_dir / meta_filename
        
        with open(meta_filepath, "w") as f:
            f.write(f"ROM Base Analysis\n")
            f.write(f"=================\n")
            f.write(f"Worker ID: {promising_rom.worker_id}\n")
            f.write(f"Worker Generation #: {promising_rom.generation_number}\n")
            f.write(f"Overall Score: {result.score:.4f}\n")
            f.write(f"Valid Opcodes: {result.valid_opcodes_ratio:.3f} ({result.valid_opcodes_ratio*100:.1f}%)\n")
            f.write(f"TIA Accesses: {result.tia_accesses}\n")
            f.write(f"RIOT Accesses: {result.riot_accesses}\n")
            f.write(f"Branch Instructions: {result.branch_instructions}\n")
            f.write(f"Jump Instructions: {result.jump_instructions}\n")
            f.write(f"Unique Opcodes: {result.unique_opcodes}\n")
            f.write(f"Has Loop Patterns: {result.has_loop_patterns}\n")
            f.write(f"Has Interrupt Vectors: {result.has_interrupt_vectors}\n")
        
        print(f"💎 SAVED: {filename}")
        print(f"    Worker: {promising_rom.worker_id} | Score: {result.score:.4f} | Opcodes: {result.valid_opcodes_ratio:.3f}")
        print(f"    TIA: {result.tia_accesses} | RIOT: {result.riot_accesses} | Branches: {result.branch_instructions}")
    
    def run_continuous(self):
        """Main continuous generation loop with multicore processing"""
        print("🕹️  Finite Atari Machine - EASY MODE")
        print("=" * 80)
        print(f"Using {self.num_workers} CPU cores")
        print(f"Saving promising ROM bases to: {self.output_dir}")
        print(f"🟢 EASY Thresholds (5th percentile - bottom 5% of real games):")
        print(f"  • Opcodes ≥ {OPCODE_THRESHOLD*100:.0f}% (real games 5th %: 65.6%)")
        print(f"  • TIA accesses ≥ {TIA_THRESHOLD} (real games 5th %: 51)")
        print(f"  • RIOT accesses ≥ {RIOT_THRESHOLD} (real games 5th %: 1)")
        print(f"  • Branches ≥ {BRANCH_THRESHOLD} (real games 5th %: 177)")
        print(f"  • Jumps ≥ {JUMP_THRESHOLD} (real games 5th %: 37)")
        print(f"  • Unique opcodes ≥ {INSTRUCTION_VARIETY} (real games 5th %: 125)")
        print(f"  • Minimum score ≥ {MIN_SCORE} (real games 5th %: 0.80)")
        print("These match the WORST 5% of functional games - much more achievable!")
        print("Press Ctrl+C to stop and see statistics")
        print("=" * 80)
        
        # Start worker processes
        print(f"🚀 Starting {self.num_workers} worker processes...")
        for i in range(self.num_workers):
            worker = mp.Process(target=worker_process, 
                              args=(i, self.result_queue, self.stats_queue, self.stop_event))
            worker.start()
            self.workers.append(worker)
            self.worker_stats[i] = (0, 0)  # (generated, promising)
        
        last_report_time = time.time()
        
        try:
            while True:
                # Check for promising ROMs
                try:
                    promising_rom = self.result_queue.get_nowait()
                    self.promising_found += 1
                    self.save_promising_rom(promising_rom)
                except queue.Empty:
                    pass
                
                # Check for stats updates
                try:
                    worker_id, generated, promising = self.stats_queue.get_nowait()
                    self.worker_stats[worker_id] = (generated, promising)
                except queue.Empty:
                    pass
                
                # Update total stats
                self.total_generated = sum(stats[0] for stats in self.worker_stats.values())
                
                # Print status every few seconds
                current_time = time.time()
                if current_time - last_report_time >= 10.0:  # Less frequent updates
                    elapsed = current_time - self.start_time
                    rate = self.total_generated / elapsed if elapsed > 0 else 0
                    success_rate = self.promising_found / max(self.total_generated, 1) * 100
                    
                    print(f"\r🔍 Generated: {self.total_generated:,} | "
                          f"Promising: {self.promising_found} | "
                          f"Success: {success_rate:.6f}% | "
                          f"Rate: {rate:.0f}/sec | "
                          f"Runtime: {elapsed:.0f}s", end="", flush=True)
                    
                    last_report_time = current_time
                
                time.sleep(0.5)  # Longer delay to reduce main thread overhead
                    
        except KeyboardInterrupt:
            self._signal_handler(signal.SIGINT, None)

def main():
    # Default to 8 cores
    num_cores = mp.cpu_count()
    if len(sys.argv) > 1:
        try:
            num_workers = int(sys.argv[1])
            if num_workers <= 0:
                num_workers = 8
        except ValueError:
            num_workers = 8
    else:
        num_workers = 8
    
    print(f"System has {num_cores} CPU cores, using {num_workers} workers")
    
    generator = MulticoreAtariGenerator(num_workers=num_workers)
    generator.run_continuous()

if __name__ == "__main__":
    main()