import os
import csv
import hashlib
from pathlib import Path
from collections import defaultdict

# Valid 6502 opcodes for Atari 2600
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

# Control flow opcodes
BRANCH_OPCODES = {0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0}
JUMP_OPCODES = {0x4C, 0x6C, 0x20}

# TIA instruction patterns (from the original analysis)
TIA_STORE_ZP = {0x85, 0x86, 0x84}  # STA, STX, STY zero page
TIA_STORE_ZPX = {0x95, 0x96, 0x94}  # STA, STX, STY zero page,X
TIA_LOAD_ZP = {0xA5, 0xA6, 0xA4}   # LDA, LDX, LDY zero page
TIA_LOAD_ZPX = {0xB5, 0xB6, 0xB4}  # LDA, LDX, LDY zero page,X
TIA_ABS = {0x8D, 0x8E, 0x8C, 0xAD, 0xAE, 0xAC}  # Absolute addressing

# RIOT instruction patterns
RIOT_ACCESS = {0x85, 0x86, 0x84, 0xA5, 0xA6, 0xA4}

def analyze_rom(rom_data):
    """Analyze a single ROM and return all metrics."""
    rom_size = len(rom_data)
    
    # Basic opcode analysis
    valid_opcodes_count = sum(1 for byte in rom_data if byte in VALID_OPCODES)
    opcode_ratio = valid_opcodes_count / rom_size if rom_size > 0 else 0
    
    # Control flow analysis
    branch_count = sum(1 for byte in rom_data if byte in BRANCH_OPCODES)
    jump_count = sum(1 for byte in rom_data if byte in JUMP_OPCODES)
    
    # Unique opcodes in first 1KB (code section)
    first_kb = rom_data[:1024] if len(rom_data) >= 1024 else rom_data
    unique_opcodes = len(set(first_kb) & VALID_OPCODES)
    
    # TIA accesses (graphics chip)
    tia_accesses = 0
    
    # Zero page addressing patterns
    for i in range(len(rom_data) - 1):
        opcode = rom_data[i]
        operand = rom_data[i + 1]
        
        # Zero page TIA access (addresses 0x00-0x2F)
        if (opcode in TIA_STORE_ZP or opcode in TIA_STORE_ZPX or 
            opcode in TIA_LOAD_ZP or opcode in TIA_LOAD_ZPX) and operand <= 0x2F:
            tia_accesses += 1
    
    # Absolute addressing patterns (3-byte instructions)
    for i in range(len(rom_data) - 2):
        opcode = rom_data[i]
        low_byte = rom_data[i + 1]
        high_byte = rom_data[i + 2]
        
        # Absolute TIA access
        if opcode in TIA_ABS and low_byte <= 0x2F and high_byte == 0x00:
            tia_accesses += 1
    
    # RIOT accesses (RAM-I/O-Timer chip)
    riot_accesses = 0
    riot_timer_accesses = 0
    riot_io_accesses = 0
    
    for i in range(len(rom_data) - 1):
        opcode = rom_data[i]
        operand = rom_data[i + 1]
        
        if opcode in RIOT_ACCESS:
            # Timer registers: 0x80-0x87 (and mirrored addresses)
            if 0x80 <= operand <= 0x87:
                riot_timer_accesses += 1
                riot_accesses += 1
            # I/O registers: 0x94-0x97 (and mirrored addresses)
            elif 0x94 <= operand <= 0x97:
                riot_io_accesses += 1
                riot_accesses += 1
    
    # Composite score (from the original analysis)
    score = (
        opcode_ratio * 0.25 + 
        min(tia_accesses / 150.0, 1.0) * 0.30 + 
        min(riot_accesses / 50.0, 1.0) * 0.20 + 
        min(branch_count / 200.0, 1.0) * 0.15 + 
        min(jump_count / 40.0, 1.0) * 0.10
    )
    
    # Calculate hash for identification
    rom_hash = hashlib.md5(rom_data).hexdigest()[:8]
    
    return {
        'hash': rom_hash,
        'size': rom_size,
        'valid_opcodes_count': valid_opcodes_count,
        'opcode_ratio': opcode_ratio,
        'unique_opcodes': unique_opcodes,
        'tia_accesses': tia_accesses,
        'riot_accesses': riot_accesses,
        'riot_timer_accesses': riot_timer_accesses,
        'riot_io_accesses': riot_io_accesses,
        'branch_count': branch_count,
        'jump_count': jump_count,
        'composite_score': score
    }

def find_extremes(results):
    """Find ROMs with extreme (min/max) values for each metric."""
    extremes = {}
    
    metrics = [
        'unique_opcodes', 'tia_accesses', 'riot_accesses', 
        'branch_count', 'jump_count', 'composite_score'
    ]
    
    for metric in metrics:
        values = [r[metric] for r in results]
        min_val = min(values)
        max_val = max(values)
        
        min_roms = [r for r in results if r[metric] == min_val]
        max_roms = [r for r in results if r[metric] == max_val]
        
        extremes[metric] = {
            'min': {'value': min_val, 'roms': min_roms},
            'max': {'value': max_val, 'roms': max_roms}
        }
    
    return extremes

def main():
    # Look for ROMs in real_roms directory
    rom_dir = Path('real_roms')
    
    if not rom_dir.exists():
        print(f"Error: Directory '{rom_dir}' not found!")
        print("Please make sure you have a 'real_roms' directory with Atari ROM files.")
        return
    
    # Find all ROM files (common extensions)
    rom_extensions = {'.bin', '.rom', '.a26'}
    rom_files = []
    
    for ext in rom_extensions:
        rom_files.extend(rom_dir.glob(f'*{ext}'))
        rom_files.extend(rom_dir.glob(f'*{ext.upper()}'))
    
    if not rom_files:
        print(f"No ROM files found in '{rom_dir}'!")
        print(f"Looking for files with extensions: {', '.join(rom_extensions)}")
        return
    
    print(f"Found {len(rom_files)} ROM files. Analyzing...")
    
    results = []
    
    for rom_file in rom_files:
        try:
            with open(rom_file, 'rb') as f:
                rom_data = f.read()
            
            analysis = analyze_rom(rom_data)
            analysis['filename'] = rom_file.name
            results.append(analysis)
            
        except Exception as e:
            print(f"Error analyzing {rom_file}: {e}")
    
    if not results:
        print("No ROMs were successfully analyzed!")
        return
    
    # Sort by composite score (descending)
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    # Write CSV file
    csv_filename = 'atari_rom_analysis.csv'
    
    fieldnames = [
        'filename', 'hash', 'size', 'valid_opcodes_count', 'opcode_ratio',
        'unique_opcodes', 'tia_accesses', 'riot_accesses', 'riot_timer_accesses',
        'riot_io_accesses', 'branch_count', 'jump_count', 'composite_score'
    ]
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nAnalysis complete! Results saved to '{csv_filename}'")
    print(f"Analyzed {len(results)} ROMs")
    
    # Find and display extremes
    extremes = find_extremes(results)
    
    print("\n" + "="*80)
    print("EXTREME OUTLIERS (answering the AtariAge forum questions)")
    print("="*80)
    
    # Minimum unique opcodes
    min_opcodes = extremes['unique_opcodes']['min']
    print(f"\nFewest unique opcodes: {min_opcodes['value']}")
    for rom in min_opcodes['roms'][:3]:  # Show top 3
        print(f"  {rom['filename']} (hash: {rom['hash']})")
    
    # Minimum jumps
    min_jumps = extremes['jump_count']['min']
    print(f"\nFewest jumps: {min_jumps['value']}")
    for rom in min_jumps['roms'][:3]:
        print(f"  {rom['filename']} (hash: {rom['hash']})")
    
    # Minimum TIA accesses
    min_tia = extremes['tia_accesses']['min']
    print(f"\nFewest TIA accesses: {min_tia['value']}")
    for rom in min_tia['roms'][:3]:
        print(f"  {rom['filename']} (hash: {rom['hash']})")
    
    # Show some statistics
    print(f"\n" + "="*50)
    print("STATISTICS")
    print("="*50)
    
    avg_score = sum(r['composite_score'] for r in results) / len(results)
    avg_opcodes = sum(r['unique_opcodes'] for r in results) / len(results)
    avg_jumps = sum(r['jump_count'] for r in results) / len(results)
    avg_tia = sum(r['tia_accesses'] for r in results) / len(results)
    
    print(f"Average composite score: {avg_score:.3f}")
    print(f"Average unique opcodes: {avg_opcodes:.1f}")
    print(f"Average jump count: {avg_jumps:.1f}")
    print(f"Average TIA accesses: {avg_tia:.1f}")
    
    print(f"\nHighest scoring ROM: {results[0]['filename']} (score: {results[0]['composite_score']:.3f})")
    print(f"Lowest scoring ROM: {results[-1]['filename']} (score: {results[-1]['composite_score']:.3f})")

if __name__ == "__main__":
    main()