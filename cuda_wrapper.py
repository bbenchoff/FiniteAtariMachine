#!/usr/bin/env python3
"""
CUDA Atari ROM Generator Build and Run Script
Compiles and runs the CUDA version for maximum speed
"""

import subprocess
import os
import sys
import time

def check_cuda():
    """Check if CUDA is available"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA compiler found")
            print(result.stdout.split('\n')[3])  # Version line
            return True
        else:
            print("❌ CUDA compiler not found")
            return False
    except FileNotFoundError:
        print("❌ CUDA not installed or not in PATH")
        return False

def check_gpu():
    """Check GPU capabilities"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', 
                               '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            print(f"🎮 GPU: {gpu_info[0]}")
            print(f"🎮 Memory: {gpu_info[1]} MB")
            print(f"🎮 Compute Capability: {gpu_info[2]}")
            return True
        else:
            print("❌ Cannot query GPU")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False

def compile_cuda():
    """Compile the CUDA code"""
    print("\n🔨 Compiling CUDA ROM generator...")
    
    compile_cmd = [
        'nvcc',
        '-O3',                          # Maximum optimization
        '-arch=sm_61',                  # GTX 1070 compute capability
        '--use_fast_math',              # Fast math operations
        '-lcurand',                     # Link cuRAND library
        'cuda_rom_generator.cu',
        '-o', 'cuda_rom_generator'
    ]
    
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Compilation successful!")
            return True
        else:
            print("❌ Compilation failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False

def estimate_performance():
    """Estimate expected performance"""
    print("\n📊 Performance Estimates for GTX 1070:")
    print("  • CUDA Cores: 1,920")
    print("  • Base Clock: 1,506 MHz")
    print("  • Memory: 8GB GDDR5")
    print("  • Expected ROM generation: 500K - 2M ROMs/second")
    print("  • Memory usage per batch: ~4GB (1M ROMs × 4KB each)")
    print("  • Time to find promising ROM: seconds to minutes")

def run_generator():
    """Run the compiled generator"""
    print("\n🚀 Starting CUDA ROM generator...")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Create output directory
        os.makedirs("promising_roms", exist_ok=True)
        
        # Run the generator
        process = subprocess.Popen(['./cuda_rom_generator'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True,
                                 bufsize=1)
        
        start_time = time.time()
        
        try:
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping generator...")
            process.terminate()
            process.wait()
            
            elapsed = time.time() - start_time
            print(f"\n📊 Runtime: {elapsed:.1f} seconds")
            print("💾 Check 'promising_roms/' directory for results")
            
    except FileNotFoundError:
        print("❌ Compiled binary not found. Run compilation first.")
        return False
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        return False

def create_cuda_file():
    """Create the CUDA source file if it doesn't exist"""
    if not os.path.exists('cuda_rom_generator.cu'):
        print("❌ cuda_rom_generator.cu not found!")
        print("Please save the CUDA code to 'cuda_rom_generator.cu'")
        return False
    return True

def main():
    print("🕹️  CUDA Atari ROM Generator Setup")
    print("=" * 50)
    
    # Check if CUDA file exists
    if not create_cuda_file():
        return
    
    # Check CUDA installation
    if not check_cuda():
        print("\n💡 Install CUDA Toolkit from: https://developer.nvidia.com/cuda-toolkit")
        return
    
    # Check GPU
    if not check_gpu():
        print("\n💡 Make sure NVIDIA drivers are installed and GPU is available")
        return
    
    # Estimate performance
    estimate_performance()
    
    # Compile
    if not compile_cuda():
        return
    
    # Ask user if they want to run
    print(f"\n🎯 Ready to generate ROMs!")
    response = input("Start generation? (y/n): ").lower().strip()
    
    if response.startswith('y'):
        run_generator()
    else:
        print("💾 Binary compiled successfully. Run './cuda_rom_generator' manually when ready.")

if __name__ == "__main__":
    main()