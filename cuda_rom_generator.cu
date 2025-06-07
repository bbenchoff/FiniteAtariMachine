// cuda_rom_generator.cu
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <fstream>

// Constants from your Python code
#define ROM_SIZE 4094
#define OPCODE_THRESHOLD 0.66f
#define TIA_THRESHOLD 51
#define RIOT_THRESHOLD 1
#define BRANCH_THRESHOLD 177
#define JUMP_THRESHOLD 37
#define INSTRUCTION_VARIETY 125
#define MIN_SCORE 0.80f

// Valid opcodes lookup table (151 valid opcodes)
__constant__ bool valid_opcodes[256];
__constant__ bool tia_registers[256];
__constant__ bool branch_opcodes[256];
__constant__ bool jump_opcodes[256];
__constant__ int riot_registers[12] = {0x280, 0x281, 0x282, 0x283, 0x284, 0x285, 0x286, 0x287, 0x294, 0x295, 0x296, 0x297};

struct HeuristicResult {
    float valid_opcodes_ratio;
    int tia_accesses;
    int riot_accesses;
    int branch_instructions;
    int jump_instructions;
    int unique_opcodes;
    bool has_loop_patterns;
    bool has_interrupt_vectors;
    float score;
    bool is_promising;
};

__device__ bool is_riot_register(int addr) {
    for(int i = 0; i < 12; i++) {
        if(addr == riot_registers[i]) return true;
    }
    return false;
}

__device__ HeuristicResult analyze_rom(unsigned char* rom) {
    HeuristicResult result = {0};
    
    // Count valid opcodes
    int valid_count = 0;
    for(int i = 0; i < ROM_SIZE; i++) {
        if(valid_opcodes[rom[i]]) valid_count++;
    }
    result.valid_opcodes_ratio = (float)valid_count / ROM_SIZE;
    
    // Count TIA accesses
    for(int i = 0; i < ROM_SIZE - 2; i++) {
        // STA absolute (0x8D)
        if(rom[i] == 0x8D) {
            int addr = rom[i+1] | (rom[i+2] << 8);
            if(tia_registers[addr & 0xFF]) {
                result.tia_accesses++;
            }
        }
        // STA zero page (0x85)
        else if(rom[i] == 0x85 && i + 1 < ROM_SIZE) {
            if(tia_registers[rom[i+1]]) {
                result.tia_accesses++;
            }
        }
    }
    
    // Count RIOT accesses
    for(int i = 0; i < ROM_SIZE - 2; i++) {
        if(rom[i] == 0x8D) {
            int addr = rom[i+1] | (rom[i+2] << 8);
            if(is_riot_register(addr)) {
                result.riot_accesses++;
            }
        }
    }
    
    // Count branches and jumps
    for(int i = 0; i < ROM_SIZE; i++) {
        if(branch_opcodes[rom[i]]) result.branch_instructions++;
        if(jump_opcodes[rom[i]]) result.jump_instructions++;
    }
    
    // Count unique opcodes
    bool used_opcodes[256] = {false};
    for(int i = 0; i < ROM_SIZE; i++) {
        if(valid_opcodes[rom[i]]) {
            used_opcodes[rom[i]] = true;
        }
    }
    for(int i = 0; i < 256; i++) {
        if(used_opcodes[i]) result.unique_opcodes++;
    }
    
    // Check for loop patterns (backward branches)
    for(int i = 0; i < ROM_SIZE - 1; i++) {
        if(branch_opcodes[rom[i]]) {
            signed char offset = (signed char)rom[i+1];
            if(offset < 0) {
                result.has_loop_patterns = true;
                break;
            }
        }
    }
    
    // Check interrupt vectors (simplified)
    if(ROM_SIZE >= 16) {
        int high_bytes = 0;
        for(int i = ROM_SIZE - 16; i < ROM_SIZE; i += 2) {
            if(i + 1 < ROM_SIZE && rom[i+1] >= 0xF0) {
                high_bytes++;
            }
        }
        result.has_interrupt_vectors = (high_bytes >= 3);
    }
    
    // Calculate composite score
    result.score = result.valid_opcodes_ratio * 0.25f +
                   fminf(result.tia_accesses / 20.0f, 1.0f) * 0.20f +
                   fminf(result.riot_accesses / 10.0f, 1.0f) * 0.15f +
                   fminf(result.branch_instructions / 15.0f, 1.0f) * 0.15f +
                   fminf(result.jump_instructions / 8.0f, 1.0f) * 0.10f +
                   fminf(result.unique_opcodes / 30.0f, 1.0f) * 0.10f +
                   (result.has_loop_patterns ? 0.05f : 0.0f) +
                   (result.has_interrupt_vectors ? 0.05f : 0.0f);
    
    // Check if promising
    result.is_promising = (result.valid_opcodes_ratio >= OPCODE_THRESHOLD &&
                          result.tia_accesses >= TIA_THRESHOLD &&
                          result.riot_accesses >= RIOT_THRESHOLD &&
                          result.branch_instructions >= BRANCH_THRESHOLD &&
                          result.jump_instructions >= JUMP_THRESHOLD &&
                          result.unique_opcodes >= INSTRUCTION_VARIETY &&
                          result.score >= MIN_SCORE);
    
    return result;
}

__global__ void generate_and_analyze_roms(unsigned char* rom_data, 
                                         HeuristicResult* results,
                                         unsigned long long seed,
                                         int num_roms) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_roms) return;
    
    // Initialize random state for this thread
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    // Generate ROM data
    unsigned char* rom = &rom_data[idx * ROM_SIZE];
    for(int i = 0; i < ROM_SIZE; i++) {
        rom[i] = curand(&state) & 0xFF;
    }
    
    // Analyze the ROM
    results[idx] = analyze_rom(rom);
}

void initialize_lookup_tables() {
    // Valid opcodes (151 of them)
    int valid_opcodes_host[256] = {0};
    int valid_list[] = {
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
    };
    for(int i = 0; i < 151; i++) {
        valid_opcodes_host[valid_list[i]] = 1;
    }
    cudaMemcpyToSymbol(valid_opcodes, valid_opcodes_host, sizeof(valid_opcodes_host));
    
    // TIA registers
    int tia_registers_host[256] = {0};
    for(int i = 0x00; i <= 0x2F; i++) {
        tia_registers_host[i] = 1;
    }
    cudaMemcpyToSymbol(tia_registers, tia_registers_host, sizeof(tia_registers_host));
    
    // Branch opcodes
    int branch_opcodes_host[256] = {0};
    int branch_list[] = {0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0};
    for(int i = 0; i < 8; i++) {
        branch_opcodes_host[branch_list[i]] = 1;
    }
    cudaMemcpyToSymbol(branch_opcodes, branch_opcodes_host, sizeof(branch_opcodes_host));
    
    // Jump opcodes
    int jump_opcodes_host[256] = {0};
    int jump_list[] = {0x4C, 0x6C, 0x20};
    for(int i = 0; i < 3; i++) {
        jump_opcodes_host[jump_list[i]] = 1;
    }
    cudaMemcpyToSymbol(jump_opcodes, jump_opcodes_host, sizeof(jump_opcodes_host));
}

void save_promising_rom(const unsigned char* rom_data, const HeuristicResult& result, 
                       int rom_id, const std::string& output_dir) {
    // Create filename
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/base_%06d_score_%.3f_gpu.bin", 
             output_dir.c_str(), rom_id, result.score);
    
    // Save ROM data
    std::ofstream rom_file(filename, std::ios::binary);
    rom_file.write((const char*)rom_data, ROM_SIZE);
    rom_file.close();
    
    // Save metadata
    char meta_filename[256];
    snprintf(meta_filename, sizeof(meta_filename), "%s/base_%06d_score_%.3f_gpu.txt", 
             output_dir.c_str(), rom_id, result.score);
    
    std::ofstream meta_file(meta_filename);
    meta_file << "ROM Base Analysis\n";
    meta_file << "=================\n";
    meta_file << "Generated on GPU\n";
    meta_file << "Overall Score: " << result.score << "\n";
    meta_file << "Valid Opcodes: " << result.valid_opcodes_ratio << " (" << (result.valid_opcodes_ratio*100) << "%)\n";
    meta_file << "TIA Accesses: " << result.tia_accesses << "\n";
    meta_file << "RIOT Accesses: " << result.riot_accesses << "\n";
    meta_file << "Branch Instructions: " << result.branch_instructions << "\n";
    meta_file << "Jump Instructions: " << result.jump_instructions << "\n";
    meta_file << "Unique Opcodes: " << result.unique_opcodes << "\n";
    meta_file << "Has Loop Patterns: " << (result.has_loop_patterns ? "true" : "false") << "\n";
    meta_file << "Has Interrupt Vectors: " << (result.has_interrupt_vectors ? "true" : "false") << "\n";
    meta_file.close();
    
    printf("💎 SAVED: base_%06d_score_%.3f_gpu.bin\n", rom_id, result.score);
    printf("    Score: %.4f | Opcodes: %.3f | TIA: %d | RIOT: %d\n",
           result.score, result.valid_opcodes_ratio, result.tia_accesses, result.riot_accesses);
}

int main() {
    const int BATCH_SIZE = 1024 * 1024;  // 1M ROMs per batch
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const std::string output_dir = "promising_roms";
    
    // Create output directory
    system(("mkdir -p " + output_dir).c_str());
    
    printf("🕹️  Finite Atari Machine - CUDA EDITION\n");
    printf("=" * 80);
    printf("\nGenerating %d ROMs per batch on GPU\n", BATCH_SIZE);
    printf("Grid size: %d blocks of %d threads each\n", GRID_SIZE, BLOCK_SIZE);
    printf("Saving promising ROM bases to: %s\n", output_dir.c_str());
    printf("🟢 EASY Thresholds (5th percentile - bottom 5%% of real games)\n");
    printf("Press Ctrl+C to stop\n");
    printf("=" * 80);
    
    // Initialize lookup tables
    initialize_lookup_tables();
    
    // Allocate device memory
    unsigned char* d_rom_data;
    HeuristicResult* d_results;
    
    cudaMalloc(&d_rom_data, BATCH_SIZE * ROM_SIZE);
    cudaMalloc(&d_results, BATCH_SIZE * sizeof(HeuristicResult));
    
    // Allocate host memory for results
    HeuristicResult* h_results = new HeuristicResult[BATCH_SIZE];
    unsigned char* h_rom_data = new unsigned char[BATCH_SIZE * ROM_SIZE];
    
    int total_generated = 0;
    int promising_found = 0;
    time_t start_time = time(NULL);
    time_t last_report = start_time;
    
    while(true) {
        // Generate random seed
        unsigned long long seed = time(NULL) + total_generated;
        
        // Launch kernel
        auto kernel_start = std::chrono::high_resolution_clock::now();
        generate_and_analyze_roms<<<GRID_SIZE, BLOCK_SIZE>>>(d_rom_data, d_results, seed, BATCH_SIZE);
        cudaDeviceSynchronize();
        auto kernel_end = std::chrono::high_resolution_clock::now();
        
        // Copy results back to host
        cudaMemcpy(h_results, d_results, BATCH_SIZE * sizeof(HeuristicResult), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rom_data, d_rom_data, BATCH_SIZE * ROM_SIZE, cudaMemcpyDeviceToHost);
        
        // Process results
        for(int i = 0; i < BATCH_SIZE; i++) {
            if(h_results[i].is_promising) {
                save_promising_rom(&h_rom_data[i * ROM_SIZE], h_results[i], promising_found, output_dir);
                promising_found++;
            }
        }
        
        total_generated += BATCH_SIZE;
        
        // Print stats every 5 seconds
        time_t current_time = time(NULL);
        if(current_time - last_report >= 5) {
            double elapsed = difftime(current_time, start_time);
            double rate = total_generated / elapsed;
            double success_rate = (double)promising_found / total_generated * 100.0;
            
            auto kernel_time = std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end - kernel_start);
            
            printf("\r🔍 Generated: %d | Promising: %d | Success: %.6f%% | Rate: %.0f/sec | Kernel: %ldms",
                   total_generated, promising_found, success_rate, rate, kernel_time.count());
            fflush(stdout);
            
            last_report = current_time;
        }
    }
    
    // Cleanup
    cudaFree(d_rom_data);
    cudaFree(d_results);
    delete[] h_results;
    delete[] h_rom_data;
    
    return 0;
}