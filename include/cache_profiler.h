#pragma once

#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <x86intrin.h>
#include "logger.h"

class CacheProfiler {
private:
    mutable uint64_t _cache_misses = 0;
    mutable uint64_t _cache_accesses = 0;
    mutable uint64_t _cycles_spent = 0;
    mutable uint64_t _instructions = 0;
    
    // Cumulative statistics
    mutable uint64_t _total_calls = 0;
    mutable uint64_t _total_memory_accesses = 0;
    mutable uint64_t _total_cycles = 0;
    mutable uint64_t _total_time_ns = 0;
    mutable uint64_t _total_intersection_count = 0;
    mutable uint64_t _total_set1_elements = 0;
    mutable uint64_t _total_set2_elements = 0;
    mutable double _sum_access_times_set1 = 0.0;
    mutable double _sum_access_times_set2 = 0.0;
    mutable uint64_t _high_access_time_warnings = 0;
    
public:
    struct CacheStats {
        uint64_t cache_misses;
        uint64_t cache_accesses;
        uint64_t cycles;
        uint64_t instructions;
        double miss_rate() const {
            return cache_accesses > 0 ? (double)cache_misses / cache_accesses : 0.0;
        }
        double cycles_per_access() const {
            return cache_accesses > 0 ? (double)cycles / cache_accesses : 0.0;
        }
    };
    
    // Cumulative statistics structure
    struct CumulativeStats {
        uint64_t total_calls;
        uint64_t total_memory_accesses;
        uint64_t total_cycles;
        uint64_t total_time_ns;
        uint64_t total_intersection_count;
        uint64_t total_set1_elements;
        uint64_t total_set2_elements;
        double avg_access_time_set1;
        double avg_access_time_set2;
        uint64_t high_access_time_warnings;
        double avg_memory_access_time_ns() const {
            return total_memory_accesses > 0 ? (double)total_time_ns / total_memory_accesses : 0.0;
        }
        double avg_cycles_per_access() const {
            return total_memory_accesses > 0 ? (double)total_cycles / total_memory_accesses : 0.0;
        }
        double avg_intersection_rate() const {
            return total_set1_elements > 0 ? (double)total_intersection_count / total_set1_elements : 0.0;
        }
    };
    
    // Performance counter reading (requires proper permissions)
    inline uint64_t read_cycles() const {
        return __rdtsc();
    }
    
    // Start profiling
    CacheStats start_profile() const {
        return {0, 0, read_cycles(), 0};
    }
    
    // End profiling and return stats
    CacheStats end_profile(const CacheStats& start) const {
        uint64_t end_cycles = read_cycles();
        return {0, 0, end_cycles - start.cycles, 0};
    }
    
    // Memory access timing
    template<typename T>
    double time_memory_access(const T* data, size_t size) const {
        auto start = std::chrono::high_resolution_clock::now();
        
        volatile T sum = 0;
        for (size_t i = 0; i < size; ++i) {
            sum += data[i];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        // Prevent optimization
        (void)sum;
        
        return duration.count() / (double)size; // ns per access
    }
    
    // Add to cumulative statistics
    void add_to_cumulative_stats(uint64_t memory_accesses, uint64_t cycles, uint64_t time_ns,
                                uint64_t intersection_count, uint64_t set1_size, uint64_t set2_size,
                                double access_time_set1, double access_time_set2, bool high_access_warning) const {
        _total_calls++;
        _total_memory_accesses += memory_accesses;
        _total_cycles += cycles;
        _total_time_ns += time_ns;
        _total_intersection_count += intersection_count;
        _total_set1_elements += set1_size;
        _total_set2_elements += set2_size;
        _sum_access_times_set1 += access_time_set1;
        _sum_access_times_set2 += access_time_set2;
        if (high_access_warning) {
            _high_access_time_warnings++;
        }
    }
    
    // Get cumulative statistics
    CumulativeStats get_cumulative_stats() const {
        return {
            _total_calls,
            _total_memory_accesses,
            _total_cycles,
            _total_time_ns,
            _total_intersection_count,
            _total_set1_elements,
            _total_set2_elements,
            _total_calls > 0 ? _sum_access_times_set1 / _total_calls : 0.0,
            _total_calls > 0 ? _sum_access_times_set2 / _total_calls : 0.0,
            _high_access_time_warnings
        };
    }
    
    // Reset cumulative statistics
    void reset_cumulative_stats() const {
        _total_calls = 0;
        _total_memory_accesses = 0;
        _total_cycles = 0;
        _total_time_ns = 0;
        _total_intersection_count = 0;
        _total_set1_elements = 0;
        _total_set2_elements = 0;
        _sum_access_times_set1 = 0.0;
        _sum_access_times_set2 = 0.0;
        _high_access_time_warnings = 0;
    }
    
    // Print cumulative statistics summary
    void print_cumulative_summary() const {
        auto stats = get_cumulative_stats();
        
        diskann::cout << "\n=== CUMULATIVE CACHE PERFORMANCE SUMMARY ===" << std::endl;
        diskann::cout << "Total Jaccard similarity calls: " << stats.total_calls << std::endl;
        diskann::cout << "Total memory accesses: " << stats.total_memory_accesses << std::endl;
        diskann::cout << "Total processing cycles: " << stats.total_cycles << std::endl;
        diskann::cout << "Total processing time: " << stats.total_time_ns << " ns" << std::endl;
        
        if (stats.total_calls > 0) {
            diskann::cout << "Average memory accesses per call: " << (double)stats.total_memory_accesses / stats.total_calls << std::endl;
            diskann::cout << "Average cycles per call: " << (double)stats.total_cycles / stats.total_calls << std::endl;
            diskann::cout << "Average time per call: " << (double)stats.total_time_ns / stats.total_calls << " ns" << std::endl;
        }
        
        diskann::cout << "Average memory access time: " << stats.avg_memory_access_time_ns() << " ns/access" << std::endl;
        diskann::cout << "Average cycles per memory access: " << stats.avg_cycles_per_access() << std::endl;
        
        diskann::cout << "Total intersections found: " << stats.total_intersection_count << std::endl;
        diskann::cout << "Total filter set elements processed: " << stats.total_set1_elements << std::endl;
        diskann::cout << "Total point label elements processed: " << stats.total_set2_elements << std::endl;
        diskann::cout << "Average intersection rate: " << (stats.avg_intersection_rate() * 100.0) << "%" << std::endl;
        
        diskann::cout << "Average access time for filter sets: " << stats.avg_access_time_set1 << " ns/element" << std::endl;
        diskann::cout << "Average access time for point labels: " << stats.avg_access_time_set2 << " ns/element" << std::endl;
        
        if (stats.high_access_time_warnings > 0) {
            diskann::cout << "High access time warnings: " << stats.high_access_time_warnings << std::endl;
            diskann::cout << "WARNING: " << (100.0 * stats.high_access_time_warnings / stats.total_calls) 
                         << "% of calls had high access times indicating cache misses" << std::endl;
        } else {
            diskann::cout << "Cache performance: GOOD (no high access time warnings)" << std::endl;
        }
        
        diskann::cout << "=============================================" << std::endl << std::endl;
    }
    
    // Estimate cache misses based on timing
    template<typename T>
    void estimate_cache_behavior(const T* data, size_t size, const char* label) const {
        const size_t L1_SIZE = 32 * 1024;  // 32KB L1 cache
        const size_t L2_SIZE = 256 * 1024; // 256KB L2 cache
        const size_t L3_SIZE = 8 * 1024 * 1024; // 8MB L3 cache
        
        double access_time = time_memory_access(data, size);
        size_t data_size = size * sizeof(T);
        
        diskann::cout << "=== Cache Analysis for " << label << " ===" << std::endl;
        diskann::cout << "Data size: " << data_size << " bytes (" << size << " elements)" << std::endl;
        diskann::cout << "Average access time: " << access_time << " ns/element" << std::endl;
        
        if (data_size <= L1_SIZE) {
            diskann::cout << "Expected: L1 cache hit (~1-2 ns/access)" << std::endl;
        } else if (data_size <= L2_SIZE) {
            diskann::cout << "Expected: L2 cache hit (~3-5 ns/access)" << std::endl;
        } else if (data_size <= L3_SIZE) {
            diskann::cout << "Expected: L3 cache hit (~10-15 ns/access)" << std::endl;
        } else {
            diskann::cout << "Expected: Memory access (~50-100+ ns/access)" << std::endl;
        }
        
        if (access_time > 20) {
            diskann::cout << "WARNING: High access time suggests cache misses!" << std::endl;
        }
        diskann::cout << "=============================" << std::endl << std::endl;
    }
};

// Global profiler instance
extern CacheProfiler g_cache_profiler;
