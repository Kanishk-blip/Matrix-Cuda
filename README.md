# CUDA Tiled Matrix Multiplication

## Overview
This repository showcases an optimized implementation of Matrix Multiplication on NVIDIA GPUs using CUDA. It demonstrates the transition from a naive global-memory approach to a highly optimized version utilizing shared memory, directly addressing the I/O and memory bandwidth bottlenecks common in data-intensive and parallel computing workloads.

## Key Performance Engineering Concepts Demonstrated
* **Memory Tiling:** Divides the global computation matrix into smaller, manageable tiles to significantly enhance memory locality.
* **Shared Memory (SRAM) Utilization:** Loads blocks of data into the GPU's ultra-fast, on-chip shared memory to minimize repetitive round-trips to the slower High Bandwidth Memory (HBM).
* **Memory Coalescing:** Aligns data fetches to ensure efficient memory access patterns for threads within a warp, maximizing memory bus utilization.
* **Thread Synchronization:** Implements `__syncthreads()` to ensure all threads within a block finish loading their respective data tiles into shared memory before mathematical computations begin, preventing race conditions.

## Implementation Structure
This project features multiple implementations for performance benchmarking and analysis:
1. **CPU Baseline:** A sequential matrix multiplication executed on the host for correctness verification and baseline latency measurement.
2. **Naive GPU Kernel:** A basic CUDA kernel where each thread computes a single output element, relying entirely on unoptimized global memory accesses.
3. **Optimized Tiled Kernel:** The core focus of this project, utilizing `__shared__` memory arrays to cache data inside the Streaming Multiprocessor (SM) for maximum reuse.

## Prerequisites
* An NVIDIA GPU with CUDA Compute Capability.
* NVIDIA CUDA Toolkit installed.
* C++ compiler (GCC/MSVC).

## How to Build and Run
Compile the code using the `nvcc` compiler. You can optionally include the architecture flag specific to your GPU (for example, `-arch=sm_80` for Ampere architectures) to ensure maximum compatibility.

`nvcc -o matrix_multiplication kernel.cu`

Run the compiled executable:

`./matrix_multiplication`

## Profiling and Benchmarking
The project utilizes CUDA Events to measure kernel execution time. To perform a deeper analysis of the performance differences and verify that the tiled implementation successfully transforms the workload from memory-bound to compute-bound, you can profile the executable using NVIDIA Nsight Systems or Nsight Compute.
