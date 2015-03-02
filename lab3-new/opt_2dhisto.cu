#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"



void* allocCudaMem(size_t size) {
	void *ptr;
	cudaMalloc(&ptr, size);
	return ptr;
}

void freeCudaMem(void *ptr) {
	cudaFree(ptr);
}

void copyToDevice(void *src, void *dst, size_t size) {
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
}

void copyFromDevice(void *src, void *dst, size_t size) {
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
}

__global__ void clear_bins(uint32_t *bins) {
	const int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y
		+ blockIdx.y * gridDim.x * blockDim.x * blockDim.y,
	    tsz = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
#pragma unroll
	for(int i = idx; i < HISTO_WIDTH * HISTO_HEIGHT; i += tsz) {
		bins[i] = 0;
	}
}

__global__ void do_histogram(uint32_t *input, size_t height, size_t width, uint32_t *bins) {
	const int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y
		+ blockIdx.y * gridDim.x * blockDim.x * blockDim.y,
	    bsz = blockDim.x * blockDim.y, sz = height * width,
	    tsz = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	const int hsz = HISTO_WIDTH * HISTO_HEIGHT;
	__shared__ uint32_t histo[hsz][4];

#pragma unroll
	for(int i = threadIdx.x + threadIdx.y * blockDim.x; i < hsz * 4; i += bsz) {
		((uint32_t*)histo)[i] = 0;
	}

	__syncthreads();

#pragma unroll
	for(int i = idx; i < sz; i += tsz) {
		//const uint32_t value = input[i];
		atomicAdd(&histo[input[i]][threadIdx.x % 4], 1);
		//atomicInc(&histo[input[i]][threadIdx.x % 4], 1 << 30);
	}

	__syncthreads();

#pragma unroll
	for(int i = threadIdx.x + threadIdx.y * blockDim.x; i < hsz; i += bsz) {
		uint32_t sum = histo[i][0] + histo[i][1] + histo[i][2] + histo[i][3];
		atomicAdd(&bins[i], sum);
	}
}

__global__ void copy_bins(uint32_t *bins32, uint8_t *bins8) {
	const int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y
		+ blockIdx.y * gridDim.x * blockDim.x * blockDim.y,
	    tsz = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
#pragma unroll
	for(int i = idx; i < HISTO_WIDTH * HISTO_HEIGHT; i += tsz) {
		bins8[i] = bins32[i] > UINT8_MAX ? UINT8_MAX : bins32[i];
	}
}

void opt_2dhisto(uint32_t *input, size_t height, size_t width, uint8_t *bins8, uint32_t *bins32)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    clear_bins<<<dim3(1, 1), dim3(32, 32)>>>(bins32);
    do_histogram<<<dim3(4,4), dim3(32, 32)>>>(input, height, width, bins32);
    copy_bins<<<dim3(1, 1), dim3(32, 32)>>>(bins32, bins8);
    cudaThreadSynchronize();
}

/* Include below the implementation of any other functions you need */

