#ifndef OPT_KERNEL
#define OPT_KERNEL

void* allocCudaMem(size_t size);

void freeCudaMem(void *ptr);

void copyToDevice(void *src, void *dst, size_t size);

void copyFromDevice(void *src, void *dst, size_t size);

void opt_2dhisto(uint32_t *input, size_t height, size_t width, uint8_t *bins8, uint32_t *bins32);

/* Include below the function headers of any other functions that you implement */


#endif
