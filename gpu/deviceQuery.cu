// CUDA Device Query

#include <stdio.h>

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
	printf("Name : %s\n", devProp.name);
	printf("Major revision number:         %d\n",  devProp.major);
	printf("Minor revision number:         %d\n",  devProp.minor);
	printf("Name:                          %s\n",  devProp.name);
	printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
	printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
	printf("Warp size:                     %d\n",  devProp.warpSize);
	printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
	printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %u\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %u\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n",  devProp.clockRate);
	printf("Total constant memory:         %u\n",  devProp.totalConstMem);
	printf("Texture alignment:             %u\n",  devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}

int main()
{
	// Number of CUDA devices
	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("CUDA Device Query...\n");
	printf("There are %d CUDA devices.\n", devCount);

	// Iterate through devices
	devCount = 4;
	for (int i = 0; i < devCount; ++i)
	{
		cudaSetDevice(i);
		// Get device properties
		printf("\nCUDA Device #%d\n", i);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		printDevProp(devProp);
	}

	printf("\nPress any key to exit...");
	char c;
	scanf("%c", &c);

	return 0;
}
