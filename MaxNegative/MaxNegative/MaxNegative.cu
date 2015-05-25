
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>


__global__ void testKernel(int val)
{
	printf("[%d]:\t\tValue is\n", blockIdx.x*blockDim.x + threadIdx.x);
}

void CudaCalculateAll(std::ofstream *outputFile, std::vector<float> &prices, std::vector<std::string> &vectorDates, float startTp, float endTp, float stepTp)
{
	dim3 dimGrid(5, 1, 1);
	dim3 dimBlock(3, 1, 1);

	testKernel<<<dimGrid, dimBlock >>>(1);
	cudaDeviceSynchronize();
}


