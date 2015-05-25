
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#define BLOCK_SIZE 128
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n",
			file, line, static_cast<unsigned int>(result), func);
		cudaDeviceReset();
			// Make sure we call CUDA Device Reset before exiting
		exit(EXIT_FAILURE);
	}
}

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )


__global__ void CalculateDrawdown(float *prices, float currTp, float *CalculateDrawdown, int *duration, int N)
{
	printf("[%d]:\t\tValue is\n", blockIdx.x*blockDim.x + threadIdx.x);
}

void CudaCalculateAll(std::ofstream *outputFile, std::vector<float> &prices, std::vector<std::string> &vectorDates, float startTp, float endTp, float stepTp)
{
	int barCount = prices.size();

	//concurrency::array_view<float, 1> P(barCount, &prices[0]);
	float *d_P;
	checkCudaErrors(cudaMalloc((void**)&d_P, barCount*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_P, &prices[0], barCount*sizeof(float), cudaMemcpyHostToDevice));
	

	for (float currTp = startTp; currTp <= endTp; currTp += stepTp)
	{
		std::vector<float> vectorDrawdown(barCount);
		std::vector<int> vectorDuration(barCount);
		/*concurrency::array_view<float, 1> D(vectorDrawdown);
		concurrency::array_view<int, 1> duration(vectorDuration);
		*/
		float *d_Drawdown;
		int *d_Duration;
		checkCudaErrors(cudaMalloc((void**)&d_Drawdown, barCount*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_Duration, barCount*sizeof(int)));

		dim3 dimGrid((barCount-1)/barCount + 1, 1, 1);
		dim3 dimBlock(BLOCK_SIZE, 1, 1);
		CalculateDrawdown<<<dimGrid, dimBlock>>>(d_P, currTp, d_Drawdown, d_Duration, barCount);

		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(&vectorDrawdown[0], d_Drawdown, barCount*sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&vectorDuration[0], d_Duration, barCount*sizeof(int), cudaMemcpyDeviceToHost));

		if (outputFile != NULL)
			for (int i = 0; i < barCount; i++)
				*outputFile << vectorDates[i] << ", " << currTp << ", " << vectorDrawdown[i] << ", " << vectorDuration[i] << "\n";

	}


}


