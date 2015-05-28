
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#define BLOCK_SIZE 256
#define TILE_SIZE 256
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


__global__ void CalculateDrawdownSeq(float *prices, float currTp, float *CalculateDrawdown, int *duration, int N)
{ 
	int absTx = blockIdx.x*blockDim.x + threadIdx.x;
	//printf("Result writing for %d", absTx);
	if (absTx >= N)
		return;

	float threshold = prices[absTx] + currTp;
	float openPrice = prices[absTx];
	float drawdown = 0;
	int step;
	for (step = absTx; step < N ; step++)
	{
		if (openPrice - prices[step] > drawdown)
			drawdown = openPrice - prices[step];

		if (prices[step] >= threshold)
			break;
	}
	CalculateDrawdown[absTx] = drawdown;
	if (step < N)
		duration[absTx] = step - absTx;
	else
		duration[absTx] = -1;
}
__global__ void CalculateDrawdownTile(float *prices, float currTp, float *CalculateDrawdown, int *duration, int N)
{
	int absTx = blockIdx.x*blockDim.x + threadIdx.x;
	int tx = threadIdx.x;

	float maxDrawdown = 0;
	float open = prices[absTx];
	float threshold = open + currTp;

	__shared__ float my_tile[TILE_SIZE];
	__shared__ int calculatedCount;
	calculatedCount = 0;

	int stride = 0;
	bool isPointCalculated = false;
	__syncthreads();

	for (; calculatedCount < TILE_SIZE; stride++)
	{
		int strideTile = stride *TILE_SIZE;
		if (absTx + strideTile < N)
			my_tile[tx] = prices[absTx + strideTile];
		else
			my_tile[tx] = 0;

		__syncthreads();
		if (!isPointCalculated)
		{
			for (int i = stride == 0 ? tx : 0; i < TILE_SIZE; i++)
			{
				if (strideTile + i + blockIdx.x * blockDim.x >= N)
				{
					isPointCalculated = true;
					atomicAdd(&calculatedCount, 1);
					CalculateDrawdown[absTx] = maxDrawdown;
					duration[absTx] = -1;
					break;
				}

				if (maxDrawdown < open - my_tile[i])
					maxDrawdown = open - my_tile[i];
				if (my_tile[i] >= threshold)
				{
					duration[absTx] = strideTile + i - tx;
					CalculateDrawdown[absTx] = maxDrawdown;
					isPointCalculated = true;
					atomicAdd(&calculatedCount, 1);
					break;
				}
			}
		}
		__syncthreads();
	}
}
void CudaCalculateAll(std::ofstream *outputFile, std::vector<float> &prices, std::vector<std::string> &vectorDates, float startTp, float endTp, float stepTp)
{
	int barCount = prices.size();

	float *d_P;
	checkCudaErrors(cudaMalloc((void**)&d_P, barCount*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_P, &prices[0], barCount*sizeof(float), cudaMemcpyHostToDevice));
	

	for (float currTp = startTp; currTp <= endTp; currTp += stepTp)
	{
		std::vector<float> vectorDrawdown(barCount);
		std::vector<int> vectorDuration(barCount);
		float *d_Drawdown;
		int *d_Duration;
		checkCudaErrors(cudaMalloc((void**)&d_Drawdown, barCount*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_Duration, barCount*sizeof(int)));

		dim3 dimGrid((barCount-1)/BLOCK_SIZE + 1, 1, 1);
		dim3 dimBlock(BLOCK_SIZE, 1, 1);
		CalculateDrawdownTile<<<dimGrid, dimBlock>>>(d_P, currTp, d_Drawdown, d_Duration, barCount);

		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(&vectorDrawdown[0], d_Drawdown, barCount*sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&vectorDuration[0], d_Duration, barCount*sizeof(int), cudaMemcpyDeviceToHost));

		if (outputFile != NULL)
			for (int i = 0; i < barCount; i++)
				*outputFile << vectorDates[i] << ", " << currTp << ", " << vectorDrawdown[i] << ", " << vectorDuration[i] << "\n";

	}


}


