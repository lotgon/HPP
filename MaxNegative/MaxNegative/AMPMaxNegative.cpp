#include "stdafx.h"
#include "AMPMaxNegative.h"
#include <iostream>
#include <fstream>

#define TILE_SIZE 64


CAMPMaxNegative::CAMPMaxNegative()
{
}


CAMPMaxNegative::~CAMPMaxNegative()
{
}
concurrency::completion_future CAMPMaxNegative::CalculateDrawdownSeq(concurrency::array_view<float, 1> prices, float currTp, concurrency::array_view<float, 1> &CalculateDrawdown, concurrency::array_view<int, 1> &duration)
{
	int N = prices.extent[0];
	CalculateDrawdown.discard_data();
	duration.discard_data();
	parallel_for_each(prices.extent, [=](concurrency::index<1> t_idx) restrict(amp)
	{
		int absTx = t_idx[0];

		if (absTx >= N)
			return;

		float threshold = prices[absTx] + currTp;
		float openPrice = prices[absTx];
		float drawdown = 0;
		int step;
		for (step = absTx; step < N; step++)
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
	);

	concurrency::completion_future complEvent = duration.synchronize_async();
	return complEvent;

}
concurrency::completion_future CAMPMaxNegative::CalculateDrawdownTile(concurrency::array_view<float, 1> P, float tp, concurrency::array_view<float, 1> &D, concurrency::array_view<int, 1> &duration)
{
	int sz = P.extent[0];
	D.discard_data();
	duration.discard_data();
	concurrency::tiled_extent<TILE_SIZE> compute_domain = D.get_extent().tile<TILE_SIZE>().pad();

	//std::wcout << "Starting GPU Calc"<<std::endl;
	parallel_for_each(compute_domain, [=](concurrency::tiled_index<TILE_SIZE> t_idx) restrict(amp)
	{

		float threshold = P[t_idx.global] + tp;
		float maxDrawdown = 0;
		float open = P[t_idx.global];

		tile_static float my_tile[TILE_SIZE];
		tile_static int calculatedCount;

		int stride = 0;
		bool isPointCalculated = false;

		calculatedCount = 0;
		t_idx.barrier.wait();

		for (; calculatedCount < TILE_SIZE; stride++)
		{
			if (t_idx.global[0] + stride*TILE_SIZE < sz)
				my_tile[t_idx.local[0]] = P[t_idx.global + stride*TILE_SIZE];
			else
				my_tile[t_idx.local[0]] = 0;

			t_idx.barrier.wait();

			if (!isPointCalculated)
			{
				for (int i = stride == 0 ? t_idx.local[0] : 0; i < TILE_SIZE; i++)
				{
					if (stride*TILE_SIZE + i + (t_idx.tile[0]) * TILE_SIZE >= sz)
					{
						isPointCalculated = true;
						concurrency::atomic_fetch_inc(&calculatedCount);
						D[t_idx.global] = maxDrawdown;
						duration[t_idx.global] = -1;
						break;
					}

					if (maxDrawdown < open - my_tile[i])
						maxDrawdown = open - my_tile[i];
					if (my_tile[i] >= threshold)
					{
						duration[t_idx.global] = stride*TILE_SIZE + i - t_idx.local[0];
						D[t_idx.global] = maxDrawdown;
						isPointCalculated = true;
						concurrency::atomic_fetch_inc(&calculatedCount);
						break;
					}
				}
			}
			t_idx.barrier.wait();

		}
	});
	D.synchronize();
	duration.synchronize();

	concurrency::completion_future complEvent = duration.synchronize_async();
	//std::wcout << "Finish GPU Calc" << std::endl;
	return complEvent;
}
void CAMPMaxNegative::CalculateAll(std::ofstream *outputFile, std::vector<float> &prices, std::vector<std::string> &vectorDates, float startTp, float endTp, float stepTp, bool isTileMode=false)
{
	int barCount = prices.size();

	concurrency::array_view<float, 1> P(barCount, &prices[0]);
	
	for (float currTp = startTp; currTp <= endTp;currTp+=stepTp)
	{
		std::vector<float> vectorDrawdown(barCount);
		std::vector<int> vectorDuration(barCount);
		concurrency::array_view<float, 1> D(vectorDrawdown);
		concurrency::array_view<int, 1> duration(vectorDuration);

		if ( isTileMode)
			CalculateDrawdownTile(P, currTp, D, duration).get();
		else
			CalculateDrawdownSeq(P, currTp, D, duration).get();

		if ( outputFile != NULL)
			for (int i = 0; i < barCount; i++)
				*outputFile << vectorDates[i] << ", " << currTp<<", "<< vectorDrawdown[i] << ", " << vectorDuration[i] << "\n";

	}

}

