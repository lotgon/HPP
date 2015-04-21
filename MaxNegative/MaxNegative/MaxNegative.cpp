// MaxNegative.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include "Csv.h"
#include <vector>
#include <amp.h>
#include "AMPHelper.h"
#include <chrono>
#include <ppltasks.h>
#include <array>
#define TILE_SIZE 256

template<typename TimeT = std::chrono::microseconds,
	typename ClockT = std::chrono::high_resolution_clock,
	typename DurationT = double>
class Stopwatch
{
private:
	std::chrono::time_point<ClockT> _start, _end;
public:
	Stopwatch() { start(); }
	void start() { _start = _end = ClockT::now(); }
	DurationT stop() { _end = ClockT::now(); return elapsed(); }
	DurationT elapsed() {
		auto delta = std::chrono::duration_cast<TimeT>(_end - _start);
		return delta.count();
	}
};


concurrency::completion_future CalculateDrawdown(concurrency::array_view<float, 1> P, float tp, concurrency::array_view<float, 1> &D, concurrency::array_view<int, 1> &duration)
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
						duration[t_idx.global] = 0;
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
	concurrency::completion_future complEvent =  D.synchronize_async();
	//std::wcout << "Finish GPU Calc" << std::endl;
	return complEvent;
}
void CalculateAll(std::vector<float> &prices, std::vector<std::string> &vectorDates)
{
	int barCount = prices.size();
	std::ofstream outputFile("..//Data//silverDayResult.csv");
	outputFile << "Date, Tp, Drawdown,BarDuration\n";

	concurrency::array_view<float, 1> P(barCount, &prices[0]);
	concurrency::array_view<float, 1> P2(barCount, &prices[0]);
	concurrency::array_view<float, 1> P3(barCount, &prices[0]);

	int number = 10;

	std::vector<concurrency::completion_future> arSync;
	int i = 0;
	float currTp = 0.5, maxTp = 2., stepTp = 10.01;
	float sum = 0;
	do
	{
		std::vector<float> vectorDrawdown(barCount);
		std::vector<int> vectorDuration(barCount);
		concurrency::array_view<float, 1> D(vectorDrawdown);
		concurrency::array_view<int, 1> duration(vectorDuration);

		std::vector<float> vectorDrawdown2(barCount);
		std::vector<int> vectorDuration2(barCount);
		concurrency::array_view<float, 1> D2(vectorDrawdown);
		concurrency::array_view<int, 1> duration2(vectorDuration);

		std::vector<float> vectorDrawdown3(barCount);
		std::vector<int> vectorDuration3(barCount);
		concurrency::array_view<float, 1> D3(vectorDrawdown);
		concurrency::array_view<int, 1> duration3(vectorDuration);

		Stopwatch<> sw0;
		sw0.start();
		arSync.push_back(CalculateDrawdown(P, currTp, D, duration));
		arSync.push_back(CalculateDrawdown(P2, currTp+0.01, D2, duration2));
		arSync.push_back(CalculateDrawdown(P3, currTp + 0.01, D3, duration3));

		//for (int i = 0; i < barCount; i++)
		//outputFile << vectorDates[i] << ", " << currTp<<", "<< vectorDrawdown[i] << ", " << vectorDuration[i] << "\n";
		sw0.stop();
		for (const auto &elem : arSync)
			elem.get();
		std::wcout << " Iteration Execution time is " << sw0.elapsed() / 1000 << " milliseconds";

		currTp += stepTp;
	} while (currTp < maxTp);

}

int _tmain(int argc, _TCHAR* argv[])
{
	CAMPHelper::default_properties();
	CAMPHelper::list_all_accelerators();
	std::ifstream ss("..//Data//silverDay.csv");

	bool result;
	result = true; //CAMPHelper::PickEmulatedAccelerator();
	//std::wstring ws = L"direct3d\\warp";//CPU accelerator";
	//std::wstring ws = L"cpu";
	//result = CAMPHelper::PickAccelerator(ws);
	if (!result)
	{
		std::wcout << "Cannot pick accelerator";
		return 0;
	}


	std::string title[3];
	std::string date1, price, date2;
	
	csv_istream(ss) >> title[0] >>title[1]>>title[2];
	std::vector<float> vectorPrices;
	std::vector<std::string> vectorDates;

	while (csv_istream(ss) >> date1 >> price >> date2)
	{
		vectorPrices.push_back(std::stof(price));
		vectorDates.push_back(date1);
	}
	

	Stopwatch<> sw0;
	sw0.start();
	
	CalculateAll(vectorPrices, vectorDates);

	sw0.stop();
	std::wcout << " Execution time is " << sw0.elapsed()/1000 << " milliseconds";

	std::wcin.get();

	return 0;
}

