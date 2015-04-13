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

#define TILE_SIZE 128

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


void CalculateDrawdown(concurrency::array_view<float, 1> P, float tp, std::vector<float> &outDrawdown, std::vector<int> &outDuration)
{
	int sz = P.extent[0];
	outDrawdown.resize(sz);
	outDuration.resize(sz);


	concurrency::array_view<float, 1> D(sz, &outDrawdown[0]);
	concurrency::array_view<int, 1> duration(sz, &outDuration[0]);
	D.discard_data();
	duration.discard_data();

	parallel_for_each(D.extent.tile<TILE_SIZE>(), [=](concurrency::tiled_index<TILE_SIZE> t_idx) restrict(amp)
	{

		float threshold = P[t_idx.global];
		tile_static float my_tile[TILE_SIZE];
		tile_static int calculatedCount;

		for (int stride=0; calculatedCount < TILE_SIZE; stride++)
		{
			my_tile[t_idx.local[0]] = P[t_idx.global[0] + stride*TILE_SIZE];
			t_idx.barrier.wait();
			for (int i =0; i < TILE_SIZE; i++)
			{
				if (my_tile[i] >= threshold)
				{
					D[t_idx.global] = 0;
					calculatedCount++;
				}
			}
		}
	});
}
void CalculateAll(std::vector<float> &prices)
{
	int barCount = prices.size();
	std::ofstream outputFile("c:\\projects\\CPlusPlus\\MaxNegative\\Data\\silverDayResult.csv");
	outputFile << "Date, Tp, Drawdown,BarDuration\n";

	concurrency::array_view<float, 1> P(barCount, &prices[0]);

	float currTp = 0.5, maxTp = 2, stepTp = 0.0001;
	do
	{
		std::vector<float> vectorDrawdown;
		std::vector<int> vectorDuration;

		CalculateDrawdown(P, currTp, vectorDrawdown, vectorDuration);
		//for (int i = 0; i < barCount; i++)
		//outputFile << vectorDates[i] << ", " << currTp<<", "<< vectorDrawdown[i] << ", " << vectorDuration[i] << "\n";

		currTp += stepTp;
	} while (currTp < maxTp);
}

int _tmain(int argc, _TCHAR* argv[])
{
	CAMPHelper::default_properties();
	CAMPHelper::list_all_accelerators();
	std::ifstream ss("c:\\projects\\CPlusPlus\\MaxNegative\\Data\\silverDay.csv");

	bool result;
	result = true;// CAMPHelper::PickEmulatedAccelerator();
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
	
	CalculateAll(vectorPrices);

	sw0.stop();
	std::wcout << " Execution time is " << sw0.elapsed()/1000 << " milliseconds";

	std::wcin.get();

	return 0;
}

