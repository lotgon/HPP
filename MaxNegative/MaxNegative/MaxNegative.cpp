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
#include "AMPMaxNegative.h"
#include "CPUMaxNegative.h"

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

extern void CudaCalculateAll(std::ofstream *outputFile, std::vector<float> &prices, std::vector<std::string> &vectorDates, float startTp, float endTp, float stepTp);


int _tmain(int argc, _TCHAR* argv[])
{
	CAMPHelper::default_properties();
	CAMPHelper::list_all_accelerators();
	std::ifstream ss("..//Data//silverDay2.csv");

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
	
	float startTp = 0.5, endTp = 2., stepTp = 2.1;
	bool IsWriteResult = false;

	///////////
	//AMP
	//////////
	std::ofstream outputFile("..//Data//silverDayResult_AMP.csv");
	outputFile << "Date, Tp, Drawdown,BarDuration\n";
	Stopwatch<> sw0;
	sw0.start();
	
	CAMPMaxNegative::CalculateAll(IsWriteResult? &outputFile : NULL, vectorPrices, vectorDates, startTp, endTp, stepTp);

	sw0.stop();
	std::wcout << "CAMP Execution time is " << sw0.elapsed()/1000 << " milliseconds";

	/////////////
	//CPU
	/////////////
	std::ofstream outputFileCPU("..//Data//silverDayResult_CPU.csv");
	outputFileCPU << "Date, Tp, Drawdown,BarDuration\n";

	Stopwatch<> sw1;
	sw1.start();

	CCPUMaxNegative::CalculateAll(IsWriteResult ? &outputFileCPU : NULL, vectorPrices, vectorDates, startTp, endTp, stepTp);

	sw1.stop();
	std::wcout << "CPU Execution time is " << sw1.elapsed() / 1000 << " milliseconds";

	/////////////
	//CUDA
	/////////////
	std::ofstream outputFileCuda("..//Data//silverDayResult_Cuda.csv");
	outputFileCuda << "Date, Tp, Drawdown,BarDuration\n";

	sw1.start();

	CudaCalculateAll(IsWriteResult ? &outputFileCuda : NULL, vectorPrices, vectorDates, startTp, endTp, stepTp);

	sw1.stop();
	std::wcout << "CPU Execution time is " << sw1.elapsed() / 1000 << " milliseconds";

	std::wcin.get();

	return 0;
}

