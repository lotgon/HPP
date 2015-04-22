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
	
	std::ofstream outputFile("..//Data//silverDayResult.csv");
	outputFile << "Date, Tp, Drawdown,BarDuration\n";

	float startTp = 0.5, endTp = 2., stepTp = 10.01;

	Stopwatch<> sw0;
	sw0.start();
	
	CAMPMaxNegative::CalculateAll(&outputFile, vectorPrices, vectorDates, startTp, endTp, stepTp);

	sw0.stop();
	std::wcout << " Execution time is " << sw0.elapsed()/1000 << " milliseconds";

	std::wcin.get();

	return 0;
}

