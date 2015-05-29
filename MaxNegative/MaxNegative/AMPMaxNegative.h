#pragma once
#include <amp.h>

static class CAMPMaxNegative
{
public:
	CAMPMaxNegative();
	~CAMPMaxNegative();

	static concurrency::completion_future CalculateDrawdownTile(concurrency::array_view<float, 1> P, float tp, concurrency::array_view<float, 1> &D, concurrency::array_view<int, 1> &duration);
	static concurrency::completion_future CalculateDrawdownSeq(concurrency::array_view<float, 1> P, float tp, concurrency::array_view<float, 1> &D, concurrency::array_view<int, 1> &duration);
	static void CalculateAll(std::ofstream *outputFile, std::vector<float> &prices, std::vector<std::string> &vectorDates, float startTp, float endTp, float stepTp, bool isTileMode);
		

};

