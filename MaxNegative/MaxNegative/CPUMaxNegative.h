#pragma once
#include <iostream>
#include <fstream>
#include <vector>

static class CCPUMaxNegative
{
public:
	CCPUMaxNegative();
	~CCPUMaxNegative();

	static void CalculateAll(std::ofstream *outputFile, std::vector<float> &prices, std::vector<std::string> &vectorDates, float startTp, float endTp, float stepTp);

};
