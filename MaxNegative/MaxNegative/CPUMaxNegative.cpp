#include "stdafx.h"
#include "CPUMaxNegative.h"
#include <iostream>
#include <fstream>
#include <string>

CCPUMaxNegative::CCPUMaxNegative()
{
}


CCPUMaxNegative::~CCPUMaxNegative()
{
}

void CCPUMaxNegative::CalculateAll(std::ofstream *outputFile, std::vector<float> &prices, std::vector<std::string> &vectorDates, float startTp, float endTp, float stepTp)
{
	int barCount = prices.size();

	for (float currTp = startTp; currTp <= endTp; currTp += stepTp)
	{
		std::vector<float> vectorDrawdown(barCount);
		std::vector<int> vectorDuration(barCount);
	

		if (outputFile != NULL)
			for (int i = 0; i < barCount; i++)
				*outputFile << vectorDates[i] << ", " << currTp << ", " << vectorDrawdown[i] << ", " << vectorDuration[i] << "\n";

	}

}