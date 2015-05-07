#include "stdafx.h"
#include "CPUMaxNegative.h"
#include <iostream>
#include <fstream>
#include <string>
#include <ppl.h>

CCPUMaxNegative::CCPUMaxNegative()
{
}


CCPUMaxNegative::~CCPUMaxNegative()
{
}

void CCPUMaxNegative::CalculateDrawdown(const std::vector<float> &P, float tp, std::vector<float> &D, std::vector<int> &duration)
{
	for (int i = 0; i < P.size(); i++)
	{
		float threshold = P[i] + tp;
		float maxDrawdown = 0;
		float open = P[i];
		int j;

		for (j = i; j != P.size(); j++)
		{	
			if (open - P[j] > maxDrawdown)
				maxDrawdown = open - P[j];
			if (P[j] >= threshold)
				break;
		}
		D[i] = maxDrawdown;
		if (j < P.size())
			duration[i] = j - i;
		else
			duration[i] = -1;
	}
}

void CCPUMaxNegative::CalculateAll(std::ofstream *outputFile, std::vector<float> &prices, std::vector<std::string> &vectorDates, float startTp, float endTp, float stepTp)
{
	int barCount = prices.size();

	//for (float currTp = startTp; currTp <= endTp; currTp += stepTp)
	Concurrency::parallel_for(0, (int)((endTp - startTp) / stepTp + 1), [=](int currIter)
	{
		float currTp = startTp + currIter*stepTp;

		std::vector<float> vectorDrawdown(barCount);
		std::vector<int> vectorDuration(barCount);
		CalculateDrawdown(prices, currTp, vectorDrawdown, vectorDuration);

		if (outputFile != NULL)
			for (int i = 0; i < barCount; i++)
				*outputFile << vectorDates[i] << ", " << currTp << ", " << vectorDrawdown[i] << ", " << vectorDuration[i] << "\n";

	}
	);

}