#pragma once
#include <amp.h>
#include <iostream>
class CAMPHelper
{
public:
	CAMPHelper();
	static void list_all_accelerators();
	static void default_properties();
	static bool CAMPHelper::PickEmulatedAccelerator();
	static bool CAMPHelper::PickAccelerator(std::wstring path);
	~CAMPHelper();
};

