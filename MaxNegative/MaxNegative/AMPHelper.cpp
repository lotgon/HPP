#include "stdafx.h"
#include "AMPHelper.h"
using namespace concurrency;


CAMPHelper::CAMPHelper()
{
}


CAMPHelper::~CAMPHelper()
{
}

void CAMPHelper::list_all_accelerators()
{
	std::wcout << "--------------List all accelerators"<<std::endl;
	std::vector<accelerator> accs = accelerator::get_all();
	for (int i = 0; i < accs.size(); i++) {
		std::wstring s = accs[i].device_path;
		std::wcout << accs[i].device_path << "\n";
		std::wcout << accs[i].description << "\n";
		std::wcout << accs[i].dedicated_memory << "\n";
		std::wcout << (accs[i].supports_cpu_shared_memory ?
			"CPU shared memory: true" : "CPU shared memory: false") << "\n";
		std::wcout << (accs[i].supports_double_precision ?
			"double precision: true" : "double precision: false") << "\n";
		std::wcout << (accs[i].supports_limited_double_precision ?
			"limited double precision: true" : "limited double precision: false") << "\n";
		std::wcout << std::endl;
	}
}
bool CAMPHelper::PickEmulatedAccelerator()
{
	std::vector<accelerator> accs = accelerator::get_all();
	accelerator chosen_one;

	auto result =
		std::find_if(accs.begin(), accs.end(), [](const accelerator& acc)
	{
		return acc.is_emulated;
	});

	if (result != accs.end())
		chosen_one = *(result);

	std::wcout << std::endl << "Pick " << chosen_one.description << std::endl;

	bool success = accelerator::set_default(chosen_one.device_path);
	return success;
}

void CAMPHelper::default_properties() {
	concurrency::accelerator default_acc;
	std::wcout << default_acc.device_path << "\n";
	std::wcout << default_acc.dedicated_memory << "\n";
	std::wcout << (default_acc.supports_cpu_shared_memory ?
		"CPU shared memory: true" : "CPU shared memory: false") << "\n";
	std::wcout << (default_acc.supports_double_precision ?
		"double precision: true" : "double precision: false") << "\n";
	std::wcout << (default_acc.supports_limited_double_precision ?
		"limited double precision: true" : "limited double precision: false") << "\n";
}
bool CAMPHelper::PickAccelerator(std::wstring path)
{
	bool success = accelerator::set_default(path);
	return success;
}
