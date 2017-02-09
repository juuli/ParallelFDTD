#if !defined(WIN32)
  #define BOOST_TEST_DYN_LINK
#endif

#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

#include "../src/kernels/cudaUtils.h"
#include "../src/App.h"
#include <vector>
#include <fstream>

BOOST_AUTO_TEST_SUITE(CudaUtilsTest)

BOOST_AUTO_TEST_CASE(dataToDevice) {
	std::vector<unsigned char> uc_test;
	std::vector<unsigned int> ui_test;
	std::vector<float> float_test;
	unsigned int mem_size = 20;

	for(unsigned int i = 0; i < mem_size; i++){
		uc_test.push_back((unsigned char)i);
		ui_test.push_back(i);
		float_test.push_back((float)i);
	}

	unsigned int* d_ui = toDevice<unsigned int>(mem_size, &(ui_test[0]), 0);
	unsigned char* d_uc = toDevice<unsigned char>(mem_size, &(uc_test[0]), 0);
	float* d_float = toDevice<float>(mem_size, &(float_test[0]), 0);
	float* d_float_zero = toDevice<float>(mem_size, 0);

	// Data back from device THIS MEMORY HAS TO BE FREED SEPARATELY

	unsigned int* h_ui = fromDevice<unsigned int>(mem_size, d_ui, 0);
	unsigned char* h_uc = fromDevice<unsigned char>(mem_size, d_uc, 0);
	float* h_float = fromDevice<float>(mem_size, d_float, 0);
	float* h_float_zero = fromDevice<float>(mem_size, d_float_zero, 0);

	bool check = true;

	for(unsigned i = 0; i < mem_size; i++){
		if(h_ui[i] != (unsigned int)i )
			check = false;
	}

	BOOST_CHECK_EQUAL(check, true); 

	check = true;

	for(unsigned i = 0; i < mem_size; i++){
		if(h_uc[i] != (unsigned char)i )
			check = false;
	}
	BOOST_CHECK_EQUAL(check, true); 

	check = true;

	for(unsigned i = 0; i < mem_size; i++){
		if(h_float[i] != (float)i )
			check = false;
	}
	BOOST_CHECK_EQUAL(check, true); 

	check = true;

	for(unsigned i = 0; i < mem_size; i++){
		if(h_float_zero[i] != 0.f)
			check = false;
	}
	BOOST_CHECK_EQUAL(check, true); 

	// Test reset function
	resetData(mem_size, d_float, 0);
	free(h_float);

	h_float = fromDevice(mem_size, d_float, 0);
	check = true;
	for(unsigned i = 0; i < mem_size; i++){
		if(h_float[i] != 0.f)
			check = false;
	}
	BOOST_CHECK_EQUAL(check, true); 

	destroyMem(d_ui);
	destroyMem(d_uc);
	destroyMem(d_float);
	destroyMem(d_float_zero);
	free(h_ui);
	free(h_uc);
	free(h_float);
	free(h_float_zero);
}


BOOST_AUTO_TEST_SUITE_END()
