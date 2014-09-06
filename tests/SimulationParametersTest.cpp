#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "../src/base/SimulationParameters.h"

BOOST_AUTO_TEST_SUITE(SimulationParametersTest)

BOOST_AUTO_TEST_CASE(SimulationParameters_constructor) {
	SimulationParameters sp;
	// Check default values
	BOOST_CHECK_EQUAL(sp.getC() , 344);
	BOOST_CHECK_EQUAL(sp.getLambda() , (double)1/sqrt((double)3));
	BOOST_CHECK_EQUAL(sp.getRho() , 1.225f);
}

BOOST_AUTO_TEST_CASE(SimulationParameters_getters) {
	SimulationParameters sp;

	sp.setSpatialFs(2000);
	float reference = sp.getC()/sp.getSpatialFs()/(1/sqrtf(3.f));
	BOOST_CHECK_EQUAL(sp.getDx(), reference);
}

BOOST_AUTO_TEST_CASE(SimulationParameters_add_source_receiver) {
	SimulationParameters sp;
	sp.addSource(Source(3.f,4.f,5.f));
	sp.addSource(7.f,8.f,9.f);

	BOOST_CHECK_EQUAL(sp.getSource(0).getP().x , 3.f);
	BOOST_CHECK_EQUAL(sp.getSource(1).getP().x , 7.f);
	BOOST_CHECK_THROW(sp.getSource(2), std::out_of_range);
	BOOST_CHECK_EQUAL(sp.getNumSources(), 2);
	
	sp.addReceiver(Receiver(0.f,4.f,5.f));
	sp.addReceiver(1.f, 5.f, 3.f);

	BOOST_CHECK_EQUAL(sp.getReceiver(0).getP().x , 0.f);
	BOOST_CHECK_EQUAL(sp.getReceiver(1).getP().x , 1.f);
	BOOST_CHECK_THROW(sp.getReceiver(2), std::out_of_range);
	BOOST_CHECK_EQUAL(sp.getNumReceivers(), 2);
}

BOOST_AUTO_TEST_CASE(SimulationParameters_update_souce_receiver) {
	SimulationParameters sp;
	sp.addSource(3.f,4.f,5.f);
	sp.addSource(7.f,4.f,3.f);

	sp.addReceiver(1.f,2.f,3.f);
	sp.addReceiver(6.f,7.f,8.f);


	BOOST_CHECK_THROW(sp.updateSourceAt(2, Source()), std::out_of_range);
	sp.updateSourceAt(1, Source(1.f, 1.f, 1.f, SRC_SOFT));
	BOOST_CHECK_EQUAL(sp.getSource(1).getP().x, 1.f);
	BOOST_CHECK_EQUAL(sp.getSource(1).getSourceType(), SRC_SOFT);

	sp.updateReceiverAt(0, Receiver(0.f, 0.f, 0.f));
	BOOST_CHECK_EQUAL(sp.getReceiver(0).getP().x, 0.f);
}

BOOST_AUTO_TEST_CASE(SimulationParameters_remove_source_receiver) {
	SimulationParameters sp;
	sp.addSource(3.f,4.f,5.f);
	sp.addSource(7.f,4.f,3.f);

	sp.addReceiver(1.f,2.f,3.f);
	sp.addReceiver(6.f,7.f,8.f);


	BOOST_CHECK_THROW(sp.removeSource(2), std::out_of_range);
	BOOST_CHECK_NO_THROW(sp.removeSource(1));
	BOOST_CHECK_EQUAL(sp.getNumSources() , 1);

	BOOST_CHECK_THROW(sp.removeReceiver(2), std::out_of_range);
	BOOST_CHECK_NO_THROW(sp.removeReceiver(1));
	BOOST_CHECK_EQUAL(sp.getNumReceivers() , 1);

}

BOOST_AUTO_TEST_CASE(SimulationParameters_getParameterPtr) {
	SimulationParameters sp;
	float* params = sp.getParameterPtr();
  double* params_d = sp.getParameterPtrDouble();

  unsigned int octave = sp.getOctave();
	// For now its only lambda and lambda^2 there
  double lambda = (1/sqrt(double(3)));
	BOOST_CHECK_EQUAL(params[0], (float)lambda);
	BOOST_CHECK_EQUAL(params[1], (float)(lambda*lambda));
	BOOST_CHECK_EQUAL(params[2], 1.f/3.f);
  BOOST_CHECK_EQUAL(params[3], (float)octave);

  BOOST_CHECK_EQUAL(params_d[0], lambda);
	BOOST_CHECK_EQUAL(params_d[1], lambda*lambda);
  BOOST_CHECK_EQUAL(params_d[2], (double)1/(double)3);
	BOOST_CHECK_EQUAL(params_d[3], (double)octave);
}

BOOST_AUTO_TEST_CASE(SimulationParameters_add_audio_input_data) {
	SimulationParameters sp;

	sp.setNumSteps(1000);

	unsigned int len_1 = 100;
	float* data_1 = new float[len_1];
	unsigned int len_2 = 200;
	float* data_2 = new float[len_2];

	for(unsigned int i = 0; i < len_2; i++) {
		if(i < len_1)
			data_1[i] = (float)i;
		data_2[i] = -1.f*(float)i;
	}

	sp.addInputData(data_1, len_1);
	sp.addInputData(data_2, len_2);

	BOOST_CHECK_EQUAL(sp.getInputDataSample(0,0), 0);
	BOOST_CHECK_EQUAL(sp.getInputDataSample(1,20), -20);
	BOOST_CHECK_EQUAL(sp.getInputDataSample(0,99), 99);
	BOOST_CHECK_EQUAL(sp.getInputDataSample(0,1000), 0);
	BOOST_CHECK_EQUAL(sp.getInputDataSample(0,1000), 0);
	BOOST_CHECK_THROW(sp.getInputDataSample(2,1000), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(SimulationParameters_getSourceSample) {
	SimulationParameters sp;

	sp.addSource(Source(1.f, 1.f, 1.f, SRC_HARD, IMPULSE, 0));
	sp.addSource(Source(1.f, 1.f, 1.f, SRC_HARD, DATA, 0));

	std::vector<float> input_data;
	for(unsigned int i = 0; i < 200; i++) {
		input_data.push_back((float)i);
	}

	sp.addInputData(&(input_data[0]), (unsigned int)(input_data.size()-1));

	// First One is an impulse
	BOOST_CHECK_EQUAL(sp.getSourceSample(0,0), 0.f);
	BOOST_CHECK_EQUAL(sp.getSourceSample(0,1), 1.f);
	BOOST_CHECK_EQUAL(sp.getSourceSample(0,2), 0.f);

	// Second has input data
	BOOST_CHECK_EQUAL(sp.getSourceSample(1,0), 0.f);
	BOOST_CHECK_EQUAL(sp.getSourceSample(1,1), 1.f);
	BOOST_CHECK_EQUAL(sp.getSourceSample(1,100), 100.f);
	// indexing over vector gives 0
	BOOST_CHECK_EQUAL(sp.getSourceSample(1,300), 0.f);
	BOOST_CHECK_THROW(sp.getSourceSample(2,1000), std::out_of_range);

}

BOOST_AUTO_TEST_CASE(SimulationParameters_getSourceSampleVector) {
  SimulationParameters sp;
  sp.setNumSteps(200);
	sp.addSource(Source(1.f, 1.f, 1.f, SRC_HARD, IMPULSE, 0));
	sp.addSource(Source(1.f, 1.f, 1.f, SRC_HARD, DATA, 0));

	std::vector<float> input_data;
	for(unsigned int i = 0; i < 200; i++) {
		input_data.push_back((float)i);
	}

	sp.addInputData(&(input_data[0]), (unsigned int)(input_data.size()-1));
  	
  float* source_1 = sp.getSourceVectorAt(0);
  float* source_2 = sp.getSourceVectorAt(1);

  for(int i = 0; i < sp.getNumSteps(); i++) {
    BOOST_CHECK_EQUAL(source_1[i], sp.getSourceSample(0, i));
    BOOST_CHECK_EQUAL(source_2[i], sp.getSourceSample(1, i));
  }

  // First One is an impulse
	BOOST_CHECK_EQUAL(source_1[0], 0.f);
	BOOST_CHECK_EQUAL(source_1[1], 1.f);
	BOOST_CHECK_EQUAL(source_1[2], 0.f);

	// Second has input data
	BOOST_CHECK_EQUAL(source_2[0], 0.f);
	BOOST_CHECK_EQUAL(source_2[1], 1.f);
	BOOST_CHECK_EQUAL(source_2[100], 100.f);

}

BOOST_AUTO_TEST_CASE(SimulationParameters_grid_ir) {
	SimulationParameters sp;
	sp.readGridIr("./Data/h_ir_3D_1000.txt");

	BOOST_CHECK_EQUAL(sp.getGridIrDataSample(0), 0.f);
	BOOST_CHECK_EQUAL(sp.getGridIrDataSample(2), -0.333333343f);
	BOOST_CHECK_EQUAL(sp.getGridIrDataSample(998), (float)-2.1860822614804370e-005);
}

BOOST_AUTO_TEST_CASE(SimulationParameters_getTransparentSample) {
	SimulationParameters sp;
	sp.readGridIr("./Data/h_ir_3D_1000.txt");

	sp.addSource(Source(1.f,1.f,1.f, SRC_TRANSPARENT, IMPULSE,0));
	sp.setNumSteps(100);

	BOOST_CHECK_EQUAL(sp.getSourceSample(0,0), 0.f);
	BOOST_CHECK_EQUAL(sp.getSourceSample(0,1), 1.f);
	BOOST_CHECK_EQUAL(sp.getSourceSample(0,3), 1.f/3.f);

}

BOOST_AUTO_TEST_SUITE_END()
