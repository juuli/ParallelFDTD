#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include "../src/global_includes.h"
#include "../src/io/FileReader.h"
#include "../src/base/GeometryHandler.h"
#include "../src/base/MaterialHandler.h"

BOOST_AUTO_TEST_SUITE(FileReaderTest)

BOOST_AUTO_TEST_CASE(FileReader_readVTK) {
	FileReader fr;
	GeometryHandler gh;
	std::string geometry_fp = "./Data/ptb_geometry.vtk";
	std::string invalid_fp = ".s21Da/ptp_geometry.vköööee3222";

	bool ret = false;
	ret = fr.readVTK(&gh, geometry_fp);
	BOOST_CHECK_EQUAL(ret, true);

	ret = false;
	ret = fr.readVTK(&gh, invalid_fp);
	BOOST_CHECK_EQUAL(ret, false);

}


BOOST_AUTO_TEST_SUITE_END()
