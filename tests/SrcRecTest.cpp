#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "../src/base/SrcRec.h"
//#include "../src/io/FileReader.h"


BOOST_AUTO_TEST_SUITE(SrcRecTest)
	
BOOST_AUTO_TEST_CASE(Position_default_constructor) {
	Position pos = Position(0.f, 0.f, 0.f);

	// Initialized
	BOOST_CHECK_EQUAL(pos.getP().y , 0.f);


}

BOOST_AUTO_TEST_CASE(Position_xyz_constructor) {
	Position pos = Position(0.f, 2.f, 0.f);

	// Initialized
	BOOST_CHECK_EQUAL(pos.getP().y , 2.f);
	

}
	
BOOST_AUTO_TEST_CASE(Source_xyz_constructor) {
	Source src = Source(0.f, 0.f, 0.f);

	BOOST_CHECK_EQUAL(src.getP().y , 0.f);
	BOOST_CHECK_EQUAL(src.getSourceType(), SRC_HARD);
	
}

BOOST_AUTO_TEST_CASE(Source_set_get) {
	Source src = Source(0.f, 0.f, 0.f);

	BOOST_CHECK_EQUAL(src.getGroup(), 0);
	src.setSourceType(SRC_SOFT);
	BOOST_CHECK_EQUAL(src.getSourceType(), SRC_SOFT);
	src.setGroup(1);
	BOOST_CHECK_EQUAL(src.getGroup(), 1);
}

BOOST_AUTO_TEST_CASE(Source_get_Element_idx) {
	Source src = Source(3.f, 4.f, 5.f);

	js::Vec3i element = src.getElementIdx(7000, 344, 1/sqrtf(3.f));


}

BOOST_AUTO_TEST_SUITE_END()