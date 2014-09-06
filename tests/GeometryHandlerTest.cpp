#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include "../src/base/GeometryHandler.h"
#include "../src/io/FileReader.h"

BOOST_AUTO_TEST_SUITE(GeometryHandlerTest)

BOOST_AUTO_TEST_CASE(GeometryHandler_constructor) {
	GeometryHandler gh;

	std::vector<float> vertices;
	std::vector<unsigned int> indices;

	for(unsigned int i = 0; i < 33; i++) {
		vertices.push_back((float)(i*10));
		indices.push_back(i);
	}

	gh.initialize(indices, vertices);

	BOOST_CHECK_EQUAL(gh.getNumberOfIndices(), 33);
	BOOST_CHECK_EQUAL(gh.getNumberOfTriangles(), 11);
	BOOST_CHECK_EQUAL(gh.getNumberOfVertices(), 11);

}

BOOST_AUTO_TEST_CASE(GeometryHandler_pointer_initialize) {
	GeometryHandler gh;
	float* vertices = (float*)calloc(33, sizeof(float));
	unsigned int* indices = (unsigned int*)calloc(33, sizeof(unsigned int));

	for(unsigned int i = 0; i < 33; i++) {
		vertices[i] = ((float)(i*10));
		indices[i] = i;
	}

	gh.initialize(indices, vertices, 33, 33);

	BOOST_CHECK_EQUAL(gh.getNumberOfIndices(), 33);
	BOOST_CHECK_EQUAL(gh.getNumberOfTriangles(), 11);
	BOOST_CHECK_EQUAL(gh.getNumberOfVertices(), 11);

	unsigned int* temp;
	temp = gh.getTriangleAt(3);
	BOOST_CHECK_EQUAL(*temp, 9);
	BOOST_CHECK_EQUAL(*(temp+1), 10);

	float* tempf;
	tempf = gh.getVertexAt(3);
	BOOST_CHECK_EQUAL(*tempf, 90.f);

	free(vertices);
	free(indices);
}

BOOST_AUTO_TEST_CASE(GeometryHandler_getters) {
	GeometryHandler gh;

	std::vector<float> vertices;
	std::vector<unsigned int> indices;

	for(unsigned int i = 0; i < 33; i++) {
		vertices.push_back((float)(i*10));
		indices.push_back(i);
	}

	gh.initialize(indices, vertices);

	unsigned int* temp;
	temp = gh.getTriangleAt(3);
	BOOST_CHECK_EQUAL(*temp, 9);
	BOOST_CHECK_EQUAL(*(temp+1), 10);

	float* tempf;
	tempf = gh.getVertexAt(3);
	BOOST_CHECK_EQUAL(*tempf, 90.f);
}

BOOST_AUTO_TEST_CASE(GeometryHandler_getArea) {
	FileReader fr;
	GeometryHandler gh;

	fr.readVTK(&gh, "./Data/Box1m.vtk", 0.1f);

	// 1m box, area should be 6m^2
	BOOST_CHECK_EQUAL(6.f, gh.getTotalSurfaceArea());
	BOOST_CHECK_EQUAL(0.5f, gh.getSurfaceAreaAt(0));
}

BOOST_AUTO_TEST_CASE(GeometryHandler_insertLauyers) {
  GeometryHandler gh;
 	FileReader fr;
  // Inserting layer into a empty geometry handler does nothing
  std::vector<int> indices(10, 0.f);
  std::string name = "Layer0";
  gh.setLayerIndices(indices, name);
 
  BOOST_CHECK_EQUAL(gh.getNumberOfLayers(), 0);

  // Read in a Geometry
	fr.readVTK(&gh, "./Data/Box1m.vtk", 0.1f);
  // Test negative index, handler should reject this
  indices.at(0) = -10;
  indices.at(2) = 10; 
  gh.setLayerIndices(indices, name);
  BOOST_CHECK_EQUAL(gh.getNumberOfLayers(), 0);

  // Insert 2 proper layers, FOR NOW OVERLAPPING INDICES ARE ALLOWED
  int n_tri =  gh.getNumberOfTriangles();
  indices.resize(n_tri/2);
  std::cout<<"Ntri / 2: "<<n_tri/2<<std::endl;

  for(int i = 0; i <n_tri/2; i++)
    indices.at(i) = i;

  gh.setLayerIndices(indices, "Layer0");
  
  for(int i = 0; i <n_tri/2; i++)
    indices.at(i) = i+n_tri/2;

  gh.setLayerIndices(indices, "Layer1");
  BOOST_CHECK_EQUAL(gh.getNumberOfLayers(), 2);

  // Test getName 
  name = gh.getLayerNameAt(3);   // Out of bounds, return empty string
  BOOST_CHECK_EQUAL(name, "");
  name = gh.getLayerNameAt(0);
  BOOST_CHECK_EQUAL(name, "Layer0");
  name = gh.getLayerNameAt(1);
  BOOST_CHECK_EQUAL(name, "Layer1");

}


BOOST_AUTO_TEST_SUITE_END()
