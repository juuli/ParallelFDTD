#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>


#include "../src/Voxelizer/include/helper_math.h"
#include "../src/Voxelizer/include/voxelizer.h"


#include "../src/kernels/voxelizationUtils.h"
#include "../src/base/GeometryHandler.h"
#include "../src/io/FileReader.h"

BOOST_AUTO_TEST_SUITE(VoxelizerTest)
/*
BOOST_AUTO_TEST_CASE(VoxelizerMaterials) {

	FileReader fr;
	GeometryHandler gh;
	MaterialHandler mh;

	unsigned int resolution;

	std::string geometry_fp = "./Data/ptb_geometry.vtk";
	fr.readVTK(&gh, geometry_fp);

	mh.setGlobalMaterial(gh.getNumberOfTriangles(), 0.5f);

	std::vector<unsigned char> mock_materials;
	mock_materials.assign(gh.getNumberOfTriangles(), 1);

	// Voxelizer(float* _vertices, uint* _indices, uchar* _materials, uint _nrOfVertices, uint _nrOfTriangles, uint _nrOfUniqueMaterials);

	Voxelizer<LongNode> voxelizer(gh.getVerticePtr(), 
																gh.getIndexPtr(), 
						gh.getNumberOfVertices(),
						gh.getNumberOfTriangles());

	voxelizer.setMaterialOutput(true);
	voxelizer.setMaterials(&(mock_materials[0]), 1);


}

BOOST_AUTO_TEST_CASE(VoxelizeGeometry) {
	// Test voxelization and compare it to the mesh of WaveModeller
	FileReader fr;
	GeometryHandler gh;
	SimulationParameters sp;

	unsigned int resolution;

	fr.readVTK(&gh, "./Data/ptb_geometry.vtk");

	// Voxelizer takes shallow copy these as fields

	float* vrt = (float*)calloc(gh.getNumberOfVertices()*3, sizeof(float));
	memcpy ( vrt, gh.getVerticePtr(), gh.getNumberOfVertices()*3*sizeof(float) );

	unsigned int* idx = (unsigned int*)calloc(gh.getNumberOfTriangles()*3, sizeof(unsigned int));
	memcpy (idx , gh.getIndexPtr(), gh.getNumberOfTriangles()*3*sizeof(unsigned int));

	Voxelizer<LongNode> voxelizer(vrt,  idx, gh.getNumberOfVertices(),gh.getNumberOfTriangles());

	cudasafe(cudaGetLastError(), "VozelizeGeometry - peek after voxelization");
	// Resolution = the number of nodes on the long edge
	resolution = gh.getNumberOfLongEdgeNodes(0.1);
	voxelizer.setResolution(resolution);

	LongNode* nodes = voxelizer.voxelizeToNodesToRAM();
	
	uint3 node_dim = voxelizer.getResolution();
	unsigned int number_of_nodes = node_dim.x*node_dim.y*node_dim.z;

	unsigned int temp_k = (unsigned int)nodes[1].bid();

	std::cout<<"x: "<<node_dim.x<<" y: "<<node_dim.y<<" z: "<<node_dim.z<<std::endl;
	
	free(vrt);
	free(idx);
	free(nodes); // This has to be freed from ram aswell

	BOOST_CHECK_EQUAL(resolution, 102); // This is completely made up
}

BOOST_AUTO_TEST_CASE(PadWithZerosTest) {


}
*/

BOOST_AUTO_TEST_SUITE_END()
