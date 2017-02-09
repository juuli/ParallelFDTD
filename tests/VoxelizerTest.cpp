#if !defined(WIN32)
  #define BOOST_TEST_DYN_LINK
#endif

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>


#include "../src/Voxelizer/include/helper_math.h"
#include "../src/Voxelizer/include/voxelizer.h"


#include "../src/kernels/voxelizationUtils.h"
#include "../src/base/GeometryHandler.h"
#include "../src/io/FileReader.h"

BOOST_AUTO_TEST_SUITE(VoxelizerTest)

BOOST_AUTO_TEST_CASE(VoxelizeGeometry) {
  GeometryHandler gh;
  FileReader fr;
  fr.readVTK(&gh, "./Data/box_753.vtk");

  // Shortcut variables
  nv::Vec3f bb = gh.getBoundingBox();
  float* verts = gh.getVerticePtr();
  unsigned int* indices = gh.getIndexPtr();
  unsigned int num_triangles = gh.getNumberOfTriangles();
  unsigned int num_vertices = gh.getNumberOfVertices();
  unsigned int num_uniq_mat = 1;

  unsigned int displace_mesh_voxels = 1;
  double voxel_edge = 0.02;
  uint3 vox_dim;
  vox_dim = get_Geometry_surface_Voxelization_dims(verts, indices,
                                                   num_triangles,
                                                   num_vertices,
                                                   voxel_edge);

  // Vox dims are padded by default to be multiple of uint3(32, 4, 1)
  // Additionally 1 layers of cells are added at each end
  unsigned int check_x = (unsigned int)round(bb.x/voxel_edge)+2;
  check_x += (32-(check_x%32));
  unsigned int check_y = (unsigned int)round(bb.y/voxel_edge)+2;
  check_y += (4-(check_y%4));
  unsigned int check_z = (unsigned int)round(bb.z/voxel_edge)+2;

  BOOST_CHECK_EQUAL(vox_dim.x, check_x);
  BOOST_CHECK_EQUAL(vox_dim.y, check_y);
  BOOST_CHECK_EQUAL(vox_dim.z, check_z);

  std::vector<unsigned char> materials(num_triangles, 1);

  unsigned char* h_pos = NULL;
  unsigned char* h_mat = NULL;

  voxelizeGeometrySurfToHost(&(verts[0]),
                             &(indices[0]),
                             &(materials[0]),
                             num_triangles,
                             num_vertices,
                             num_uniq_mat,
                             voxel_edge,
                             &h_pos,
                             &h_mat,
                             &vox_dim,
                             displace_mesh_voxels,
                             VOX_6_SEPARATING,
                             make_uint3(32, 4, 1),
                             0, vox_dim.z);

  unsigned int dim_x = vox_dim.x;
  unsigned int dim_y = vox_dim.y;
  unsigned int dim_z = vox_dim.z;
  unsigned int dim_xy = vox_dim.x*vox_dim.y;

  // Edge of the geometry is at coordinate 1, 1, 2
  BOOST_CHECK_EQUAL(h_pos[1+dim_x+dim_xy*2], 0);

  // Check how many voxels are inside
  bool in = false;
  unsigned int count = 0;
  for(unsigned int i = 0; i < dim_x; i++) {
    unsigned int cur = i+2*dim_x+dim_xy*3;
    unsigned char val = h_pos[cur];
    if(val == 0) {
      printf("IN/OUT, x: %u\n", i);
      in = !in;
    }
    if(val == 128 && in)
      count++;
  }

  // Calculate the number of voxels from the initial bounding box
  unsigned int check = (unsigned int)floor(bb.x/voxel_edge);
  BOOST_CHECK_EQUAL(count, check);

  // Check boundary value calculation
  calcBoundaryValuesInplace(h_pos, dim_x, dim_y, dim_z);

  // "outside" point should still be where it was
  BOOST_CHECK_EQUAL((unsigned int)h_pos[1+dim_x+dim_xy*2], 0);
  // A corner should lie one step diagonally in
  BOOST_CHECK_EQUAL((unsigned int)h_pos[2+dim_x*2+dim_xy*3], 3+128);
  // One next to it towards x should be an edge
  BOOST_CHECK_EQUAL((unsigned int)h_pos[3+dim_x*2+dim_xy*3], 4+128);
  // One step towards y should be a wall
  BOOST_CHECK_EQUAL((unsigned int)h_pos[3+dim_x*3+dim_xy*3], 5+128);
  // One step towards z should be air
  BOOST_CHECK_EQUAL((unsigned int)h_pos[3+dim_x*3+dim_xy*4], 6+128);

  // Check material index calculation
  calcMaterialIndicesInplace(h_mat, h_pos, dim_x, dim_y, dim_z);

  // Only one material now so all should be 1
  BOOST_CHECK_EQUAL((unsigned int)h_mat[2+dim_x*2+dim_xy*3], 1);
  BOOST_CHECK_EQUAL((unsigned int)h_mat[3+dim_x*2+dim_xy*3], 1);
  BOOST_CHECK_EQUAL((unsigned int)h_mat[3+dim_x*3+dim_xy*3], 1);
  // Except air is 0
  BOOST_CHECK_EQUAL((unsigned int)h_mat[3+dim_x*3+dim_xy*4], 0);

  delete[] h_pos;
  delete[] h_mat;
}

BOOST_AUTO_TEST_CASE(PadWithZerosTest) {


}


BOOST_AUTO_TEST_SUITE_END()
