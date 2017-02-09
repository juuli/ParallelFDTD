#if !defined(WIN32)
  #define BOOST_TEST_DYN_LINK
#endif

#define BOOST_TEST_MAIN


#include <boost/test/unit_test.hpp>
#include "../src/App.h"

static volatile bool interrupt = false;

void generateTestCube(std::vector<float>& verts,
                      std::vector<unsigned int>& indices) {
  float vert[24] = { 0.0,5.0,0.0, 0.0,0.0,0.0, 0.0,0.0,3.0, 0.0,5.0,3.0,
                      7.0, 0.0,3.0, 7.0,5.0,3.0, 7.0,5.0,0.0, 7.0,0.0,0.0};
  unsigned int idx[36] = {3, 1, 2, 1, 3, 0, 4, 3, 2, 3, 4, 5, 6, 1, 0, 1, 6, 7,
                          3, 6, 0, 6, 3, 5, 6, 4, 7, 4, 6, 5, 4, 1, 7, 1, 4, 2};

  for(int i = 0; i < 24; i++) verts.push_back(vert[i]);
  for(int i = 0; i < 36; i++) indices.push_back(idx[i]);

}
enum VoxelizationType vox_type = SOLID;
float dx = 0.015;

BOOST_AUTO_TEST_SUITE(AppTest)

BOOST_AUTO_TEST_CASE(App_init) {
  FDTD::App app;

  std::vector<float> vertices;
  std::vector<unsigned int> indices;
  generateTestCube(vertices, indices);

  float global_a = 0.05;

  // The app will throw -1 on error
  try {
    app.initializeDevices();
    app.initializeGeometry(&indices[0], &vertices[0], 36, 24);
    app.m_materials.setGlobalMaterial(app.m_geometry.getNumberOfTriangles(),
                                      absorption2Admitance(global_a));
    float fs = 1.0/dx*344.0*sqrt(3.0);
    app.m_parameters.setVoxelizationType(vox_type);
    app.m_parameters.setSpatialFs(fs);
    app.m_parameters.setNumSteps(300);
    app.m_parameters.setUpdateType(SRL_FORWARD);
    app.m_parameters.readGridIr("./Data/grid_ir.txt");
    app.m_parameters.addSource(Source(12.5f, 2.5f,20.5f, SRC_HARD, GAUSSIAN, 1));
    app.runSimulation();
  }
  catch(...) {

  }

  float V = app.getVolume();
  BOOST_CHECK_EQUAL(V, 5.0*7.0*3.0);

  BOOST_CHECK_EQUAL(app.m_geometry.getNumberOfIndices(), 36);
  BOOST_CHECK_EQUAL(app.m_geometry.getNumberOfTriangles(), 12);
  float SA = app.getTotalAborptionArea(0);
  float A = app.m_geometry.getTotalSurfaceArea();
  BOOST_CHECK_EQUAL(SA, (5.0*7.0+5.0*3.0+7.0*3.0)*2.0*global_a);

  float sab = app.getSabine(0);
  BOOST_CHECK_EQUAL(sab, 0.1611*V/SA);

  float eyr = app.getEyring(0);
  BOOST_CHECK_EQUAL(eyr, 0.1611*V/(-1.f*A*logf(1.f- SA/A)));

  app.close();
}

BOOST_AUTO_TEST_CASE(App_init_material_list) {
  FDTD::App app;

  std::vector<float> vertices;
  std::vector<unsigned int> indices;
  generateTestCube(vertices, indices);

  float global_a = 0.05;
  std::vector<float> materials;

  int number_of_surfaces = 12;
  int number_of_coefficients = 20;

  // add same material to all 12 surfaces
  for(int i = 0; i < 12; i ++){
    if(i < 2) global_a = 0.01;
    else global_a = 0.05;

    for (int j = 0; j < 20; j++) {
      materials.push_back(absorption2Admitance(global_a));
    }
  }
  BOOST_CHECK_EQUAL(materials.size(), 240);
  // The app will throw -1 on error
  try {
    app.initializeDevices();
    app.initializeGeometry(&indices[0], &vertices[0], 36, 24);
    app.m_materials.addMaterials(&materials[0],
                                 number_of_surfaces,
                                 number_of_coefficients);
    float fs = 1.0/dx*344.0*sqrt(3.0);
    app.m_parameters.setSpatialFs(fs);
    app.setVoxelizationType(vox_type);
    app.m_parameters.setNumSteps(300);
    app.m_parameters.setUpdateType(SRL_FORWARD);
    app.m_parameters.readGridIr("./Data/grid_ir.txt");
    app.m_parameters.addSource(Source(12.5f, 2.5f,20.5f, SRC_HARD, GAUSSIAN, 1));
    app.runSimulation();
  }
  catch(...) {

  }

  float V = app.getVolume();
  BOOST_CHECK_EQUAL(V, 5.0*7.0*3.0);

  BOOST_CHECK_EQUAL(app.m_geometry.getNumberOfIndices(), 36);
  BOOST_CHECK_EQUAL(app.m_geometry.getNumberOfTriangles(), 12);
  float SA = app.getTotalAborptionArea(0);
  float A = app.m_geometry.getTotalSurfaceArea();
  BOOST_CHECK_EQUAL(SA, ((5.0*7.0+.0+7.0*3.0)*2.0+5.0*3)*0.05+5.0*3*0.01);

  float sab = app.getSabine(0);
  BOOST_CHECK_EQUAL(sab, 0.1611*V/SA);

  float eyr = app.getEyring(0);
  BOOST_CHECK_EQUAL(eyr, 0.1611*V/(-1.f*A*logf(1.f- SA/A)));

  app.close();
}


BOOST_AUTO_TEST_SUITE_END()
