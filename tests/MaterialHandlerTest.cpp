#if !defined(WIN32)
  #define BOOST_TEST_DYN_LINK
#endif

#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include "../src/base/MaterialHandler.h"
#include "../src/global_includes.h"

BOOST_AUTO_TEST_SUITE(MaterialHandlerTest)

BOOST_AUTO_TEST_CASE(MaterialHandler_constructor) {
  MaterialHandler mh;
  // Check default values
  BOOST_CHECK_EQUAL(mh.getNumberOfSurfaces() , 0);

}

BOOST_AUTO_TEST_CASE(MaterialHandler_addSurfaceMaterial) {
  MaterialHandler mh;
  mh.setMaterialIndexIncrement(0);
  // Initialize temp material vector
  std::vector<float> temp;
  for(unsigned int i = 0; i < mh.getNumberOfCoefficients(); i++) {
    temp.push_back((float)i);
  }

  mh.addSurfaceMaterial(temp);
  mh.addSurfaceMaterial(temp);

  BOOST_CHECK_EQUAL(mh.getNumberOfUniqueMaterials() , 1);
  BOOST_CHECK_EQUAL(mh.getNumberOfSurfaces(), 2);

  temp.at(0) = 3.f;
  mh.addSurfaceMaterial(temp);
  BOOST_CHECK_EQUAL(mh.getNumberOfUniqueMaterials() , 2);
  BOOST_CHECK_EQUAL(mh.getNumberOfSurfaces(), 3);

  temp.clear();
  for(unsigned int i = 0; i < 10; i++) {
    temp.push_back((float)i);
  }
  mh.addSurfaceMaterial(temp);
  BOOST_CHECK_EQUAL(mh.getNumberOfUniqueMaterials() , 3);
  BOOST_CHECK_EQUAL(mh.getNumberOfSurfaces(), 4);
}

BOOST_AUTO_TEST_CASE(MaterialHandler_addSurfaceMaterial_with_increment) {
  MaterialHandler mh;
  unsigned int idx_increment = 0;
  mh.setMaterialIndexIncrement(idx_increment);
  // Initialize temp material vector
  std::vector<float> temp;
  for(unsigned int i = 0; i < mh.getNumberOfCoefficients(); i++) {
    temp.push_back((float)i);
  }

  mh.addSurfaceMaterial(temp);
  mh.addSurfaceMaterial(temp);

  BOOST_CHECK_EQUAL(mh.getNumberOfUniqueMaterials() , 1+idx_increment);
  BOOST_CHECK_EQUAL(mh.getNumberOfSurfaces(), 2);

  temp.at(0) = 3.f;
  mh.addSurfaceMaterial(temp);
  BOOST_CHECK_EQUAL(mh.getNumberOfUniqueMaterials() , 2+idx_increment);
  BOOST_CHECK_EQUAL(mh.getNumberOfSurfaces(), 3);

  temp.clear();
  for(unsigned int i = 0; i < 10; i++) {
    temp.push_back((float)i);
  }
  mh.addSurfaceMaterial(temp);
  BOOST_CHECK_EQUAL(mh.getNumberOfUniqueMaterials() , 3+idx_increment);
  BOOST_CHECK_EQUAL(mh.getNumberOfSurfaces(), 4);
}

BOOST_AUTO_TEST_CASE(MaterialHandler_getMaterial) {
  MaterialHandler mh;
  unsigned int idx_increment = 0;
  mh.setMaterialIndexIncrement(idx_increment);
  // Initialize temp material vector
  std::vector<float> temp;
  for(unsigned int i = 0; i < mh.getNumberOfCoefficients(); i++) {
    temp.push_back((float)i);
  }

  mh.addSurfaceMaterial(temp);
  mh.addSurfaceMaterial(temp);
  temp.at(0) = 3.f;
  mh.addSurfaceMaterial(temp); // 3rd surface differs


  temp.clear(); // Insert material with less than 20 coefficients
  for(unsigned int i = 0; i < 10; i++) {
    temp.push_back((float)i);
  }

  mh.addSurfaceMaterial(temp); // 4th surface


  BOOST_CHECK_EQUAL(mh.getNumberOfSurfaces(), 4);

  BOOST_CHECK_EQUAL(mh.getMaterialIdxAt(0), 0+idx_increment);
  BOOST_CHECK_EQUAL(mh.getMaterialIdxAt(1), 0+idx_increment);
  BOOST_CHECK_EQUAL(mh.getMaterialIdxAt(2), 1+idx_increment);
  BOOST_CHECK_EQUAL(mh.getMaterialIdxAt(3), 2+idx_increment);

  // The indexing starts at 0 to call
  BOOST_CHECK_EQUAL(mh.getUniqueCoefAt(0,0), 0.f);
  BOOST_CHECK_EQUAL(mh.getUniqueCoefAt(0,19), 19.f);
  BOOST_CHECK_EQUAL(mh.getUniqueCoefAt(0,3), 3.f);
  BOOST_CHECK_EQUAL(mh.getUniqueCoefAt(1,0), 3.f);

  // Getters for surface coefficients
  BOOST_CHECK_EQUAL(mh.getSurfaceCoefAt(0,5), 5.f);
  BOOST_CHECK_EQUAL(mh.getSurfaceCoefAt(2,0), 3.f);

  // Material with less than 10 coefs has zeros at the end
  BOOST_CHECK_EQUAL(mh.getSurfaceCoefAt(3,11), 0.f);

  // Exeptions
  BOOST_CHECK_THROW(mh.getMaterialIdxAt(4), std::out_of_range);
  BOOST_CHECK_THROW(mh.getUniqueCoefAt(4,0), std::out_of_range);
  BOOST_CHECK_THROW(mh.getUniqueCoefAt(1,22), std::out_of_range);
  BOOST_CHECK_THROW(mh.getSurfaceCoefAt(1,22), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(MaterialHandler_addMaterials) {
  MaterialHandler mh;
  mh.setMaterialIndexIncrement(0);
  float* material_ptr = new float[50];
  unsigned int number_of_surfaces = 5;
  unsigned int number_of_coefficients = 10;

  // Add 5 new surface materials, all different
  for(unsigned int i = 0; i < number_of_surfaces*number_of_coefficients; i++) {
    material_ptr[i] = (float)i;
  }

  mh.addMaterials(material_ptr, number_of_surfaces, number_of_coefficients);
  delete [] material_ptr;

  BOOST_CHECK_EQUAL(mh.getNumberOfUniqueMaterials(), 5);

  // Check the materials
  // getUniqueCoefAt(unsigned int material_idx, unsigned int coef_idx)
  BOOST_CHECK_EQUAL(mh.getUniqueCoefAt(1,0), 10.f);
  BOOST_CHECK_EQUAL(mh.getUniqueCoefAt(0,14), 0.f);
  BOOST_CHECK_EQUAL(mh.getUniqueCoefAt(4,9), 49.f);
}

BOOST_AUTO_TEST_CASE(MaterialHandler_getPointer) {
  MaterialHandler mh;
  mh.setMaterialIndexIncrement(0);
  unsigned int number_of_coefficients = 10;

  // Initialize temp material vector
  std::vector<float> temp;
  for(unsigned int i = 0; i < number_of_coefficients; i++) {
    temp.push_back(((float)i)/10.f);
  }

  mh.addSurfaceMaterial(temp);
  temp.at(0) = 0.7f;
  mh.addSurfaceMaterial(temp);

  float* m_ptr =  mh.getMaterialCoefficientPtr();
  double* m_ptr_d =  mh.getMaterialCoefficientPtrDouble();
  BOOST_CHECK_EQUAL(20, mh.getNumberOfCoefficients());
  BOOST_CHECK_EQUAL(2,  mh.getNumberOfUniqueMaterials());

  // Material has 20 coefficients, first 10 are admitances and second 10 scattering
  // If less than 20 coef are assigned, rest are padded
  // with zeros

  BOOST_CHECK_EQUAL(m_ptr[0], 0.f);
  BOOST_CHECK_EQUAL(m_ptr[19], 0.f);
  BOOST_CHECK_EQUAL(m_ptr[9], 0.9f);
  BOOST_CHECK_EQUAL(m_ptr[20], 0.7f);


  BOOST_CHECK_EQUAL(m_ptr_d[0], (double)0.0f);
  BOOST_CHECK_EQUAL(m_ptr_d[19], (double)0.0f);
  BOOST_CHECK_EQUAL(m_ptr_d[9], (double)0.9f);
  BOOST_CHECK_EQUAL(m_ptr_d[20], (double)0.7f);

}

BOOST_AUTO_TEST_CASE(MaterialHandler_getPointer_with_increment) {
  MaterialHandler mh;
  unsigned int idx_increment = 1;
  mh.setMaterialIndexIncrement(idx_increment);
  unsigned int number_of_coefficients = 10;

  // Initialize temp material vector
  std::vector<float> temp;
  for(unsigned int i = 0; i < number_of_coefficients; i++) {
    temp.push_back(((float)i)/10.f);
  }

  mh.addSurfaceMaterial(temp);
  temp.at(0) = 0.7f;
  mh.addSurfaceMaterial(temp);
  BOOST_CHECK_EQUAL(2+idx_increment,  mh.getNumberOfUniqueMaterials());

  float* m_ptr =  mh.getMaterialCoefficientPtr();
  double* m_ptr_d =  mh.getMaterialCoefficientPtrDouble();
  BOOST_CHECK_EQUAL(20, mh.getNumberOfCoefficients());


  // Material has 20 coefficients, first 10 are admitances and second 10 scattering
  // If less than 20 coef are assigned, rest are padded
  // with zeros

  int c_inc = 20*idx_increment;
  BOOST_CHECK_EQUAL(m_ptr[0], 0.f);
  BOOST_CHECK_EQUAL(m_ptr[0+c_inc], 0.f);
  BOOST_CHECK_EQUAL(m_ptr[19+c_inc], 0.f);
  BOOST_CHECK_EQUAL(m_ptr[9+c_inc], 0.9f);
  BOOST_CHECK_EQUAL(m_ptr[20+c_inc], 0.7f);


  BOOST_CHECK_EQUAL(m_ptr_d[0+c_inc], (double)0.0f);
  BOOST_CHECK_EQUAL(m_ptr_d[19+c_inc], (double)0.0f);
  BOOST_CHECK_EQUAL(m_ptr_d[9+c_inc], (double)0.9f);
  BOOST_CHECK_EQUAL(m_ptr_d[20+c_inc], (double)0.7f);

}

BOOST_AUTO_TEST_CASE(MaterialHandler_setMaterialIdx) {
  MaterialHandler mh;
  int num_surfaces = 10;
  float coef = 10.f;

  // Add clobal material, so 1 material in the list
  mh.setGlobalMaterial(num_surfaces, coef);

  // check for the 3rd surface, index should be 1 as the indexing starts from 1
  BOOST_CHECK_EQUAL(mh.getMaterialIdxAt(3), 1);

  // Manually change the material index and check
  mh.setMaterialIndexAt(3, 3);
  BOOST_CHECK_EQUAL(mh.getMaterialIdxAt(3), 3);

  // Going out of bounds does nothing except logs the error
  mh.setMaterialIndexAt(12, 3);

}


BOOST_AUTO_TEST_SUITE_END()
