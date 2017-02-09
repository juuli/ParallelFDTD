# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 10:20:05 2014

@author: Jukka Saarelma

"""

import numpy as np
import scipy.signal as sig
import scipy.weave as weave
from scipy.weave import converters
import h5py

################################################################################
# Assign simulation parameters
###############################################################################
dx = 0.01
fs = 1.0/dx*344.0*np.sqrt(3.0)
c=344.0
lamb = 1.0/np.sqrt(3.0)

# Simulation parameters
update_type = 0 # 0: SRL forward, 1: SRL sliced, 2: SRL centred, 3: FCC beta
double_precision = False
num_partitions = 1
num_steps = int(fs*0.015)


src_type = 0 # 0: Hard, 1: Soft, 2: Transparent
input_type = 3 # 0: Delta, 1: Gaussian, 2: Sine, 3: Given data

src = np.array([[1.6, 0.5, 1.5]])/dx

rec = []
for i in range(10):
  rec.append(np.array([2.1, 2.6, 0.4+0.2*i])/dx)

src_data = np.zeros(num_steps, dtype=np.float32)
src_data[:200] = sig.firwin(200, 0.1, window=('chebwin', 60))

captures = False


def evaluateBoundariesWeave(pos, switch_ptr, beta, admittance):
  dim_x, dim_y, dim_z = np.shape(pos)

  evaluate= """
    unsigned char* pos_  = (unsigned char*)pos_array->data;
    unsigned char* switch_ = (unsigned char*)switch_ptr_array->data;

    unsigned int dim_xy = dim_x*dim_y;
    unsigned char count = 0;
    int dim_z_ = dim_z-1;
    for(int i = 1; i < dim_x-1; i ++) {
      for(int j = 1; j < dim_y-1; j++) {
        for(int k = 1; k < dim_z_; k++) {
          int idx = i*dim_xy+j*dim_x+k;

          count = (switch_[idx+1]+switch_[idx-1]+
                   switch_[idx+dim_x]+switch_[idx-dim_x]+
                   switch_[idx+dim_xy]+switch_[idx-dim_xy]);

          pos_[idx] = count;
        }
      }
    }
  """

  weave.inline(evaluate, ['dim_x', 'dim_y', 'dim_z', 'pos',
                          'beta', 'switch_ptr', 'admittance'],
                           extra_compile_args=['-O3 ' ],
                           type_converters = converters.blitz,
                           support_code = \
                            r"""
                            #include <stdio.h>
                            #include <math.h>
                            """)

###############################################################################
# Init domiain locally
dim = int(6.0/dx) # Size float ~((((32*31)**3)*10)/1e9) = 9.76 G

mat_vec = np.zeros((dim,dim,dim), dtype=np.uint8)

# Python array is row major -> x and z switch places for receivers/sources
# pos_vec[1, :, :] = 5+128
# pos_vec[0, :, :] = 0
mat_vec[1, :, :] = 1
materials = np.zeros((20, 2))

pos_vec = np.zeros((dim, dim, dim), dtype = np.uint8)
mat_idx = np.zeros((dim, dim, dim), dtype=np.uint8)
mat_coef = np.zeros((2, 20), dtype=np.float64)

switch_ptr = np.ones((dim, dim, dim), dtype = np.uint8)
switch_ptr[:,:,0] = 0; switch_ptr[:,:,-1] = 0
switch_ptr[:,0,:] = 0; switch_ptr[:,-1,:] = 0
switch_ptr[0,:,:] = 0; switch_ptr[-1:,:,:] = 0

diag_points = []


for i in range(dim):
  for j in range(dim):
    for k in range(dim):
      if(k+j==dim-i):
        switch_ptr[i,j,k] = 0
        #diag_points.append(np.array([i,j,k]))

#diag_points = np.array(diag_points)

evaluateBoundariesWeave(pos_vec, switch_ptr, 0, 0.0)
pos_vec[switch_ptr==0] = 0
pos_vec[pos_vec>0] += 128

del switch_ptr

###############################################################################
# Init source position in terms of elements (as a multiple of dx)

sr_m = 0.2/np.sqrt(2.0)
sr=int(sr_m/dx)
src_pos = []

for j in range(-sr, sr):
  for k in range(-sr, sr):
    x_incr = -0.5*j
    y_incr = -j-x_incr
    if j%2==0:
      y_incr+=0.5
      x_incr-=0.5

    y_incr+=0.5
    x_incr+=0.5
    src_pos.append(np.array([dim/3+k+x_incr, dim/3-k+y_incr, dim/3+j+1]))

print "Src append done"
src_type = 1 # 0: Hard, 1: Soft, 2: Transparent
input_type = 3 # 0: Delta, 1: Gaussian, 2: Sine, 3: Given data
src_d = np.zeros(num_steps)
src_d[0] = 1.0

srcwin = sig.firwin(200, 0.2, window=('chebwin', 60))
src_d[:200] = srcwin

def rotation_matrix(axis, theta):
  """
  Return the rotation matrix associated with counterclockwise rotation about
  the given axis by theta radians.
  """
  axis = np.asarray(axis)
  axis = axis/np.sqrt(np.dot(axis, axis))
  a = np.cos(theta/2.0)
  b, c, d = -axis*np.sin(theta/2.0)
  aa, bb, cc, dd = a*a, b*b, c*c, d*d
  bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
  return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                   [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                   [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


rot_1 = rotation_matrix([0, 0, 1], 45.0/180.0*np.pi)
rot_2 = rotation_matrix([np.sqrt(2.0), -np.sqrt(2.0), 0], np.arcsin(1/np.sqrt(3)))

n_rec = 20
el = np.linspace(0.0, np.pi/2.0, n_rec)

rr = 2
r = rr/dx
rec = []

for i in range(n_rec):
  point = [r*np.sin(el[i]),
           np.cos(el[i])*r,#*np.cos(np.pi/4.0*0.0)*r)),
           np.cos(el[i])*r*0.0]#*np.cos(np.pi/4.0*0.0)*r))]

  point = np.dot(rot_1, point)
  point = np.dot(rot_2, point)

  rec.append(np.array(point))

rec = np.round(np.array(rec))

diag_inc = np.round((1.0/3.0*dim+1))

rec += np.array([diag_inc,
                 diag_inc,
                 diag_inc])


print "Rec append done"
rec = np.array(rec, dtype = np.float32)*dx
src = np.array(src_pos, dtype = np.float32)*dx

###############################################################################
# Assign reflection coefficients to different layers
###############################################################################

# The materials are given as a [N x 20] matrix where
# N is the number of polygon in the geometry

# The solver takes admittance values
def reflection2Admittance(R):
    return (1.0-R)/(1.0+R)

def absorption2Admittance(alpha):
    return reflection2Admittance(np.sqrt(1.0-alpha))

def reflection2Absorption(R):
  return 1-R**2

num_coef = 20 # Default number of coefficients

R_glob = 1.0
materials = np.ones((1, num_coef))*reflection2Admittance(R_glob)
materials = materials.astype(np.float32)


###############################################################################
# Write HDF5 file from the parameters
###############################################################################

N_x_CUDA = 32
N_y_CUDA = 4
N_z_CUDA = 1

f = h5py.File('dataset.hdf5', 'w')

f["fs"] = np.array([fs], dtype=np.float64)
f["dX"] = np.array([dx], dtype=np.float64)
f["c_sound"] = np.array([c], dtype=np.float64)
f["lambda_sim"] = np.array([lamb], dtype=np.float64)
f["CUDA_steps"] = np.array([num_steps], dtype=np.int32)
f["N_x_CUDA"] = np.array([N_x_CUDA], dtype=np.int32)
f["N_y_CUDA"] = np.array([N_y_CUDA], dtype=np.int32)
f["N_z_CUDA"] = np.array([N_z_CUDA], dtype=np.int32)
f["GPU_partitions"] = np.array([num_partitions], dtype=np.int32)
f["double_precision"] = np.array([int(double_precision)], dtype=np.int32)

f["num_rec"] = np.array([np.shape(rec)[0]], dtype=np.int32)
f["num_src"] = np.array([np.shape(src)[0]], dtype=np.int32)
f["source_type"] = np.array([src_type], dtype=np.float32)
f["dim_x"] = np.array([dim], dtype=np.int32)
f["dim_y"] = np.array([dim], dtype=np.int32)
f["dim_z"] = np.array([dim], dtype=np.int32)

rec = np.round(np.array(rec, dtype=np.float32).flatten())
src = np.round(np.array(src, dtype=np.float32).flatten())


f.create_dataset("rec_coords", data=rec, dtype=np.float32)#
f.create_dataset("src_coords", data=src, dtype=np.float32)#
f.create_dataset("src_data", data=src_data, dtype=np.float32)#
f.create_dataset("h_pos", data=pos_vec, dtype=np.uint8, compression="gzip")
f.create_dataset("h_mat", data=mat_idx, dtype=np.uint8, compression="gzip")
f["mat_coef_vect_len"] = np.array([np.size(materials)], dtype=np.int64)
f["num_mat_coefs"] = np.array([num_coef], dtype=np.int64)


f.create_dataset("material_coefficients", data=materials, dtype=np.float32)#

f.close()

print "File written and closed"
