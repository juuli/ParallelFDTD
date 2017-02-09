# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 10:20:05 2014

@author: Jukka Saarelma

"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import json
import h5py

################################################################################
# Assign simulation parameters
###############################################################################
update_type = 0 # 0: SRL forward, 1: SRL sliced, 2: SRL centred
double_precision = True
num_partitions = 1
voxelization_type = 2 # 0: solid, 1: 6 separating, 2: conservative surf


dx = 0.0025
fs = 1.0/dx*344.0*np.sqrt(3.0)
c=344.0
lamb = 1.0/np.sqrt(3.0)
num_steps = fs*0.5


src_type = 0 # 0: Hard, 1: Soft, 2: Transparent
input_type = 3 # 0: Delta, 1: Gaussian, 2: Sine, 3: Given data

src = np.array([[1.6, 0.5, 1.5]])/dx

rec = []
for i in range(10):
  rec.append(np.array([2.1, 2.6, 0.4+0.2*i])/dx)

src_data = np.zeros(num_steps, dtype=np.float32)
src_data[:200] = sig.firwin(200, 0.1, window=('chebwin', 60))
visualization = True
captures = False

###############################################################################
# Load the model from JSON format
###############################################################################

# A JSON importer is used in this case, any format which is possible
# to import a geometry to the workspace can be used.
# The end product after parsing should be a list of vertices coordinates
# (in meters), and a list of triangle indices defining the geometry.
# Triangle indices have to start from 0.

fp = "./Data/larun_hytti.json"
file_stream = open(fp)
m = json.load(file_stream)
file_stream.close()

###############################################################################
# Parse the geometry from the data
###############################################################################
vertices = np.reshape(m["vertices"], (np.size(m["vertices"])/3, 3))
indices = np.reshape(m["indices"], (np.size(m["indices"])/3, 3))

###############################################################################
# Get the layer list, enumerate the surfaces on each layer
###############################################################################
layer_list = m["layers_of_triangles"]
layer_names = m["layer_names"]
layers = {}

for k in range(0, len(layer_names)):
  layer_indices = [i for i, j in enumerate(layer_list) if j == layer_names[k]]
  layers[layer_names[k]] = layer_indices

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

num_triangles = np.size(indices)/3
num_vertices = np.size(vertices)/3
num_coef = 20 # Default number of coefficients

R_glob = 0.1
materials = np.ones((num_triangles, num_coef))*reflection2Admittance(R_glob)
materials = materials.astype(np.float32)


# Grab the triangle indices of the given layer from the 'layers' list.
# Assign a material to those triangles in the material list

#materials[layers['walls'], :] = reflection2Admittance(R_glob)
#materials[layers['ceiling'], :] = reflection2Admittance(R_glob)
materials[layers['ceiling'], :] = absorption2Admittance(0.05)
materials[layers['floor'], :] = absorption2Admittance(0.9)
materials[layers['window'], :] = absorption2Admittance(0.3)

###############################################################################
# Assign image captures to the simulations
###############################################################################

# Captures of slices of the domain in certain time steps
# size is Nx3, where N is the number of captures and thereflection
# three parameters given are [slice, step, orientation]

slice_n = 128
step = 1
orientation = 1
capture = [slice_n, step, orientation]

###############################################################################
# Write HDF5 file from the parameters
###############################################################################

N_x_CUDA = 32
N_y_CUDA = 4
N_z_CUDA = 1

f = h5py.File('dataset.hdf5', 'w')

f["num_triangles"] =  np.array([num_triangles], dtype=np.int32)
f["num_vertices"] = np.array([num_vertices], dtype=np.int32)

f.create_dataset("vertices", data=vertices.flatten(), dtype=np.float32)#
f.create_dataset("triangles", data=indices.flatten(), dtype=np.uint32)#

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

rec = np.round(np.array(rec, dtype=np.float32).flatten())
src = np.round(np.array(src, dtype=np.float32).flatten())


f.create_dataset("rec_coords", data=rec, dtype=np.float32)#
f.create_dataset("src_coords", data=src, dtype=np.float32)#
f.create_dataset("src_data", data=src_data, dtype=np.float32)#

f["mat_coef_vect_len"] = np.array([np.size(materials)], dtype=np.int64)
f["num_mat_coefs"] = np.array([num_coef], dtype=np.int64)

f["voxelization_type"] = voxelization_type

f.create_dataset("material_coefficients", data=materials, dtype=np.float32)#

f["mesh_file"] = np.string_(fp)

f.close()

print "File written and closed"
