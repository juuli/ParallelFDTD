# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 10:20:05 2014

@author: Jukka Saarelma

"""

import numpy as np
import matplotlib.pyplot as plt
import json

# The FDTD library is loaded as module
import libPyFDTD as pf 

###############################################################################
# Assign simulation parameters
###############################################################################
update_type = 0 # 0: SRL forward, 1: SRL sliced, 2: SRL centred
num_steps = 2000
fs = 100000
double_precision = False
num_partition = 1

src_type = 0 # 0: Hard, 1: Soft, 2: Transparent
input_type = 1 # 0: Delta, 1: Gaussian, 2: Sine, 3: Given data

src = [0.5, 0.5, 0.5]
 
rec = [[0.6, 0.6, 0.6],
       [0.4, 0.4, 0.4]]

visualization = False
captures = True

###############################################################################
# Load the model from JSON format
###############################################################################

# A JSON importer is used in this case, any format which is possible
# to import a geometry to the workspace can be used. 
# The end product after parsing should be a list of vertices coordinates 
# (in meters), and a list of triangle indices defining the geometry. 
# Triangle indices have to start from 0.

fp = "./jsaarelm_data/box.json"
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

num_triangles = np.size(indices)/3
num_coef = 20 # Default number of coefficients

R_glob = 0.99
materials = np.ones((num_triangles, num_coef))*reflection2Admittance(R_glob)

# Grab the triangle indices of the given layer from the 'layers' list. 
# Assign a material to those triangles in the material list
materials[layers['walls'], :] = reflection2Admittance(R_glob)
materials[layers['ceiling'], :] = reflection2Admittance(R_glob)
materials[layers['floor'], :] = reflection2Admittance(R_glob)

###############################################################################
# Assign image captures to the simulations
###############################################################################

# Captures of slices of the domain in certain time steps
# size is Nx3, where N is the number of captures and the 
# three parameters given are [slice, step, orientation]

slice_n = 60
step = 100
orientation = 1
capture = [slice_n, step, orientation]

###############################################################################
# Initialize and run the FDTD solver 
###############################################################################
app = pf.App()
app.initializeDevices()
app.initializeGeometryPy(indices.flatten().tolist(), vertices.flatten().tolist())
app.setUpdateType(update_type)
app.setNumSteps(int(num_steps))
app.setSpatialFs(fs)
app.setDouble(double_precision)
app.forcePartitionTo(num_partition);
app.addSurfaceMaterials(materials.flatten().tolist(), num_triangles, num_coef)

app.addSource(src[0], src[1], src[2], src_type, input_type)

for i in range(0, np.shape(rec)[0]):
  app.addReceiver(rec[i][0],rec[i][1],rec[i][2])

if visualization:
    app.runVisualization()
elif captures:
    app.addSliceToCapture(capture[0], capture[1], capture[2])
    app.runCapture()
else:
    app.runSimulation()

###############################################################################
# Parse the return values
###############################################################################

ret = []
for i in range(0, np.shape(rec)[0]):
  if double_precision:
    ret.append(np.array(app.getResponseDouble(i)))
  else:
    ret.append(np.array(app.getResponse(i)))

# Cast to Numpy array for convenient slicing
ret = np.transpose(np.array(ret))
plt.plot(ret)

# Remember to close and delete the solver after you're done!
app.close()
del app
