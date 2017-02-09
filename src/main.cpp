///////////////////////////////////////////////////////////////////////////////
//
// This file is a part of the PadallelFDTD Finite-Difference Time-Domain
// simulation library. It is released under the MIT License. You should have
// received a copy of the MIT License along with ParallelFDTD.  If not, see
// http://www.opensource.org/licenses/mit-license.php
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// For details, see the LICENSE file
//
// (C) 2013-2014 Jukka Saarelma
// Aalto University School of Science
//
///////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <signal.h>
#include <iostream>
#include <fstream>
#include "App.h"
#include "global_includes.h"

static volatile bool interrupt = false;

int main(int argc, char** argv){
  loggerInit();
  log_msg<LOG_INFO>(L"Main: begin");
  FDTD::App app;

  // The app will throw -1 on error
  try {
    app.initializeDevices();
    app.initializeGeometryFromFile("./data/box_1175.vtk");
    app.m_materials.setGlobalMaterial(app.m_geometry.getNumberOfTriangles(), reflection2Admitance(0.9f));
    app.m_parameters.setSpatialFs(21000);
    app.m_parameters.setNumSteps(300);
    app.m_parameters.setUpdateType(SRL_FORWARD);
    app.m_parameters.setVoxelizationType(SURFACE_6);
    app.setForcePartitionTo(1);
    app.setDouble(false);
    // app.addSliceToCapture(40, 3, 0);
    app.m_parameters.addSource(Source(5.5f, 2.5f,1.5f, SRC_HARD, GAUSSIAN, 1));

    for(int i = 1; i < 10; i++) {
      app.m_parameters.addReceiver(i*0.2,i*0.2,i*0.2);
      nv::Vec3i rec = app.m_parameters.getReceiverElementCoordinates(i-1);
      std::cout<<"Receiver at "<<i<<" x: "<<rec.x<<" y: "<<rec.y<<" z: "<<rec.z<<std::endl;
    }


    #ifdef COMPILE_VISUALIZATION
      app.runVisualization();
    #else
      app.runSimulation();
    #endif
  }
  catch(...) {

  }
  app.close();

  return 0;
}
