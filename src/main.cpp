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
    app.initializeGeometryFromFile("./Data/hytti.vtk");
    app.m_materials.setGlobalMaterial(app.m_geometry.getNumberOfTriangles(), reflection2Admitance(0.9f));
    app.m_parameters.setSpatialFs(11000);
    app.m_parameters.setNumSteps(300);
    app.m_parameters.setUpdateType(SRL_FORWARD);
    app.m_parameters.readGridIr("./Data/grid_ir.txt");
    app.m_parameters.addSource(Source(12.5f, 2.5f,20.5f, SRC_HARD, GAUSSIAN, 1));
    app.runVisualization();
  }
  catch(...) {
  
  }
  app.close();
  char c;
  std::cin>>c;
  return 0;
}
