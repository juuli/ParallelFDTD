#include "mex.h"
#include "mex_includes.h"
#include <signal.h>


FDTD::App app;

typedef void (*SignalHandlerPointer)(int);
SignalHandlerPointer prev_handler;

void sigAbortHandler(int signum) {
    printf("Inside handler\n");
    if(signum == SIGINT)
        app.close();
}

extern "C" {
void mexProgressCallback(int step, int max_step, float t_per_step ){
	float estimate = t_per_step*(float)(max_step-step);
    mexPrintf("Step %d/%d, time per step %f, estimated time left %f s \n", 
           step, max_step, t_per_step, estimate);
    mexEvalString("drawnow;");
    return;
}
}
// Matlabs interrupt query function
extern "C" {
    bool utIsInterruptPending();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
           
    // Return value pointers
    float *ret_ptr1;
    double *ret_ptr1_double;
    
    float *ret_ptr2;
    float *ret_ptr3;
    float *ret_ptr4;
    float *ret_ptr5;
    float *ret_ptr6;
    float *ret_ptr7;
    float *ret_ptr8;
    
    
    // Input variables       
    float *vertices = (float*)NULL;
    unsigned int *indices = (unsigned int*)NULL;
    float *materials = (float*)NULL;
    double *input_data = (double*)NULL;
    unsigned int number_of_input_samples = 0;
    unsigned int number_of_input_data_vectors = 0;
    unsigned int number_of_surfaces = 0;
    unsigned int number_of_coefficients = 0;
    unsigned int number_of_steps = 0;
    unsigned int number_of_sources = 0;
    unsigned int number_of_receivers = 0;
    
    // Source and receiver positions
    float *src_list;
    float *rec_list;
    // Simulation parameters
    float *parameters;
    unsigned int spatial_fs, size_of_vertices, size_of_indices, visualization, update_type;
    unsigned int number_of_captures;
    unsigned int* capture_instructions = (unsigned int*)NULL;
    unsigned int number_of_mesh_captures;
    unsigned int* mesh_capture_instructions = (unsigned int*)NULL;
    unsigned int* double_precision = (unsigned int*)NULL;
    unsigned int* force_partition_to = (unsigned int*)NULL;
    unsigned int* octave = (unsigned int*)NULL;
    
    /////////// Parse input argumets
    // Input is a struct containing....
    // - geometry (raw data) 
    // - materials per polygon
    // - source/receiver   
    //      - input data
    //      - source types and groups
    // - run parameters
    // - capture instructions
    
    mexPrintf("_________________________________\n");
    mexPrintf("Parse input arguments\n");
    
    if(nrhs == 15) {
        // Vertices
        size_of_vertices = mxGetNumberOfElements(prhs[0]); 
        vertices = (float*)mxGetData(prhs[0]);
        mexPrintf("Number of vertices : %u \n", size_of_vertices/3); 
        
        // Indices
        size_of_indices = mxGetNumberOfElements(prhs[1]);
        indices = (unsigned int*)mxGetData(prhs[1]);
        mexPrintf("Number of triangles : %u \n", size_of_indices/3); 
        
        if(size_of_indices/3 == 0 || size_of_vertices/3 == 0) {
            mexPrintf("No geometry assigned, check the geometry file, returning\n");
            return;
        }
    
        // Materials
        unsigned int size_of_materials = mxGetNumberOfElements(prhs[2]);
        number_of_surfaces = mxGetN(prhs[2]);
        mexPrintf("Number of surfaces in materials : %u \n", number_of_surfaces);
        
        number_of_coefficients = mxGetM(prhs[2]);
        materials = (float*)mxGetData(prhs[2]);
        mexPrintf("Size of materials : %u \n", size_of_materials); 
        
        // Sources
        number_of_sources = mxGetN(prhs[3]);
        src_list = (float*)mxGetData(prhs[3]);
        
        // Receivers
        number_of_receivers = mxGetN(prhs[4]);
        rec_list = (float*)mxGetData(prhs[4]);
         
        // Source Data
        number_of_input_data_vectors = mxGetN(prhs[5]);
        number_of_input_samples =  mxGetM(prhs[5]);
        input_data = (double*)mxGetData(prhs[5]);
        
        // Parameters
        spatial_fs = *((unsigned int*)mxGetData(prhs[6]));
        number_of_steps = *((unsigned int*)mxGetData(prhs[7]));
        
         // Update_type
        update_type = *((unsigned int*)mxGetData(prhs[8]));
        visualization = *((unsigned int*)mxGetData(prhs[9]));
       
        // Slice Capture data
        number_of_captures = mxGetN(prhs[10]);
       	capture_instructions = (unsigned int*)mxGetData(prhs[10]);
        
        // Mesh Captures
        number_of_mesh_captures = mxGetN(prhs[11]);
       	mesh_capture_instructions = (unsigned int*)mxGetData(prhs[11]);
        
        unsigned int i; 
        for(i = 0; i < size_of_vertices; i++) {
         //printf("%f \n", vertices[i]);   
        }
         
        for(i = 0; i < size_of_indices; i++) {
         //printf("%u \n", indices[i]);   
        }
         
        for(i = 0; i < size_of_materials; i++) {
         //printf("%f \n", materials[i]);   
        }
        
        mexPrintf("Number of captures %d \n", number_of_captures);
        
        for(i = 0; i < number_of_captures; i++) {   
         mexPrintf("slice %u, step %u dim %u \n", capture_instructions[i*3], capture_instructions[i*3+1], capture_instructions[i*3+2]);   
        }
        
        mexPrintf("Number of sources %d \n", number_of_sources);
        mexPrintf("Number of receivers %d \n", number_of_receivers);
        mexPrintf("Number of input data samples %d \n", number_of_input_samples);
        
        double_precision = (unsigned int*)mxGetData(prhs[12]);
        
        if(visualization == 1) {
            mexPrintf("Visualization selected, single precision forced\n");
            *double_precision = 0;
        }
        
        force_partition_to = (unsigned int*)mxGetData(prhs[13]);
        octave = (unsigned int*)mxGetData(prhs[14]);
    }
    else {
        mexErrMsgTxt("Not enough input argumets");
    }
    
    ///////////////////////////////////////
    // Initialize solver and run
    mexPrintf("_________________________________\n");
    mexPrintf("Init and run\n");
    mexEvalString("drawnow;"); 
    loggerInit();
	log_msg<LOG_INFO>(L"Mex: begin");
    
    prev_handler = signal(SIGINT, sigAbortHandler);
	app.initializeDevices();
    app.m_interrupt = (InterruptCallback)utIsInterruptPending;
    app.m_progress = (ProgressCallback)mexProgressCallback;
	app.initializeGeometry(indices, vertices, size_of_indices, size_of_vertices);
    app.m_materials.addMaterials(materials, number_of_surfaces, number_of_coefficients);
    app.m_parameters.setSpatialFs(spatial_fs);
	app.m_parameters.setNumSteps(number_of_steps);
    app.m_parameters.readGridIr("./Data/grid_ir.txt");
    app.m_parameters.setUpdateType((enum UpdateType)update_type);
    app.m_parameters.setOctave(*octave);
    app.setForcePartitionTo((int)(*force_partition_to));
   
    for(int i = 0; i < number_of_input_data_vectors; i++) {
        if(*double_precision == 1) {
            std::vector<double> input_data_vec(number_of_input_samples, 0.0);
            for(int j = 0; j < number_of_input_samples; j++) {
                int idx = j+i*number_of_input_samples;
                input_data_vec.at(j) = input_data[j+i*number_of_input_samples];
            }
            app.m_parameters.addInputDataDouble(input_data_vec);
        }
        else {
            std::vector<float> input_data_vec(number_of_input_samples, 0.f);
            for(int j = 0; j < number_of_input_samples; j++) {
                input_data_vec.at(j) = (float)input_data[j+i*number_of_input_samples];
            }
            app.m_parameters.addInputData(input_data_vec);   
        }
    }
    
    for(unsigned int i = 0; i < number_of_sources; i++) {
        unsigned int idx = i*6;
        app.m_parameters.addSource(Source(src_list[idx], src_list[idx+1], src_list[idx+2], 
                                  (enum SrcType)((unsigned int)src_list[idx+3]), 
                                  (enum InputType)((unsigned int)src_list[idx+4]), src_list[idx+5]));
    }
    
    for(unsigned int i = 0; i < number_of_receivers; i++) {
        unsigned int idx = i*3;
        app.m_parameters.addReceiver(Receiver(rec_list[idx], rec_list[idx+1], rec_list[idx+2]));
    }
	
    for(unsigned int i = 0; i < number_of_captures; i++) {
        app.addSliceToCapture(capture_instructions[i*3], capture_instructions[i*3+1], capture_instructions[i*3+2]);
    }
    
    for(unsigned int i = 0; i < number_of_mesh_captures; i++){
        app.addMeshToCapture(mesh_capture_instructions[i]);
    }

    try {
    if(visualization) {  
        app.runVisualization();
    }
    else {
        if(number_of_captures == 0 && number_of_mesh_captures == 0){
            if(*double_precision == 1)
                app.m_mesh.setDouble(true);
            app.runSimulation();
        }
        else
            app.runCapture();
    }
    }
    catch(...) {
       mexPrintf("Execption catched, closing application\n");
       app.close();
       return;
    }
    //////////////////////////////////////
    // Parse output arguments
    
    mexPrintf("_________________________________\n");
    mexPrintf("Parsing Output Arguments\n");
    
    if (nlhs == 1 && app.getResponseSize() != 0) {
        unsigned int n_steps = app.m_parameters.getNumSteps();
        plhs[0] = mxCreateNumericMatrix(number_of_receivers, n_steps, mxSINGLE_CLASS, mxREAL);
        ret_ptr1 = (float*)mxGetData(plhs[0]);
        
        for(unsigned int i = 0; i < n_steps; i++) {
            for(unsigned int j = 0; j < number_of_receivers; j++) {
            ret_ptr1[i*(number_of_receivers)+j] = app.getResponseSampleAt(i, j);
            
            }
        }
    }
    
     if (nlhs == 3 && app.getResponseSize() != 0) {
        printf("3 arguments\n");
        unsigned int n_steps = app.m_parameters.getNumSteps();
        plhs[0] = mxCreateNumericMatrix(number_of_receivers, n_steps, mxSINGLE_CLASS, mxREAL);
        plhs[1] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        plhs[2] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        ret_ptr1 = (float*)mxGetData(plhs[0]);
        ret_ptr2 = (float*)mxGetData(plhs[1]);
        ret_ptr3 = (float*)mxGetData(plhs[2]);
       
        for(unsigned int i = 0; i < n_steps; i++) {
            for(unsigned int j = 0; j < number_of_receivers; j++) {
            ret_ptr1[i*(number_of_receivers)+j] = app.getResponseSampleAt(i, j);
            
            }
        }
        
        ret_ptr2[0] = (float)app.getNumElements();
        ret_ptr3[0] = app.getTimePerStep();
    }
     // If 4 arguments, 4th is the captured meshes
     if (nlhs == 8) {
        unsigned int n_steps = app.m_parameters.getNumSteps();
        unsigned int n_mesh_captures = app.getNumberOfMeshCaptures();
        unsigned int n_elements = app.m_mesh.getNumberOfElements();
        
      
        plhs[1] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        plhs[2] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        plhs[3] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        plhs[4] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        plhs[5] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        plhs[6] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        plhs[7] = mxCreateNumericMatrix(n_mesh_captures, n_elements, mxSINGLE_CLASS, mxREAL);
        
        ret_ptr2 = (float*)mxGetData(plhs[1]);
        ret_ptr3 = (float*)mxGetData(plhs[2]);
        ret_ptr4 = (float*)mxGetData(plhs[3]);
        ret_ptr5 = (float*)mxGetData(plhs[4]);
        ret_ptr6 = (float*)mxGetData(plhs[5]);
        ret_ptr7 = (float*)mxGetData(plhs[6]);
        ret_ptr8 = (float*)mxGetData(plhs[7]);
        
        if(app.m_mesh.isDouble()) {
              plhs[0] = mxCreateNumericMatrix(number_of_receivers, n_steps, mxDOUBLE_CLASS, mxREAL);
              ret_ptr1_double = (double*)mxGetData(plhs[0]);
              
        }
        else {
            plhs[0] = mxCreateNumericMatrix(number_of_receivers, n_steps, mxSINGLE_CLASS, mxREAL);
            ret_ptr1 = (float*)mxGetData(plhs[0]); 
        }
         
        ret_ptr2[0] = (float)app.getNumElements();
        ret_ptr3[0] = app.getTimePerStep();
        ret_ptr4[0] = app.m_mesh.getDimX();
        ret_ptr5[0] = app.m_mesh.getDimY();
        ret_ptr6[0] = app.m_mesh.getDimZ();
        ret_ptr7[0] = app.m_parameters.getDx();
        
        
        if(app.getResponseSize() != 0) {
        for(unsigned int i = 0; i < n_steps; i++) {
            for(unsigned int j = 0; j < number_of_receivers; j++) {
                if(app.m_mesh.isDouble())
                    ret_ptr1_double[i*(number_of_receivers)+j] = app.getResponseDoubleSampleAt(i, j);
                else
                    ret_ptr1[i*(number_of_receivers)+j] = app.getResponseSampleAt(i, j);
            }
        }
        }
        else {
            printf("else\n");
            plhs[0] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
            plhs[1] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
            plhs[2] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
            ret_ptr1 = (float*)mxGetData(plhs[0]);
            ret_ptr2 = (float*)mxGetData(plhs[0]);
            ret_ptr3 = (float*)mxGetData(plhs[0]);
            ret_ptr1[0] = 0.f;
            ret_ptr2[0] = 0.f;
            ret_ptr3[0] = 0.f;
        }
        
         for(unsigned int i = 0; i < n_mesh_captures; i ++) {
            float* mesh_capture = app.getMeshCaptureAt(i);
            for(unsigned int j = 0; j < n_elements; j++) {
              ret_ptr8[j*n_mesh_captures+i] = *(mesh_capture+j);  
            }
        }
     } // End output parsing
    app.close();
}

