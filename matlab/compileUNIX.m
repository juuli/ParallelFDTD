clear mex
fprintf('Compiling  mex_FDTD\n');

% Specify include and library paths
boostPath = '/usr/local/include';
boostLibPath = '/usr/local/lib';
glutIncludePath = '/usr/include'
cudaInclude = '/usr/local/cuda-5.5/include';
cudaLibPath = '/usr/local/cuda-5.5/lib64';
additionalLibPath =  './lib/';


% Specify linker flags for different libraries
cudaLibraries =  '-lcudart -lcuda ';
glutAndGLlibraries = '-lglut -lGLU -lGL ';
glewLibrary = '-l"/usr/lib/x86_64-linux-gnu/libGLEW.so" ';
boostLibraries = '-lboost_thread-mt -lboost_system-mt -lboost_date_time-mt ';
mexSpesificLibraries = '-lmx -lut ';
FDTDlibraries = '-llibParallelFDTD -lVoxelizer';

% Whole link command
link_flags = [cudaLibraries, ...
              glutAndGLlibraries, ...
              glewLibrary, ...
              boostLibraries, ...
              mexSpesificLibraries, ...
              FDTDlibraries
              ];
          
compile_command = ['mex -I"' glutIncludePath '" ', ...
                  ' -I"' boostPath  '"', ...
                  ' -I"' cudaInclude  '"', ...
                  ' -L"' additionalLibPath '" ', ...
                  ' -L"' cudaLibPath '" ',...
                  ' -L"' boostLibPath '" ',...
                  ' -L"' additionalLibPath '" ', ...
                  link_flags]

% Compile
eval([compile_command ' device_reset.cpp ']);
eval([compile_command ' mex_FDTD.cpp']);

fprintf('Compile done\n');
clear all