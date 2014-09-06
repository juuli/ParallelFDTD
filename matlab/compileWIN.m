clear mex
fprintf('Compiling  mex_FDTD\n');

% Specify include and library paths
boostPath = 'C:\Program Files\boost\boost_1_55_0';
boostLibPath = 'C:\Program Files\boost\boost_1_55_0\lib\';
cudaInclude = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\include';
cudaLibPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\lib\x64';
glewLibPath = 'C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\lib\amd64';
additionalLibPath =  './lib/';

% Specify linker flags for different libraries
cudaLibraries =  '-lcudart -lcuda ';
glutAndGLlibraries = '-lfreeglut ';
glewLibrary = '-lglew32 ';
boostLibraries = ['-llibboost_thread-vc110-mt-1_55 -llibboost_system-vc110-mt-1_55 ', ...
                 '-llibboost_chrono-vc110-mt-1_55 -llibboost_date_time-vc110-mt-1_55 '];
mexSpesificLibraries = '-llibmx -llibut ';
FDTDlibraries = '-llibParallelFDTD -lVoxelizer';

% Whole link command
link_flags = [cudaLibraries, ...
              glutAndGLlibraries, ...
              glewLibrary, ...
              boostLibraries, ...
              mexSpesificLibraries, ...
              FDTDlibraries
              ];

compile_command = ['mex -I"' boostPath '" ', ...
                  ' -I"' cudaInclude  '"', ...
                  ' -L"' additionalLibPath '" ', ...
                  ' -L"' cudaLibPath '" ',...
                  ' -L"' boostLibPath '" ',...
                  ' -L"' additionalLibPath '" ', ...
                  link_flags]

% Compile
eval([compile_command ' mex_FDTD.cpp']);
eval([compile_command ' device_reset.cpp ']);
eval([compile_command ' mem_check.cpp ']);

fprintf('Compile done\n');
clear all