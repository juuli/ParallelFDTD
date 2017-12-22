clear mex
fprintf('Compiling  mex_FDTD\n');

% Specify include and library paths
boostPath = 'C:/Users/jks/Documents/boost_1_63_0' 
boostLibPath =  [boostPath '\stage\lib'] 
cudaInclude = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include';
cudaLibPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64';
glewLibPath = 'C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\lib\amd64';
additionalLibPath =  './lib/';

% Specify linker flags for different libraries
cudaLibraries =  '-lcudart -lcuda ';
glutAndGLlibraries = '-lfreeglut ';
glewLibrary = '-lglew32 ';
boostLibraries = ['-llibboost_thread-vc140-mt-1_63 -llibboost_system-vc140-mt-1_63 ', ...
                 '-llibboost_chrono-vc140-mt-1_63 -llibboost_date_time-vc140-mt-1_63 '];
mexSpesificLibraries = '-llibmx -llibut ';
FDTDlibraries = ' -llibParallelFDTD -lVoxelizer ';

% Whole link command
link_flags = [FDTDlibraries, ...
              cudaLibraries, ...
              glutAndGLlibraries, ...
              glewLibrary, ...
              boostLibraries, ...
              mexSpesificLibraries, ...
              ];

compile_command = ['mex -DCOMPILE_VISUALIZATION -I"' boostPath '" ', ...
                  ' -I"' cudaInclude  '"', ...
                  ' -L"' additionalLibPath '" ', ...
                  ' -L"' cudaLibPath '" ',...
                  ' -L"' boostLibPath '" ',...
                  ' -L"' additionalLibPath '"', ...
                  link_flags]

% Compile
eval([compile_command ' mex_FDTD.cpp']);
eval([compile_command ' device_reset.cpp ']);
eval([compile_command ' mem_check.cpp ']);

fprintf('Compile done\n');
clear all