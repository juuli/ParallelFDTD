clear mex
clear data;
addpath(genpath('functions'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assign simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Which update to use is device dependent. SRL forward is most probably
% the most efficient choice. SRL forward sliced might be faster with some
% devices, and when using double precision. SRL centered is a bit of a
% curiosity and definitely has the worst computational performance of the choices.


update_type = 0;                % 0: SRL forward, 1: SRL forward sliced 2: SRL centered 
num_steps = 20000;

% The downside of SRL FDTD scheme which is used here, is the dispersion
% error. The error is well known, and an interested reader should get
% familiar with following publications [3][6]. A 10 time oversampling is 
% commonly used with the SRL scheme that results in ~2 % dispersion error.
% An example, if a 1000 Hz band limit is to be achieved, the sampling
% frequency of the domain should be 10,000 Hz.

% Another downside of SRL scheme is that the simulation domain is a so
% called staircase approximation. This can lead to a slight deviation of the
% modes of the space. An interested reader can see [1].

fs = 40000;


force_partition_to = 2;         % Force the App to use given number of 
                                % partitions, namely the number of devices.
                                % Visualization will override this and use
                                % only one device.

                                
% The selection of precision and the type of source has an effect on the stability
% of the simulation [2]. As a guideline, double precision should be used
% for simulation of impulse responses. For visualization purposes, single precision
% is the only precision supported, and suffices for the purpose of
% visualization of early reflections and scattering. 

double_precision = 0;           % 0: single precision, 1: double precision
source_type = 1;                % 0: hard, 1: soft, 2: transparent
input_type = 3;                 % 0: impulse, 1: gaussian, 2: sine, 3: given input data
input_data_idx = 0;             % In the case of arbitrary input data, the 
                                % input data vector index has to be given
                                % for each source. The input data is loaded
                                % later in this script.
                                
% Source setup. Sources have to be in shape of N X 6, where N is the number
% of sources
src = [1.04, 0.56, 1.17, source_type, input_type, input_data_idx;
       2.24, 0.56, 1.17, source_type, input_type, input_data_idx;];
   
% Receivers, shape N X 3
rec = [1.17, 3.24, 1.5; 
       1.4, 3.24, 1.5];

visualization = 1;             % on: 1 or off: 0

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the model from JSON format
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A JSON importer is used in this example. Any format which is possible
% to import to the workspace can be used. The end product after parsing
% should be a list of vertex coordinates (in meters), and a list of triangle
% indices defining the geometry. Triangle indices have to start from 0.

m = loadjson('./Data/larun_hytti.json')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse the geometry from the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vertices = reshape(m.vertices, [3, length(m.vertices)/3])'; % Vertices = N X 3 matrix
indices = reshape(m.indices, [3, length(m.indices)/3])'; % Triangle indices = Number of triangles x 3 matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assign reflection coefficients to different layers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The materials are given as a [N x 20] matrix where
% N is the number of the polygon/triangle in the geometry.
% The format is slightly misleading, only one value is used from the
% vector, namely the admittance of the selected "octave". The variable octave indicates
% which one of the values in the array is used. For example, value 0 fetches the 
% first value of the material vector at each triangle. 

octave = 0;                                % Lets use the first "octave"

% 20 values is somwhat arbitrary
% number, which rationale is to accomodate 2 variables per 10 octave bands
% to be used in the calculation in future work. 
% Now only one variable is used, which is an admittance value

% First set an uniform material
materials = ones(size(indices, 1), 20)*reflection2admitance(0.99); % Number of triangles x 20 coefficients

% Assign material to a layer
materials(strcmp(m.layers_of_triangles(:), 'ceiling'), :) = reflection2admitance(0.99);
materials(strcmp(m.layers_of_triangles(:), 'floor'), :) = reflection2admitance(0.99);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assign input data for the used sources
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The input data should be in the shape [N, num_steps], where N is the
% number of different input data vectors used in the simulation. Now, 
% two vectors are assigned, first one is a low pass filtered impulse, and
% the second one a unprocessed impulse

src_data = zeros(2, num_steps);
src_data(:, 1) = 1;
b = fir1(100, 0.2);
src_data(1,:)= filter(b,1,src_data(1,:));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assign reflection coefficients to different layers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Captures of slices of the domain in certain time steps
% size is Nx3, where N is the number of captures and the 
% three parameters given are [slice, step, orientation]

slice_capture = [];
mesh_captures = [];

captures = [];

%captures = [100, 1000, 0; ...
%            100, 1000, 1; ...
%            200, 1000, 2]


% Captures of full domains in certain time steps. This funcitonality is
% currently working with a single device only

% mesh_captures = [100];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize and run the FDTD solver 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P_rs = [];

if visualization == 0
[p mesh_r params] = runFDTD(vertices, ... // Vertices
                            indices, ... // Triangle indices
                            materials, ... // Material parameters in a format K X [r1, r2, .... r10, s1, s2, ..... s10]
                            src, ... % sources [N x 6]
                            rec, ...  % receivers [N x 3]
                            src_data, ... % Source input data
                            fs, ... % Sampling frequency of the mesh
                            num_steps, ... % Number of steps to run
                            visualization, ... % Run the visualization
                            captures, ...   % Assigned captures in the formant N X [slice, step, orientation]
                            update_type, ... 
                            mesh_captures, ... % Mesh captures in a format M X [step to be captured]
                            double_precision, ... % 0: single prescision , 1: double precision
                            force_partition_to, ... % Number of partition / devices used
                            octave);  % Which value of material vector is used 
                                      % (in default mode a number between
                                      % [0 , 20])

% The visualization is run on a separate worker for stable OpenGL.
% This is a Windows spesific precaution, on unix based systems the 
% matlabpool commands can be safely commented out.

else
     matlabpool open 1
     spmd
     try
     [p mesh_r params] = runFDTD(vertices, ...
                                indices, ... 
                                materials, ... 
                                src, ... 
                                rec, ...  
                                src_data, ... 
                                fs, ... 
                                num_steps, ... 
                                visualization, ... 
                                captures, ...   
                                update_type, ... 
                                mesh_captures, ...
                                double_precision, ... 
                                force_partition_to, ...
                                octave); 
     catch e
     end
     end
     p = p{:};
     matlabpool close
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Post filter the simulation result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Filter and resample the result                        
P_rs = resample(FDTDpostFilter(double(p), fs, 0.2), 48000, fs);


% [1] Bilbao, S. Modeling of complex geometries and boundary conditions
%     in finite difference/finite volume time domain room acoustics simulation.
%     Audio, Speech, and Language Processing, IEEE Transactions on 21, 7
%     (2013), 1524-1533.

% [2] Botts, J., and Savioja, L. "Effects of sources on time-domain Finite
%     difference models." The Journal of the Acoustical Society of America 136,
%     1 (2014), 242-247.
    
% [3] Kowalczyk, K. "Boundary and medium modelling using compact finite
%     diffrence schemes in simulation of room acoustics for audio and archi-
%     tectural design applications." PhD thesis, School of Electorincs, Electrical
%     Engineering and Computer Science, Queen's University Belfast, 2008.

% [4] Luizard, P., Otani, M., Botts, J., Savioja, L., and Katz, B. F.
%     "Comparison of sound field measurements and predictions in coupled
%     volumes between numerical methods and scale model measurements." In
%     Proceedings of Meetings on Acoustics (2013), vol. 19, Acoustical Society
%     of America, p. 015114.

% [5] Sheaffer, J., Walstijn, M., and Fazenda, B. A "Physically constrained
%     source model for fdtd acoustic simulation." In Proc. of the
%     15th Int. Conference on Digital Audio Effects (DAFx-12) (2012).

% [6] Trefethen, L. N. "Group velocity in finite difference schemes." SIAM review 24, 2
%    (1982), 113-136.

