function  [responses return_mesh return_params] = runFDTD(vertices, indices, materials, src, rec, src_data, fs, ...
                                                          num_steps, visualization, capture_slice_step, update_type, ...
                                                          mesh_captures, double_precision, force_partition_to, octave)

%[responses return_mesh return_params] = runFDTD(vertices, indices, materials, src, rec, src_data, fs, ...
%                                                num_steps, visualization, capture_slice_step, update_type, ...
%                                                mesh_captures, double_precision, force_partition_to, octave)

clear mex;

% Test if the domain can theoretically fit on the devices available.
bounding_box = max(vertices)-min(vertices);
v = prod(bounding_box);
dx = 344/fs*sqrt(3);
v_n = dx.^3;
est_num_nodes = v/v_n;
fprintf('Estimated number of nodes: %u\n', uint32(est_num_nodes));

mem = mem_check();
if visualization == 1
    mem = mem(1);
else
    mem = sum(mem);
end

node_size = 10;
if(double_precision & visualization == 0)
    node_size = 18; 
end

limit_fs = (344*sqrt(3))/(v/(mem/node_size))^(1/3);
fprintf('The sampling frequency limit is approximately %u Hz\n', uint32(limit_fs)); 

if est_num_nodes*node_size > mem
    fprintf('The sampling frequency of %u Hz is too high, not enough memory\n', fs); 
    return
end

% Domains is under the limit, go on and reset the devices
clear mex;
fprintf('runFDTD - reset devices before simulation\n');
device_reset;

% Run the simulation
clear mex;
fprintf('runFDTD - run simulation\n');

try
    
[responses, ... 
 num_e, ...
 t_per_step, ...
 dim_x, ...
 dim_y, ...
 dim_z, ...
 dx, ...
 return_mesh] = mex_FDTD(single(vertices'), ...
                         uint32((indices)'), ...
                         single(materials'), ...
                         single(src'), ...
                         single(rec'), ... 
                         double(src_data'), ...
                         uint32(fs), ...
                         uint32(num_steps), ...
                         uint32(update_type), ...
                         uint32(visualization), ...
                         uint32(capture_slice_step'), ...
                         uint32(mesh_captures), ...
                         uint32(double_precision), ...
                         uint32(force_partition_to), ...
                         uint32(octave));
 
return_params.num_elements = num_e;
return_params.t_per_step = t_per_step;
return_params.dim_x = dim_x;
return_params.dim_y = dim_y;
return_params.dim_z = dim_z;
return_params.dx = dx;

clear mex;
fprintf('Run FDTD done\n');

catch e
    
end