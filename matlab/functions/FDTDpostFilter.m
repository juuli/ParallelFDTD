function ir = FDTDpostFilter( ir, fs, frac )
%FDTDFILTER Summary of this function goes here
%   Detailed explanation goes here

%FDTDFILTER filtering function used in rendering 
% function ir = FDTDfilter(ir, fs, frac)
%
% ir: the response generated with a FDTD simulation
% fs: Samplingrate of the simulation
% opt: 'fdtd' low-pass + dc-block
% frac: the cuttoff frequency, normalized 


% Filter takes column vectors, transpose if the input has row vectors
[m n] = size(ir);
if(m > n)
    ir = ir';
end

% Filter the responses
b_low = fir1(200, frac);

% Set the cutoff frequency to 5 Hz
dc_block = dcblock(5, fs);

ir = filter(b_low, 1, ir,[], 2);
ir = filter([1 -1],[1 -dc_block], ir, [], 2)';


end

