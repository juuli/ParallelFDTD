function reflection_coefficient = admitance2reflection( admitance )
%ADMITANCE2REFLECTION Summary of this function goes here
%   Detailed explanation goes here
reflection_coefficient = (1./admitance-1)./(1./admitance+1);

end

