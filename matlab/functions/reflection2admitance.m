function admitance = reflection2admitance(reflection_coefficient)
%admitance = reflection2admitance(reflection_coefficient)

admitance = (1-reflection_coefficient)./(1+reflection_coefficient);

end