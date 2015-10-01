function [result] = randMatrix(size, isGPU, type)
%%%
%
% Initialize random matrix.
% 
% Thang Luong @ 2014, <lmthang@stanford.edu>
%
%%%
  if isGPU
    result = rand(size, type, 'gpuArray');  
  else
    result = rand(size, type);
  end
end
