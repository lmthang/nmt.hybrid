function [result] = randSimpleMatrix(size, isGPU, type)
%%%
%
% Initialize random matrix.
% 
% Hieu Pham @ 2014.
% Thang Luong @ 2014, <lmthang@stanford.edu>: added GPU support.
%
%%%
  if isGPU
    result = rand(size, type, 'gpuArray');  
  else
    result = rand(size, type);
  end
end