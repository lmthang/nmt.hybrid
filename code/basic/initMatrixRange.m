function [result] = initMatrixRange(rangeSize, size, isGPU, type)
%%%
%
% Initialize random matrix.
% 
% Hieu Pham @ 2014.
% Thang Luong @ 2014, <lmthang@stanford.edu>: added GPU support.
%
%%%
  if isGPU
    result = 2*rangeSize * (rand(size, type, 'gpuArray') - 0.5);  
  else
    result = 2*rangeSize * (rand(size, type) - 0.5);
  end
end