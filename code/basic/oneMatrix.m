function [result] = oneMatrix(size, isGPU, type)
%%%
%
% Initialize zero matrix with GPU supprot
% 
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  if isGPU
    result = ones(size, type, 'gpuArray');  
  else
    result = ones(size, type);
  end
end