function [result] = zeroMatrix(size, isGPU, type)
%%%
%
% Initialize zero matrix with GPU supprot
% 
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  if isGPU
    result = zeros(size, type, 'gpuArray');  
  else
    result = zeros(size, type);
  end
end
