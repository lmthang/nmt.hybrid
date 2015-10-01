%% 
% Forward outVec = W*inVec
% Compute inGrad, grad_W
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%%
function [inGrad, grad_W] = linearLayerBackprop(W, outGrad, inVec)  
  % grad_W
  grad_W = outGrad*inVec';
  
  % inGrad
  inGrad = W'*outGrad;
end