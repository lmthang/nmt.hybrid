function [inGrad, grad_W] = hiddenLayerBackprop(W, outGrad, inVec, nonlinear_f_prime, outVec)
%% 
% Forward outVec = f(W*inVec)
% Compute inGrad, grad_W
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%%
  
  % f'(outVec).*outGrad
  tmpResult = nonlinear_f_prime(outVec).*outGrad;  
  
  % grad_W
  grad_W = tmpResult*inVec';
  
  % inGrad
  inGrad = W'*tmpResult;
end