%% 
% Forward outVec = f(W*inVec)
% Compute inGrad, grad_W
% IMPORTANT: this method only works when nonlinear_f_prime is either for
%   sigmoid or tanh in which we can reuse the forward computation (outVec) 
%   to compute the gradient faster.
% Thang Luong @ 2015, <lmthang@stanford.edu>
%%
function [inGrad, grad_W] = hiddenLayerBackprop(W, outGrad, inVec, nonlinear_f_prime, outVec)  
  % f'(outVec).*outGrad
  tmpResult = nonlinear_f_prime(outVec).*outGrad;  
  
  % grad_W
  grad_W = tmpResult*inVec';
  
  % inGrad
  inGrad = W'*tmpResult;
end