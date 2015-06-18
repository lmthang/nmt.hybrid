%%%
%
% Nonlinear transformation f(W*inVec)
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [outVec] = hiddenLayerForward(W, inVec, nonlinear_f)
  outVec = nonlinear_f(W*inVec);
end