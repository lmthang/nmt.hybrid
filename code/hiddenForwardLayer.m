function [outVec] = hiddenForwardLayer(W, inVec, nonlinear_f)
  outVec = nonlinear_f(W*inVec);
end