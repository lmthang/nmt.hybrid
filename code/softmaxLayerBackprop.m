%%
%
% Perform softmax backprop.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%
function [grad_W, inGrad] = softmaxLayerBackprop(W, inVec, probs, scoreIndices)
  probs(scoreIndices) = probs(scoreIndices) - 1; % minus one at predicted words

  % softmax_h
  inGrad = W'* probs;

  % W_soft
  grad_W = probs*inVec';
end