%%%
% For attention-based models, given:
%   grad_scores: numPositions * batchSize
%   srcHidVecs: lstmSize * batchSize * numPositions
%   h_t: lstmSize * batchSize
% compute the following grads:
%   grad_ht: lstmSize * batchSize
%   grad_srcHidVecs: lstmSize * batchSize * numPositions
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [grad_ht, grad_srcHidVecs] = srcCompareLayerBackprop(grad_scores, h_t, srcHidVecs)
  grad_scores = permute(grad_scores, [3, 2, 1]); % 1 * batchSize * numPositions
  grad_ht = sum(bsxfun(@times, srcHidVecs, grad_scores), 3); % sum along numPositions: lstmSize * batchSize
  grad_srcHidVecs = bsxfun(@times, h_t, grad_scores);
end
