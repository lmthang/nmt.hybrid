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
function [grad_ht, grad_srcHidVecs] = srcCompareLayerBackprop(grad_alignWeights, alignWeights, srcHidVecs, h_t, mask, params)
  % grad_alignWeights -> grad_scores
  [grad_scores] = normLayerBackprop(grad_alignWeights, alignWeights, mask, params);
  
  % assert
  if params.assert
    assert(sum(sum(abs(grad_scores(mask==0))))==0);
  end
  
  grad_scores = permute(grad_scores, [3, 2, 1]); % batchSize * numPositions
  grad_ht = sum(bsxfun(@times, srcHidVecs, grad_scores), 3); % sum along numPositions: lstmSize * batchSize
  grad_srcHidVecs = bsxfun(@times, h_t, grad_scores);
end