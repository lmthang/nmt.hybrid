%%%
%
% For attention-based models, given:
%   alignWeights: numPositions * batchSize.
%   srcHidVecs: lstmSize * batchSize * numPositions.
% compute the context vectors (weighted sum):
%   contextVecs: lstmSize * batchSize.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [contextVecs] = contextLayerForward(alignWeights, srcHidVecs)
  % change alignWeights -> 1 * batchSize * numPositions
  % multiply then sum across the numPositions dimension.
  contextVecs = squeeze(sum(bsxfun(@times, srcHidVecs, permute(alignWeights, [3, 2, 1])), 3)); % lstmSize * batchSize
end