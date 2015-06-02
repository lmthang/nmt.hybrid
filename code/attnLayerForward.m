function [outVec, alignWeights] = attnLayerForward(W_a, inVec, srcHidVecs, mask)
%%%
%
% Compute context vectors for attention-based models.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  % s_t = W_a * inVec
  % align weights a_t = softmax(s_t): numAttnPositions*curBatchSize
  alignWeights = softmax(W_a*inVec); % numAttnPositions*curBatchSize

  outVec = contextLayerForward(bsxfun(@times, alignWeights, mask), srcHidVecs);
end

%   % alignWeights: numAttnPositions*curBatchSize
%   % mask: 1 * curBatchSize
%   % -> alignWeights: 1 * curBatchSize * numAttnPositions
%   alignWeights = permute(bsxfun(@times, alignWeights, mask), [3, 2, 1]);
%   
%   % srcHidVecs: lstmSize * curBatchSize * numAttnPositions
%   % alignWeights: 1 * curBatchSize * numAttnPositions
%   % attention vectors: attn_t = H_src* a_t (weighted average of src vectors)
%   % sum over numAttnPositions
%   outVec = squeeze(sum(bsxfun(@times, srcHidVecs, alignWeights), 3)); 