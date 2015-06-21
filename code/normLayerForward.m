%%
%
% Return normalized probs as well as scores (numerators in log domain) & norms.
% Note that scores & norms are subtracted/scaled by a constant factor.
% 
% Here, we only work on a subset linearIds of the raw scores.
%
% Thang Luong @ 2015, <lmthang@stanford.edu
%
%%
function [probs, scores, norms] = normLayerForward(rawScores, mask, params)
  linearIds = find(mask==1);
  
  % rawScores: numPositions * batchSize
  scores = zeroMatrix(size(rawScores), params.isGPU, params.dataType);
  scores(linearIds) = rawScores(linearIds);

  % subtract max elements, scores: numClasses * ...
  mx = max(scores, [], 1);
  scores = bsxfun(@minus, scores, mx); 
  
  % probs
  probs = zeroMatrix(size(scores), params.isGPU, params.dataType);
  probs(linearIds) = exp(scores(linearIds)); % unnormalized probs
  norms = sum(probs, 1); % normalization factors
  norms(norms==0) = 1; % for zero columns, set to 1.
  probs = bsxfun(@rdivide, probs, norms); % normalize
end