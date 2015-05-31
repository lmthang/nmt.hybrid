%%
%
% Efficient (hopefully) softmax implementation.
% Return normalized probs as well as scores (numerators in log domain) & norms.
% Note that scores & norms are subtracted/scaled by a constant factor.
% 
% Here, we only work on a subset linearIds of the raw scores.
%
% Thang Luong @ 2015, <lmthang@stanford.edu
%
%%
function [probs, scores, norms] = softmaxMask(raw, linearIds, params)
  scores = zeroMatrix(size(raw), params.isGPU, params.dataType);
  scores(linearIds) = raw(linearIds);

  % subtract max elements, scores: numClasses * ...
  mx = max(scores, [], 1);
  scores = bsxfun(@minus, scores, mx); 
  
  % probs
  probs = zeroMatrix(size(raw), params.isGPU, params.dataType);
  probs(linearIds) = exp(scores(linearIds)); % unnormalized probs
  norms = sum(probs, 1); % normalization factors
  norms(norms==0) = 1; % for zero columns, set to 1.
  probs = bsxfun(@rdivide, probs, norms); % normalize
end