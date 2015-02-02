function [probs, scores, norms] = softmax(W_soft, h_t)
%%%
%
% Efficient (hopefully) softmax implementation.
% Return normalized probs as well as scores (numerators in log domain) & norms
% Note that scores & norms are subtracted/scaled by a constant factor.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  scores = W_soft * h_t;  % params.outVocabSize * curBatchSize
  mx = max(scores);
  scores = bsxfun(@minus, scores, mx); % subtract max elements 
  probs = exp(scores); % unnormalized probs 
  norms = sum(probs); % normalization factors
  probs = bsxfun(@rdivide, probs, norms); % normalized probs
end