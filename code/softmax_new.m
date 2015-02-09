function [probs, scores, norms] = softmax_new(scores, mask)
%%%
%
% Efficient (hopefully) softmax implementation.
% Return normalized probs as well as scores (numerators in log domain) & norms.
% Note that scores & norms are subtracted/scaled by a constant factor.
% If mask exists, mask out probs.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  mx = max(scores);
  scores = bsxfun(@minus, scores, mx); % subtract max elements 
  probs = exp(scores); % unnormalized probs 
  norms = sum(probs); % normalization factors
  if exist('mask', 'var')
    probs = bsxfun(@times, probs, mask./norms); % normalize and zero out at masked positions
  else
    probs = bsxfun(@rdivide, probs, norms); % normalized probs
  end
end