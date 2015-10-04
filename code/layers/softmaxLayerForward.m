%%
% Perform softmax prediction.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%
function [cost, probs, scores, scoreIndices] = softmaxLayerForward(W, inVec, predLabels, curMask)
  mask = curMask.mask;
  unmaskedIds = curMask.unmaskedIds;
  
  % softmax_h -> predictions
  [probs, scores, norms] = softmax(W*inVec, mask);
  
  % cost
  predLabels = predLabels(unmaskedIds);
  scoreIndices = sub2ind(size(scores), predLabels, unmaskedIds); % 1 * length(tgtPredictedWords)
  cost = - sum(scores(scoreIndices)) + sum(log(norms).*mask);
end