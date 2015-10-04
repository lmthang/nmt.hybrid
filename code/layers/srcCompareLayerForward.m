%%%
%
% For attention-based models, given:
%   srcHidVecs: lstmSize * curBatchSize * numPositions
%   h_t: lstmSize * curBatchSize
% compute:
%   alignScores: numPositions * curBatchSize
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [alignScores] = srcCompareLayerForward(srcHidVecs, h_t, params)
  alignScores = squeeze(sum(bsxfun(@times, srcHidVecs, h_t), 1))'; % numPositions * curBatchSize
  
  if params.curBatchSize==1 || params.numAttnPositions==1 % handle special cases that causing squeezing to transpose row/col vectors.
    alignScores = alignScores';
  end

  % assert
  if params.assert
    alignScores1 = zeroMatrix(size(alignScores), params.isGPU, params.dataType);
    for ii=1:params.curBatchSize
      alignScores1(:, ii) = squeeze(srcHidVecs(:, ii, :))'*h_t(:, ii);
    end
    
    assert(computeSum(alignScores-alignScores1, params.isGPU)<1e-4);
  end
end