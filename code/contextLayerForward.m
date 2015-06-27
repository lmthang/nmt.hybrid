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
function [contextVecs] = contextLayerForward(alignWeights, srcHidVecs, params)
  % change alignWeights -> 1 * batchSize * numPositions
  % multiply then sum across the numPositions dimension.
  contextVecs = squeeze(sum(bsxfun(@times, srcHidVecs, permute(alignWeights, [3, 2, 1])), 3)); % lstmSize * batchSize
  
  % assert
  if params.assert
    contextVecs1 = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
    for ii=1:params.curBatchSize
      avgVec = zeroMatrix([params.lstmSize, 1], params.isGPU, params.dataType);
      for jj=1:size(alignWeights, 1)
        avgVec = avgVec + alignWeights(jj, ii)*srcHidVecs(:, ii, jj);
      end
      contextVecs1(:, ii) = avgVec;
    end
    
    assert(computeSum(contextVecs-contextVecs1, params.isGPU)<1e-5);
  end
end