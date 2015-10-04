%%%
%
% For attention-based models, given:
%   grad_contextVecs: lstmSize * batchSize
%   alignWeights: numPositions * batchSize.
%   srcHidVecs: lstmSize * batchSize * numPositions.
% compute the following grads:
%   grad_srcHidVecs: lstmSize * batchSize * numPositions
%   grad_alignWeights: numPositions * batchSize
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [grad_alignWeights, grad_srcHidVecs] = contextLayerBackprop(grad_contextVecs, alignWeights, srcHidVecs, unmaskedIds, params)
  % change from grad_contextVecs lstmSize*batchSize -> lstmSize*batchSize*1
  grad_contextVecs = permute(grad_contextVecs, [1, 2, 3]); 
  
  %% Grad formulae:
  %   contextVecs = H_src* a_t
  %   grad_srcHidVecs: outGrad * alignWeights'
  %   grad_alignWeights = H_src' * contexGrad (per example, to scale over multiple examples, i.e., batchSize, need to use bsxfun)
  
  %% grad_srcHidVecs
  grad_srcHidVecs = zeroMatrix(size(srcHidVecs), params.isGPU, params.dataType);
  grad_srcHidVecs(:, unmaskedIds, :) = bsxfun(@times, grad_contextVecs(:, unmaskedIds), permute(alignWeights(:, unmaskedIds), [3, 2, 1])); % change alignWeights -> 1 * batchSize * numPositions
  
  %% grad_alignWeights
  grad_alignWeights = zeroMatrix(size(alignWeights), params.isGPU, params.dataType);
  if size(srcHidVecs, 3)==1
    grad_alignWeights(:, unmaskedIds) = sum(bsxfun(@times, srcHidVecs(:, unmaskedIds, :), grad_contextVecs(:, unmaskedIds)), 1); % sum across lstmSize: numPositions * batchSize
  else
    grad_alignWeights(:, unmaskedIds) = squeeze(sum(bsxfun(@times, srcHidVecs(:, unmaskedIds, :), grad_contextVecs(:, unmaskedIds)), 1))'; % sum across lstmSize: numPositions * batchSize
  end
  
  
  %% assert
  if params.assert
    assert(isequal(size(grad_alignWeights), size(alignWeights)));
    
    % compute grad_srcHidVec in a different way
    grad_srcHidVecs1 = zeroMatrix(size(grad_srcHidVecs), params.isGPU, params.dataType);
    for jj=1:length(unmaskedIds)
      ii = unmaskedIds(jj);
      grad_srcHidVecs1(:, ii, :) = grad_contextVecs(:, ii, 1) * alignWeights(:, ii)';
    end
    assert(computeSum(grad_srcHidVecs1-grad_srcHidVecs, params.isGPU)==0);
  end
end
