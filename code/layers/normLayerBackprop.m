function [grad_scores] = normLayerBackprop(grad_alignWeights, alignWeights) %, maskedIds, params)
  %% from grad_alignWeights -> grad_scores
  % alignWeights a = softmax(scores)
  % Let's derive per indices grad align weight w.r.t scores
  %   der a_i / der s_j = der exp(s_i) / sum_k (exp(s_k)) / der s_j =
  %     (1/sum) * (der exp(s_i) / der s_j) - (exp(s_i)/sum^2)*exp(s_j) =
  %      a_i*I{i==j} - a_i*_a_j
  %
  % Now let's try to optimize the vector grad for a single example i: 
  %   grad_score_i = (diag(a_i) - a_i*a_i')*grad_a_i 
  %                = a_i.*grad_a_i - a_i*(a_i'*grad_a_i)
  %                = a_i.*grad_a_i - a_i*alpha_i
  % multiple examples: alpha = sum(a.*grad_a, 1) % 1*curBatchSize
  %     grad_scores = a.*grad - bsxfun(@times, a, alpha)
  % tmpResult = alignWeights.*grad_alignWeights; % numAttnPositions * curBatchSize
  
  tmpResult = alignWeights.*grad_alignWeights; % numAttnPositions * curBatchSize
  grad_scores = tmpResult - bsxfun(@times, alignWeights, sum(tmpResult, 1));
    
%   % assert
%   if params.assert
%     % compute grad_scores in a different way
%     grad_scores1 = zeroMatrix(size(grad_scores), params.isGPU, params.dataType);
%     for ii=1:params.curBatchSize
%       grad_scores1(:, ii) = (diag(alignWeights(:, ii))-alignWeights(:, ii)*alignWeights(:, ii)')*grad_alignWeights(:, ii);
%     end
%     assert(computeSum(grad_scores-grad_scores1, params.isGPU)<1e-5);
%     
%     assert(computeSum(alignWeights(maskedIds), params.isGPU)==0);
%     assert(computeSum(grad_scores(maskedIds), params.isGPU)==0);
%     assert(computeSum(tmpResult(maskedIds), params.isGPU)==0);
%     tmpResult = bsxfun(@times, alignWeights, sum(tmpResult, 1));
%     assert(computeSum(tmpResult(maskedIds), params.isGPU)==0);
%   end
end
  