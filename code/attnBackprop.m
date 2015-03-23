function [attnGrad, grad_ht, grad_srcHidVecs] = attnBackprop(model, batchData, softmax_h, grad_softmax_h, attn_h_concat, alignWeights, attnInput, params, curMask)
%%%
%
% Compute grad for attention-based models.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  %% grad_softmax_h -> grad.W_ah, grad_ah 
  % attn_h_concat = [attn_t; tgt_h_t]
  % softmax_h = f(W_ah*attn_h_concat)
  % f'(softmax_h).*grad_softmax_h
  tmpResult = params.nonlinear_f_prime(softmax_h).*grad_softmax_h;  
  % grad.W_ah
  attnGrad.W_ah = tmpResult*attn_h_concat';
  % grad_ah
  grad_ah = model.W_ah'*tmpResult;
  
  %% grad_ah -> grad_ht, grad_attn
  % grad_ht
  grad_ht = grad_ah(params.lstmSize+1:end, :);
  % grad_attn
  grad_attn = permute(grad_ah(1:params.lstmSize, :), [1, 2, 3]); % change from lstmSize*curBatchSize -> lstmSize*curBatchSize*1

  %% from grad_attn -> grad_srcHidVecs, grad_alignWeights
  % Grad formulae:
  %   attn_t = H_src* a_t
  %   grad_srcHidVecs: grad_attn * alignWeights'
  %   grad_alignWeights = H_src' * grad_attn (per example, to scale over multiple examples, i.e., curBatchSize, need to use bsxfun)
  
  % Sizes:
  %   batchData.srcHidVecs: lstmSize * curBatchSize * numAttnPositions
  %   grad_attn: lstmSize * curBatchSize * 1
  %   alignWeights: 1 * curBatchSize * numAttnPositions
  %   attnGrad.srcHidVecs: lstmSize * curBatchSize * numAttnPositions
  %   grad_alignWeights: numAttnPositions * curBatchSize

  grad_srcHidVecs = bsxfun(@times, grad_attn, alignWeights);  
  grad_alignWeights = squeeze(sum(bsxfun(@times, batchData.srcHidVecs, grad_attn), 1))'; % bsxfun along numAttnPositions, sum across lstmSize

  if params.assert % numAttnPositions x curBatchSize
    assert(size(grad_alignWeights, 1)==params.numAttnPositions);
    assert(size(grad_alignWeights, 2)==params.curBatchSize);
    
    % compute grad_srcHidVec in a different way
    grad_srcHidVecs1 = zeroMatrix(size(grad_srcHidVecs), params.isGPU, params.dataType);
    for ii=1:params.curBatchSize
      grad_srcHidVecs1(:, ii, :) = grad_attn(:, ii, 1) * squeeze(alignWeights(1, ii, :))';
    end
    assert(sum(sum(sum(abs(grad_srcHidVecs1-grad_srcHidVecs))))==0);
  end

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
  alignWeights = squeeze(alignWeights)'; % alignWeights now: numAttnPositions * curBatchSize
  tmpResult = alignWeights.*grad_alignWeights; % numAttnPositions * curBatchSize
  grad_scores = tmpResult - bsxfun(@times, alignWeights, sum(tmpResult, 1));

  if params.assert
    assert(sum(sum(abs(attnInput(:, curMask.maskedIds))))==0);
    
    % compute grad_scores in a different way
    grad_scores1 = zeroMatrix(size(grad_scores), params.isGPU, params.dataType);
    for ii=1:params.curBatchSize
      grad_scores1(:, ii) = (diag(alignWeights(:, ii))-alignWeights(:, ii)*alignWeights(:, ii)')*grad_alignWeights(:, ii);
    end
    assert(sum(sum(abs(grad_scores-grad_scores1)))<1e-10);
  end
  
  %% grad_scores -> grad.Wa, grad_ht
  % s_t = W_a * attnInput
  % here attnInput = h_t
  
  % grad.W_a = grad_scores * attnInput'
  attnGrad.W_a = grad_scores * attnInput';
  % grad_attn_input = W_a' * grad_scores
  grad_attn_input = model.W_a'*grad_scores;
  
  % alignScores = model.W_a*tgt_h_t;
  grad_ht = grad_ht + grad_attn_input;   
end

%   % since attnInput = [tgt_h_t; srcLens], accumulating grad_ht
%   grad_ht = grad_ht + grad_attn_input(1:end-1, :);
