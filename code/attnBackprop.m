function [attnGrad, grad_ht] = attnBackprop(model, topHidVecs, softmax_h, grad_softmax_h, attn_h_concat, alignWeights, alignScores, attnInput, params)
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
  % attn_t = H_src* a_t
  % topHidVecs(:, :, 1:params.numSrcHidVecs): lstmSize * curBatchSize * numSrcHidVecs
  % grad_attn: lstmSize * curBatchSize * 1
  % alignWeights: 1 * curBatchSize * numSrcHidVecs
  % grad_srcHidVecs = grad_attn * alignWeights'
  attnGrad.srcHidVecs = bsxfun(@times, grad_attn, alignWeights);

  % grad_alignWeights = H_src' * grad_attn (per example, to scale over multiple examples, i.e., curBatchSize, need to use bsxfun)
  grad_alignWeights = squeeze(sum(bsxfun(@times, topHidVecs(:, :, 1:params.numSrcHidVecs), grad_attn), 1))'; % bsxfun along numSrcHidVecs, sum across lstmSize

  if params.assert % numSrcHidVecs x curBatchSize
    assert(size(grad_alignWeights, 1)==params.numSrcHidVecs);
    assert(size(grad_alignWeights, 2)==size(softmax_h, 2));
  end

  %% from grad_alignWeights -> grad_scores
  % alignWeights a = softmax(scores)
  % single example i: grad_score_i = (diag(a_i) - a_i*a_i')*grad_a_i 
  %                            = a_i.*grad_a_i - a_i*(a_i'*grad_a_i)
  %                            = a_i.*grad_a_i - a_i*alpha_i
  % multiple examples: alpha = sum(a.*grad_a, 1) % 1*curBatchSize
  %     grad_scores = a.*grad - bsxfun(@times, a, alpha)
  % tmpResult = alignWeights.*grad_alignWeights; % numSrcHidVecs * curBatchSize
  alignWeights = squeeze(alignWeights)'; % alignWeights now: numSrcHidVecs * curBatchSize
  tmpResult = alignWeights.*grad_alignWeights; % numSrcHidVecs * curBatchSize
  grad_scores = tmpResult - bsxfun(@times, alignWeights, sum(tmpResult, 1));

  %% grad_scores -> grad.Wa, grad_ht
  if params.attnFunc==1 % s_t = W_a * attnInput
    tmpResult = grad_scores;
  elseif params.attnFunc==2 % s_t = f(W_a * attnInput)
    tmpResult = params.nonlinear_f_prime(alignScores).*grad_scores;
  end
  % grad.W_a = tmpResult * attnInput'
  attnGrad.W_a = tmpResult * attnInput';
  % grad_attn_input = W_a' * tmpResult
  grad_attn_input = model.W_a'*tmpResult;
  
  % alignScores = model.W_a*tgt_h_t;
  grad_ht = grad_ht + grad_attn_input;   
end

%   % since attnInput = [tgt_h_t; srcLens], accumulating grad_ht
%   grad_ht = grad_ht + grad_attn_input(1:end-1, :);
