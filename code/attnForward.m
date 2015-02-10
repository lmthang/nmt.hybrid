function [attnHidVecs, attn_h_concat, alignWeights, alignScores, attnInput] = attnForward(tgt_h_t, model, srcAlignStates, mask, params, curBatchSize, srcLens)
%%%
%
% Compute context vectors for attention-based models.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  attnInput = [tgt_h_t; srcLens];
  
  % align scores
  if params.attnFunc==1 % s_t = W_a * attnInput
    alignScores = model.W_a*attnInput;
  elseif params.attnFunc==2 % a_t softmax(tanh(W_a * attnInput))
    alignScores = params.nonlinear_f(model.W_a*attnInput);
  end
  
  % align weights a_t = softmax(s_t)
  alignWeights = softmax_new(alignScores);
  
  % mask
  % alignWeights = bsxfun(@times, alignWeights, mask), then change alignWeights from maxSentLen*curBatchSize-> 1 * curBatchSize * maxSentLen
  alignWeights = permute(bsxfun(@times, alignWeights, mask), [3, 2, 1]);
  
  % % alignWeights: maxSentLen * curBatchSize
  % attnVecs = squeeze(sum(bsxfun(@times, srcAlignStates, alignWeights), 1))'; % lstmSize * curBatchSize
  
  % srcAlignStates: lstmSize * curBatchSize * maxSentLen
  % alignWeights: 1 * curBatchSize * maxSentLen
  % attention vectors: attn_t = H_src* a_t (weighted average of src vectors)
  % sum over maxSentLen
  attnVecs = squeeze(sum(bsxfun(@times, srcAlignStates, alignWeights), 3)); 
  if params.assert % lstmSize x curBatchSize
    assert(size(attnVecs, 1)==params.lstmSize);
    assert(size(attnVecs, 2)==curBatchSize);
  end
  
  % attention hidden vectors: h_attn_t = f(W_ah*[attn_t; tgt_h_t])
  attn_h_concat = [attnVecs; tgt_h_t];
  attnHidVecs = params.nonlinear_f(model.W_ah*attn_h_concat);
end
