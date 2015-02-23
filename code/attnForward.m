function [attnHidVecs, attn_h_concat, alignWeights, alignScores, attnInput] = attnForward(tgt_h_t, model, params, trainData, curMask)
%%%
%
% Compute context vectors for attention-based models.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  attnInput = [tgt_h_t; trainData.srcLens];
  
  % align scores
  if params.attnFunc==1 % s_t = W_a * attnInput
    alignScores = model.W_a*attnInput;
  elseif params.attnFunc==2 % s_t = tanh(W_a * attnInput)
    alignScores = params.nonlinear_f(model.W_a*attnInput);
  end
  
  % align weights a_t = softmax(s_t)
  alignWeights = softmax(alignScores);
  
  % mask
  % alignWeights = bsxfun(@times, alignWeights, mask), then change alignWeights from maxSentLen*curBatchSize-> 1 * curBatchSize * maxSentLen
  alignWeights = permute(bsxfun(@times, alignWeights, curMask.mask), [3, 2, 1]);
  
  % % alignWeights: maxSentLen * curBatchSize
  % attnVecs = squeeze(sum(bsxfun(@times, srcHidVecs, alignWeights), 1))'; % lstmSize * curBatchSize
  
  % srcHidVecs: lstmSize * curBatchSize * maxSentLen
  % alignWeights: 1 * curBatchSize * maxSentLen
  % attention vectors: attn_t = H_src* a_t (weighted average of src vectors)
  % sum over maxSentLen
  attnVecs = squeeze(sum(bsxfun(@times, trainData.srcHidVecs, alignWeights), 3)); 
  if params.assert % lstmSize x curBatchSize
    assert(size(attnVecs, 1)==params.lstmSize);
    assert(size(attnVecs, 2)==trainData.curBatchSize);
  end
  
  % attention hidden vectors: attnHid = f(W_ah*[attn_t; tgt_h_t])
  attn_h_concat = [attnVecs; tgt_h_t];
  attnHidVecs = params.nonlinear_f(model.W_ah*attn_h_concat);
end
