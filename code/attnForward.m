function [attnHidVecs, attn_h_concat, alignWeights, alignScores, attnInput] = attnForward(tgt_h_t, model, params, srcHidVecs, curMask)
%%%
%
% Compute context vectors for attention-based models.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  attnInput = tgt_h_t;
  alignScores = model.W_a*attnInput;
  
  
  % align weights a_t = softmax(s_t): numAttnPositions*curBatchSize
  alignWeights = softmax(alignScores);
  
  % mask &change alignWeights to 1 * curBatchSize * numAttnPositions
  alignWeights = permute(bsxfun(@times, alignWeights, curMask.mask), [3, 2, 1]);
  
  % srcHidVecs: lstmSize * curBatchSize * numAttnPositions
  % alignWeights: 1 * curBatchSize * numAttnPositions
  % attention vectors: attn_t = H_src* a_t (weighted average of src vectors)
  % sum over numAttnPositions
  attnVecs = squeeze(sum(bsxfun(@times, srcHidVecs, alignWeights), 3)); 
  
  % attention hidden vectors: attnHid = f(W_ah*[attn_t; tgt_h_t])
  attn_h_concat = [attnVecs; tgt_h_t];
  attnHidVecs = params.nonlinear_f(model.W_ah*attn_h_concat);
end

%   attnInput = [tgt_h_t; trainData.srcLens];
%   % align scores
%   if params.attnFunc==1 % s_t = W_a * attnInput
%     alignScores = model.W_a*attnInput;
%   elseif params.attnFunc==2 % s_t = tanh(W_a * attnInput)
%     alignScores = params.nonlinear_f(model.W_a*attnInput);
%   end
