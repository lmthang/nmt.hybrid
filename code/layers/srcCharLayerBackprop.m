function [grad_W_rnn, grad_W_emb, emb_indices] = srcCharLayerBackprop(W_rnn, charData, charGrad)
% Backprop for char layer from word gradients to chars.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>

  assert(length(charGrad.indices) == charData.numSeqs);
  topGrads = cell(charData.maxLen, 1);
  topGrads{end} = charGrad.embs(:, charData.rareWordMap(charGrad.indices));
  
  % init state
  params = charData.params;
  zeroBatch = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  zeroState = cell(params.numLayers, 1);
  zeroGrad = cell(params.numLayers, 1);
  for ll=1:params.numLayers % layer
    zeroState{ll}.h_t = zeroBatch;
    zeroState{ll}.c_t = zeroBatch;
    zeroGrad{ll} = zeroBatch;
  end
  
  charRnnFlags = struct('decode', 0, 'attn', 0, 'feedInput', 0, 'char', 0);
  [~, ~, grad_W_rnn, grad_W_emb, emb_indices, ~, ~, ~] = rnnLayerBackprop(W_rnn, charData.states, zeroState, ...
  topGrads, zeroGrad, zeroGrad, charData.batch, charData.mask, charData.params, charRnnFlags, [], [], []);
end