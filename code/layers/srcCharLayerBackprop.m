function [grad_W_rnn, grad_W_emb, emb_indices] = srcCharLayerBackprop(W_rnn, charData, charGrad)
% Backprop for char layer from word gradients to chars.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>

  if charData.numRareWords == 0
    grad_W_rnn = []; 
    grad_W_emb = [];
    emb_indices = [];
    return;
  end
  
  assert(length(charGrad.indices) == charData.numRareWords);
  
  params = charData.params;
  if params.assert
    assert(isequal(sort(charData.rareWordMap(charGrad.indices))', 1:charData.numRareWords));
  end
  
  topGrads = cell(charData.maxLen, 1);
  topGrads{end} = charGrad.embs(:, charData.rareWordMap(charGrad.indices));

  % init state
  zeroBatch = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  zeroState = cell(params.numLayers, 1);
  zeroGrad = cell(params.numLayers, 1);
  for ll=1:params.numLayers % layer
    zeroState{ll}.h_t = zeroBatch;
    zeroState{ll}.c_t = zeroBatch;
    zeroGrad{ll} = zeroBatch;
  end
  
  [~, ~, grad_W_rnn, grad_W_emb, emb_indices, ~, ~, ~] = rnnLayerBackprop(W_rnn, charData.states, zeroState, ...
  topGrads, zeroGrad, zeroGrad, charData.batch, charData.mask, charData.params, charData.rnnFlags, [], [], []); 
end
