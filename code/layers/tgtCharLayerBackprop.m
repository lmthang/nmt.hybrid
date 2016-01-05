function [grad_W_rnn, grad_W_emb, emb_indices, grad_init_emb] = tgtCharLayerBackprop(W_rnn, charData, topGrads)
% Backprop for char layer from word gradients to chars.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>

  
  % init state
  params = charData.params;
  zeroBatch = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  zeroGrad = cell(params.numLayers, 1);
  for ll=1:params.numLayers % layer
    zeroGrad{ll} = zeroBatch;
  end
  
  [~, ~, grad_W_rnn, grad_W_emb, emb_indices, ~, ~, charGrad] = rnnLayerBackprop(W_rnn, charData.states, charData.initState, ...
  topGrads, zeroGrad, zeroGrad, charData.batch, charData.mask, charData.params, charData.rnnFlags, [], [], []);
  grad_init_emb = charGrad.initEmb;
end