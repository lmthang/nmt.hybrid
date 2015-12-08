function [lstmStates, attnInfo] = rnnLayerForward(T, W_rnn, W_emb, prevState, input, maskInfos, params, isTest, isDecoder, trainData, model)
% Running Multi-layer RNN for one time step.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%   prevState: previous hidden state, e.g., for LSTM, prevState.c{ll}, prevState.h{ll}.
%   input: indices for the current batch
%   isTest: 1 -- don't store intermediate results
%
% Output:
%   nextState
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
  
batchSize = size(input, 1);
lstmStates = cell(T, 1);
softmax_h = zeroMatrix([params.lstmSize, batchSize], params.isGPU, params.dataType);
if isDecoder && params.attnFunc 
  attnInfo = cell(T, 1);
else
  attnInfo = [];
end

for tt=1:T % time
  % emb input
  if isDecoder
    x_t = getLstmDecoderInput(input(:, tt)', W_emb, softmax_h, params);
  else
    x_t = W_emb(:, input(:, tt));
  end
  
  % multi-layer RNN
  [lstmStates{tt}] = rnnStepLayerForward(W_rnn, prevState, x_t, maskInfos{tt}.maskedIds, params, isTest);
  
  % attention
  if isDecoder && params.attnFunc 
    % TODO: save memory here, h2sInfo.input only keeps track of srcHidVecs or attnVecs, but not h_t.
    [attnInfo{tt}] = attnLayerForward(lstmStates{tt}{params.numLayers}.h_t, params, model, trainData, maskInfos{tt}, tt);
    softmax_h = attnInfo{tt}.softmax_h;
  else
    softmax_h = lstmStates{tt}{params.numLayers}.h_t;
  end

  % update    
  prevState = lstmStates{tt};
end