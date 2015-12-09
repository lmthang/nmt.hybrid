function [lstmStates, trainData, attnInfo] = rnnLayerForward(T, W_rnn, W_emb, prevState, input, maskInfos, params, ...
  isTest, isFeedInput, isDecoder, trainData, model)
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
  
lstmStates = cell(T, 1);

% attention
if isDecoder && params.attnFunc 
  attnInfo = cell(T, 1);
else
  attnInfo = [];
end

for tt=1:T % time
  % emb input
  if isDecoder && params.feedInput
    x_t = [W_emb(:, input(:, tt)); prevState{end}.softmax_h];
  else
    x_t = W_emb(:, input(:, tt));
  end
  
  % multi-layer RNN
  [lstmStates{tt}] = rnnStepLayerForward(W_rnn, prevState, x_t, maskInfos{tt}.maskedIds, params, isTest, isFeedInput);
  
  % decoder
  if isDecoder
    % attention
    if params.attnFunc 
      % TODO: save memory here, attnInfo.input only keeps track of srcHidVecs or attnVecs, but not h_t.
      [attnInfo{tt}] = attnLayerForward(lstmStates{tt}{params.numLayers}.h_t, params, model, trainData, maskInfos{tt}, tt);
      lstmStates{tt}{end}.softmax_h = attnInfo{tt}.softmax_h;
    else
      lstmStates{tt}{end}.softmax_h = lstmStates{tt}{end}.h_t;
    end
  end
  
  % update    
  prevState = lstmStates{tt};
end

% attention, store information on the encoder side
if params.attnFunc && isDecoder == 0
  % Record src hidden states
  for tt=1:params.numSrcHidVecs
    trainData.srcHidVecsOrig(:, :, tt) = lstmStates{tt}{params.numLayers}.h_t;
  end

  if params.attnGlobal == 0 % local
    trainData.srcHidVecs = trainData.srcHidVecsOrig;
  end
end