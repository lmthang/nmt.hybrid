function [lstmStates, trainData, attnInfos] = rnnLayerForward(W_rnn, W_emb, prevState, input, masks, params, ...
  isTest, isDecoder, trainData, model)
% Running Multi-layer RNN for one time step.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%   prevState: previous hidden state, e.g., for LSTM, prevState.c{ll}, prevState.h{ll}.
%   input: indices for the current batch
%   isTest: 1 -- don't store intermediate results in each state
%   isDecoder: 0 -- encoder, 1 -- decoder
% Output:
%   nextState
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
  
T = size(input, 2);
lstmStates = cell(T, 1);

% attention
attnInfos = cell(T, 1);
if params.attnFunc && isDecoder == 0 % encoder
  assert(T <= params.numSrcHidVecs);
  trainData.srcHidVecsOrig = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
end

for tt=1:T % time
  % multi-layer RNN
  [prevState, attnInfos{tt}] = rnnStepLayerForward(W_rnn, W_emb, prevState, input(:, tt), masks(:, tt), params, isTest, isDecoder, ...
    model, trainData);
  
  % encoder, attention
  if isDecoder == 0 && params.attnFunc 
    trainData.srcHidVecsOrig(:, :, tt) = prevState{end}.h_t;
  end
  
  % store all states
  lstmStates{tt} = prevState;
end