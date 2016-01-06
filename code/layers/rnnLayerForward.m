function [lstmStates, attnData, attnInfos] = rnnLayerForward(W_rnn, W_emb, prevState, input, masks, params, rnnFlags, attnData, model)
% Running Multi-layer RNN for one time step.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%   prevState: previous hidden state, e.g., for LSTM, prevState.c{ll}, prevState.h{ll}.
%   input: indices for the current batch
%   isTest: 1 -- don't store intermediate results in each state
%   isAttn: for attention, require attnData to be non-empty, has
%     attnData.srcHidVecsOrig and attnData.srcLens.
%   isDecoder: 0 -- encoder, 1 -- decoder
% Output:
%   nextState
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
  
T = size(input, 2);
lstmStates = cell(T, 1);

% attention
attnInfos = cell(T, 1);
if rnnFlags.attn && rnnFlags.decode == 0 % encoder
  assert(T <= params.numSrcHidVecs);
  attnData.srcHidVecsOrig = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
end

for tt=1:T % time
  % local monotonic alignment
  if params.attnLocalMono
    attnData.tgtPos = tt;
  end
  
  % multi-layer RNN
  [prevState, attnInfos{tt}] = rnnStepLayerForward(W_rnn, W_emb(:, input(:, tt)), prevState, masks(:, tt), params, rnnFlags, attnData, model);
  
  % encoder, attention
  if rnnFlags.attn && rnnFlags.decode == 0
    attnData.srcHidVecsOrig(:, :, tt) = prevState{end}.h_t;
  end
  
  % store all states
  lstmStates{tt} = prevState;
end