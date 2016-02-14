function [lstmStates, trainData, attnInfos] = rnnLayerForward(W_rnn, W_emb, prevState, input, masks, params, rnnFlags, trainData, model, charData)
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

% attention, encoder
attnInfos = cell(T, 1);
if rnnFlags.attn && rnnFlags.decode == 0
  assert(T == params.numSrcHidVecs);
  trainData.srcHidVecsOrig = zeroMatrix([params.lstmSize, params.curBatchSize, T], params.isGPU, params.dataType); % params.numSrcHidVecs
end

for tt=1:T % time
  if rnnFlags.charSrcRep && rnnFlags.decode == 0 % char representation, encoder
    inputEmb = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
    
    % charData.rareFlags: to know which words are rare
    % charData.rareWordReps: the actual rare word representations
    % rareWordMap: to find out indices in rareWordReps
    rareIds = find(charData.rareFlags(:, tt));
    freqIds = find(~charData.rareFlags(:, tt));
    assert(all(ismember(rareIds, find(masks(:, tt)==1))));
    
    % embeddings for rare words
    if ~isempty(rareIds)
      inputEmb(:, rareIds) = charData.rareWordReps(:, charData.rareWordMap(input(rareIds, tt)));
    end
    
    % embeddings for frequent words
    inputEmb(:, freqIds) = W_emb(:, input(freqIds, tt));
  else
    inputEmb = W_emb(:, input(:, tt));
  end
  
  % multi-layer RNN
  [prevState, attnInfos{tt}] = rnnStepLayerForward(W_rnn, inputEmb, prevState, masks(:, tt), params, rnnFlags, trainData, model);
  
  % encoder, attention
  if rnnFlags.attn && rnnFlags.decode == 0
    trainData.srcHidVecsOrig(:, :, tt) = prevState{end}.h_t;
  end
  
  % store all states
  lstmStates{tt} = prevState;
end
