function [charData] = tgtCharLayerForward(W_rnn, W_emb, input, decStates, charMap, params, isTest)
% Running char layer forward to prepare for target word generation later.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%   input: indices for the current batch
%   isTest: 1 -- don't store intermediate results in each state
%   isDecoder: 0 -- encoder, 1 -- decoder
% Output:
%   charData
%
% Thang Luong @ 2015, <lmthang@stanford.edu>

  charData.params = params;
  charData.params.numLayers = params.charNumLayers;

  charData.rareFlags = input > params.tgtCharShortList;
  rareWords = input(charData.rareFlags);
  charData.params.curBatchSize = length(rareWords);
  rareByTime = sum(charData.rareFlags, 1);
  
  T = length(decStates);
  count = 0;
  initEmb = zeroMatrix([params.lstmSize, charData.params.curBatchSize], params.isGPU, params.dataType);
  for tt=1:T
    initEmb(:, count+1:count+rareByTime(tt)) = decStates{tt}{end}.h_t(:, charData.rareFlags(:, tt));
    count = count + rareByTime(tt);
  end
  
  % TODO: might need to make batch size smaller 
  if ~isempty(rareWords)    
    % TODO: sort & split batches
    [charData.batch, charData.mask, charData.maxLen, charData.numSeqs] = rightPad(charMap(rareWords), params.tgtCharEos, params.tgtCharSos);
    zeroState = createZeroState(charData.params);
    rnnFlags = struct('decode', 0, 'test', isTest, 'attn', 0, 'feedInput', 0, 'char', 0);
    [charData.states, ~, ~] = rnnLayerForward(W_rnn, W_emb, zeroState, charData.batch, charData.mask, charData.params, rnnFlags, [], [], [], initEmb);
  
%     charData.rareWordMap = zeros(vocabSize, 1);
%     charData.rareWordMap(rareWords) = 1:length(rareWords);
%     charData.rareWordReps = charData.states{end}{end}.h_t;
  end
end