function [charData] = srcCharLayerForward(W_rnn, W_emb, input, charMap, vocabSize, params, isTest)
% Running char layer forward to compute word representations
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

  % find rare words
  charData.rareFlags = input > params.srcCharShortList;
  rareWords = unique(input(charData.rareFlags));
  charData.numRareWords = length(rareWords);
  charSeqs = charMap(rareWords);
  seqLens = cellfun(@(x) length(x), charSeqs);
  
  if ~isempty(rareWords)    
    charData.params.curBatchSize = length(rareWords);
    [charData.batch, charData.mask, charData.maxLen, charData.numSeqs] = leftPad(charMap(rareWords), seqLens, params.srcCharSos, params.srcCharEos);
    
    charData.rnnFlags = struct('decode', 0, 'test', isTest, 'attn', 0, 'feedInput', 0, 'charSrcRep', 0, 'charTgtGen', 0);
    zeroState = createZeroState(charData.params);
    [charData.states, ~, ~] = rnnLayerForward(W_rnn, W_emb, zeroState, charData.batch, charData.mask, charData.params, charData.rnnFlags, [], [], []);
    charData.rareWordReps = charData.states{end}{end}.h_t;
    charData.rareWordMap = zeros(vocabSize, 1);
    charData.rareWordMap(rareWords) = 1:length(rareWords);
  end
end