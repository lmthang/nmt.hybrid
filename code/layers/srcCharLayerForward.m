function [charData] = srcCharLayerForward(W_rnn, W_emb, input, mask, charMap, vocabSize, params, isTest)
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
  charData.rareWords = unique(input(charData.rareFlags));
  
  % sample from frequent words
  if params.charSrcSample > 0
    % select by types: not masked and not unk
    freqWords = unique(input(~charData.rareFlags & mask & (input ~= params.srcUnk)));
    numSelect = floor(length(freqWords)*params.charSrcSample);
    perm = randperm(length(freqWords));
    selectFreqWords = freqWords(perm(1:numSelect));
    
    % assert
    if params.assert
      assert(isempty(intersect(charData.rareWords, selectFreqWords)) == 1);
      assert(ismember(params.srcSos, selectFreqWords) == 0);
    end
    
    % update rare words and flags
    charData.rareWords = union(charData.rareWords, selectFreqWords);
    charData.rareFlags = ismember(input, charData.rareWords);
    
    if params.assert
      assert(isempty(find(charData.rareWords == params.srcUnk, 1)));
    end
  end
  
  charData.numRareWords = length(charData.rareWords);
  charSeqs = charMap(charData.rareWords);
  seqLens = cellfun(@(x) length(x), charSeqs);
  
  if charData.numRareWords > 0
    charData.params.curBatchSize = charData.numRareWords;
    [charData.batch, charData.mask, charData.maxLen, charData.numSeqs] = leftPad(charMap(charData.rareWords), seqLens, params.srcCharSos, params.srcCharEos);
    
    charData.rnnFlags = struct('decode', 0, 'test', isTest, 'attn', 0, 'feedInput', 0, 'charSrcRep', 0, 'charTgtGen', 0);
    zeroState = createZeroState(charData.params);
    [charData.states, ~, ~] = rnnLayerForward(W_rnn, W_emb, zeroState, charData.batch, charData.mask, charData.params, charData.rnnFlags, [], [], []);
    charData.rareWordReps = charData.states{end}{end}.h_t;
    charData.rareWordMap = zeros(vocabSize, 1);
    charData.rareWordMap(charData.rareWords) = 1:charData.numRareWords;
  end
end