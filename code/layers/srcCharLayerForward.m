function [charData] = srcCharLayerForward(W_rnn, W_emb, input, charMap, vocabSize, params, isTest)  % , isDecoder
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

  charData.rareFlags = input > params.srcCharShortList;
  rareWords = unique(input(charData.rareFlags));
  charData.params.curBatchSize = length(rareWords);
  
  if ~isempty(rareWords)    
    % TODO: sort & split batches
    [charData.batch, charData.mask, charData.maxLen, charData.numSeqs] = leftPad(charMap(rareWords), params.srcCharSos, params.srcCharEos);
    
    rnnFlags = struct('decode', 0, 'test', isTest, 'attn', 0, 'feedInput', 0, 'char', 0);
    zeroState = createZeroState(charData.params);
    [charData.states, ~, ~] = rnnLayerForward(W_rnn, W_emb, zeroState, charData.batch, charData.mask, charData.params, rnnFlags, [], [], []);
  
    charData.rareWordMap = zeros(vocabSize, 1);
    charData.rareWordMap(rareWords) = 1:length(rareWords);
    charData.rareWordReps = charData.states{end}{end}.h_t;
  end
end

  
%   assert(isDecoder == 0);
%   if isDecoder
%     charData.params.charSos = params.tgtCharSos;
%     charData.params.charEos = params.tgtCharEos;
%     charData.rareFlags = input > params.tgtCharShortList;
%   else  
%   end

%     [charData.states, charData.batch, charData.mask, charData.maxLen, charData.numSeqs] = char2wordReps(W_rnn, W_emb, ...
%       rareWords, charMap, charData.params, isTest);
