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

  charData.rareFlags = input > params.srcCharShortList;
  rareWords = unique(input(charData.rareFlags));
  
  charData.numRareWords = length(rareWords);
  charSeqs = charMap(rareWords);
  seqLens = cellfun(@(x) length(x), charSeqs);
  
  if ~isempty(rareWords)    
    charData.params.curBatchSize = length(rareWords);
    [charData.batch, charData.mask, charData.maxLen, charData.numSeqs] = leftPad(charMap(rareWords), seqLens, params.srcCharSos, params.srcCharEos);
    
    if params.debug
      fprintf(2, '# srcCharLayerForward before, %s: %d words, maxLen %d\n', gpuInfo(params.gpu), charData.numRareWords, charData.maxLen);
    end
    
    charData.rnnFlags = struct('decode', 0, 'test', isTest, 'attn', 0, 'feedInput', 0, 'charSrcRep', 0, 'charTgtGen', 0, 'initEmb', []);
    zeroState = createZeroState(charData.params);
    [charData.states, ~, ~] = rnnLayerForward(W_rnn, W_emb, zeroState, charData.batch, charData.mask, charData.params, charData.rnnFlags, [], [], []);
    charData.rareWordReps = charData.states{end}{end}.h_t;
    charData.rareWordMap = zeros(vocabSize, 1);
    charData.rareWordMap(rareWords) = 1:length(rareWords);
    
    if params.debug
      fprintf(2, '  after, %s\n', gpuInfo(params.gpu));
    end
    
%     [charData] = srcCharMultiBatchForward(W_rnn, W_emb, charSeqs, seqLens, charData, params, isTest, vocabSize, rareWords);
%     
%     % assert: compare with non-multi batch
%     if params.assert
%       [charData1] = srcCharMultiBatchForward(W_rnn, W_emb, charSeqs, seqLens, charData, params, isTest, vocabSize, rareWords);
%       assert(sum(abs(charData.rareWordReps(:) - charData1.rareWordReps(:))) < 1e-10);
%     end
  end
end

% function [charData] = srcCharMultiBatchForward(W_rnn, W_emb, charSeqs, seqLens, charData, params, isTest, vocabSize, rareWords)
%   [charData] = multiBatchCharLayerForward(W_rnn, W_emb, charSeqs, seqLens, charData, params, isTest, 0, []);    
%   charData.rareWordReps = zeroMatrix([params.lstmSize, charData.numRareWords], params.isGPU, params.dataType);
%   count = 0;
%   for bb=1:charData.numBatches
%     charData.rareWordReps(:, count+1:count+charData.batches{bb}.params.curBatchSize) = charData.batches{bb}.states{end}{end}.h_t;
%     count = count + charData.batches{bb}.params.curBatchSize;
%   end
%   assert(count == charData.numRareWords);
% 
%   % note that this rareWordReps has been sorted, so we need to make it right!
%   charData.rareWordReps(:, charData.sortedIndices) = charData.rareWordReps;
%   charData.rareWordMap = zeros(vocabSize, 1);
%   charData.rareWordMap(rareWords) = 1:length(rareWords);
% end
