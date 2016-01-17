function [charData] = tgtCharLayerForward(W_rnn, W_emb, input, tgtHidVecs, charMap, params, isTest)
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

  charData.rareFlags = input > params.tgtCharShortList;
  rareWords = input(charData.rareFlags);
  charData.numRareWords = length(rareWords);
  charSeqs = charMap(rareWords);
  seqLens = cellfun(@(x) length(x), charSeqs);
  charData.numChars = sum(seqLens) + charData.numRareWords; % we also predict tgt eos
  assert(isequal(size(input), [size(tgtHidVecs, 2), size(tgtHidVecs, 3)]));
  
  % TODO: might need to make batch size smaller 
  if charData.numRareWords > 0
    charParams = params;
    charParams.numLayers = params.charNumLayers;
   
    tgtHidVecs = reshape(tgtHidVecs, params.lstmSize, []);
    embIndices = find(charData.rareFlags(:));
    
    % sort if the number of rare words is large
    if charData.numRareWords > params.batchSize
      [seqLens, sortedIndices] = sort(seqLens);
      embIndices = embIndices(sortedIndices);
      charSeqs = charSeqs(sortedIndices);
    else
      sortedIndices = 1 : charData.numRareWords;
    end
    charData.sortedIndices = sortedIndices;
    
    % split into batches
    charData.numBatches = floor((charData.numRareWords - 1) / params.batchSize) + 1;
    charData.batches = cell(charData.numBatches, 1);
    
    for ii=1:charData.numBatches
      startId = (ii-1)*params.batchSize + 1;
      endId = ii*params.batchSize;
      if endId > charData.numRareWords
        endId = charData.numRareWords;
      end
      charParams.curBatchSize = endId - startId + 1;
      
      charData.batches{ii}.params = charParams;
      charData.batches{ii}.rnnFlags = struct('decode', 1, 'test', isTest, 'attn', 0, 'feedInput', 0, 'charSrcRep', 0, 'charTgtGen', 0, ...
        'initEmb', tgtHidVecs(:, embIndices(startId:endId)));
      
      [charData.batches{ii}.batch, charData.batches{ii}.mask, charData.batches{ii}.maxLen, charData.batches{ii}.numSeqs] = rightPad(...
        charSeqs(startId:endId), seqLens(startId:endId), params.tgtCharEos, params.tgtCharSos);
      charData.batches{ii}.initState = createZeroState(charParams);
      [charData.batches{ii}.states, ~, ~] = rnnLayerForward(W_rnn, W_emb, charData.batches{ii}.initState, charData.batches{ii}.batch, ...
        charData.batches{ii}.mask, charParams, charData.batches{ii}.rnnFlags, [], [], []);
    end
  end
end