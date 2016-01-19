function [charData] = multiBatchCharLayerForward(W_rnn, W_emb, charSeqs, seqLens, charData, params, isTest, isDecode, tgtHidVecs)
  charParams = params;
  charParams.numLayers = params.charNumLayers;

  if isDecode
    tgtHidVecs = reshape(tgtHidVecs, params.lstmSize, []);
    embIndices = find(charData.rareFlags(:));
  end

  % sort if the number of rare words is large
  if charData.numRareWords > params.batchSize
    [seqLens, sortedIndices] = sort(seqLens);
    charSeqs = charSeqs(sortedIndices);
    
    if isDecode
      embIndices = embIndices(sortedIndices);
    end
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

    if isDecode
      initEmb = tgtHidVecs(:, embIndices(startId:endId));
    else
      initEmb = [];
    end
    
    charData.batches{ii}.params = charParams;
    charData.batches{ii}.rnnFlags = struct('decode', isDecode, 'test', isTest, 'attn', 0, 'feedInput', 0, 'charSrcRep', 0, 'charTgtGen', 0, ...
      'initEmb', initEmb);

    if isDecode
      [charData.batches{ii}.batch, charData.batches{ii}.mask, charData.batches{ii}.maxLen, charData.batches{ii}.numSeqs] = rightPad(...
        charSeqs(startId:endId), seqLens(startId:endId), params.tgtCharEos, params.tgtCharSos);
    else
      [charData.batches{ii}.batch, charData.batches{ii}.mask, charData.batches{ii}.maxLen, charData.batches{ii}.numSeqs] = leftPad(...
        charSeqs(startId:endId), seqLens(startId:endId), params.srcCharSos, params.srcCharEos);
    end
    
    charData.batches{ii}.initState = createZeroState(charParams);
    [charData.batches{ii}.states, ~, ~] = rnnLayerForward(W_rnn, W_emb, charData.batches{ii}.initState, charData.batches{ii}.batch, ...
      charData.batches{ii}.mask, charParams, charData.batches{ii}.rnnFlags, [], [], []);
  end
end