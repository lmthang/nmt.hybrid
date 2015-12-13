function [wordReps] = char2wordReps(W_rnn, W_emb, rareWords, charMap, charParams, isTest)
  
  charSeqs = charMap(rareWords);
  [batch, mask] = rightBatch(charSeqs, charParams.srcSos);
  
  % TODO: sort & split batches
  isDecoder = 0;
  isAttn = 0;
  attnData = [];
  model = [];
  isChar = 0;
  charData = [];
  charParams.curBatchSize = length(rareWords);
  assert(charParams.curBatchSize == size(batch, 1));
  prevState = createZeroState(charParams);
  [encStates, ~, ~] = rnnLayerForward(W_rnn, W_emb, prevState, batch, mask, ...
    charParams, isTest, isDecoder, isAttn, attnData, model, isChar, charData);
  wordReps = encStates{end}{end}.h_t;
end

function [batch, mask] = rightBatch(seqs, padSymbol)
  numSeqs = length(seqs);
  lens = cellfun(@(x) length(x), seqs);
  maxLen = max(lens);
  batch = padSymbol*ones(numSeqs, maxLen);
  for ii=1:numSeqs
    len = lens(ii);
    batch(ii, maxLen-len+1:end) = seqs{ii}(1:len);      
  end
  mask = batch ~= padSymbol;
end