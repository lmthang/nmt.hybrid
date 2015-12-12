function [wordReps] = char2wordReps(W_rnn, W_emb, rareWords, prevState, charMap, charVocab, charParams, isTest)
  charSeqs = charMap(rareWords);
  charVocab(charSeqs{1})
  [batch, mask] = rightBatch(charSeqs, charParams.srcSos);
  
  % TODO: sort & split batches
  isDecoder = 0;
  isAttn = 0;
  attnData = [];
  model = [];
  [encStates, ~, ~] = rnnLayerForward(W_rnn, W_emb, prevState, batch, mask, ...
    charParams, isTest, isDecoder, isAttn, attnData, model);
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