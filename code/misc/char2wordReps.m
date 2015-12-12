function [wordReps] = char2wordReps(rareWords, W_rnn, W_emb, prevState, charMap, charVocab, padSymbol)
  charSeqs = charMap(rareWords);
  charVocab(charSeqs{1})
  [batch, mask] = rightBatch(charSeqs, padSymbol);
  
  % TODO: sort & split batches
  isDecoder = 0;
  isAttn = 0;
  attnData = [];
  [encStates, ~, ~] = rnnLayerForward(W_rnn, W_emb, prevState, batch, mask, ...
      params, isTest, isDecoder, isAttn, attnData, model);
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