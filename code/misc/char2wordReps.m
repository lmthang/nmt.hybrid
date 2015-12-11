function [wordReps] = char2wordReps(rareWords, charMap, charVocab, padSymbol)
  charSeqs = charMap(rareWords);
  charVocab(charSeqs{1})
  [batch] = rightBatch(charSeqs, padSymbol);
  
  % TODO: sort & split batches
%   [encStates, trainData, ~] = rnnLayerForward(encLen, model.W_src, model.W_emb_src, zeroState, trainData.srcInput, trainData.srcMask, ...
%       params, isTest, isDecoder, trainData, model);
end

function [batch] = rightBatch(seqs, padSymbol)
  numSeqs = length(seqs);
  lens = cellfun(@(x) length(x), seqs);
  maxLen = max(lens);
  batch = padSymbol*ones(numSeqs, maxLen);
  for ii=1:numSeqs
    len = lens(ii);
    batch(ii, maxLen-len+1:end) = seqs{ii}(1:len);      
  end
end