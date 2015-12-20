function [states, batch, mask, maxLen, numSeqs] = char2wordReps(W_rnn, W_emb, rareWords, charMap, charParams, isTest)
  
  charSeqs = charMap(rareWords);
  [batch, mask, maxLen, numSeqs] = leftPad(charSeqs, charParams.charSos, charParams.charEos);
  
  % TODO: sort & split batches
  rnnFlags = struct('decode', 0, 'test', isTest, 'attn', 0, 'feedInput', 0, 'char', 0);

  prevState = createZeroState(charParams);
  [states, ~, ~] = rnnLayerForward(W_rnn, W_emb, prevState, batch, mask, ...
    charParams, rnnFlags, [], [], []);
end

function [batch, mask, maxLen, numSeqs] = leftPad(seqs, padSymbol, eos)
  numSeqs = length(seqs);
  lens = cellfun(@(x) length(x), seqs);
  maxLen = max(lens);
  
  % append eos
  if eos > 0
    maxLen = maxLen + 1;
  end
  
  batch = padSymbol*ones(numSeqs, maxLen);
  for ii=1:numSeqs
    len = lens(ii);
    
    if eos > 0
      batch(ii, end-len:end-1) = seqs{ii}(1:len);
      batch(ii, end) = eos;
    else
      batch(ii, end-len+1:end) = seqs{ii}(1:len);      
    end
  end
  mask = batch ~= padSymbol;
end