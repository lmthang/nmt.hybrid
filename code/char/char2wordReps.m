function [states, batch, mask, maxLen, numSeqs] = char2wordReps(W_rnn, W_emb, rareWords, charMap, charParams, isTest)
  
  charSeqs = charMap(rareWords);
  [batch, mask, maxLen, numSeqs] = leftPad(charSeqs, charParams.charSos, charParams.charEos);
  
  % TODO: sort & split batches
  rnnFlags = struct('decode', 0, 'test', isTest, 'attn', 0, 'feedInput', 0, 'char', 0);

  prevState = createZeroState(charParams);
  [states, ~, ~] = rnnLayerForward(W_rnn, W_emb, prevState, batch, mask, ...
    charParams, rnnFlags, [], [], []);
end