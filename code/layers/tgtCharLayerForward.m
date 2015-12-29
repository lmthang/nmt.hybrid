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

  % TODO: might need to make batch size smaller 
  if ~isempty(rareWords)    
    charData.params = params;
    charData.params.numLayers = params.charNumLayers;
    charData.params.curBatchSize = length(rareWords);

    tgtHidVecs = reshape(tgtHidVecs, params.lstmSize, []);
    initEmb = tgtHidVecs(:, charData.rareFlags);

    % TODO: sort & split batches
    [charData.batch, charData.mask, charData.maxLen, charData.numSeqs] = rightPad(charMap(rareWords), params.tgtCharEos, params.tgtCharSos);
    zeroState = createZeroState(charData.params);
    charData.rnnFlags = struct('decode', 1, 'test', isTest, 'attn', 0, 'feedInput', 0, 'charSrcRep', 0, 'charTgtGen', 0, 'initEmb', initEmb);
    [charData.states, ~, ~] = rnnLayerForward(W_rnn, W_emb, zeroState, charData.batch, charData.mask, charData.params, charData.rnnFlags, [], [], []);
  end
end