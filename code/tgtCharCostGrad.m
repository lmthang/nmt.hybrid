function [totalCharCost, charGrad, numChars] = tgtCharCostGrad(decStates, attnInfos, model, input, charMap, params, isTest)
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

  rareFlags = input > params.tgtCharShortList;
  rareWords = input(rareFlags);
  numRareWords = length(rareWords);
  charSeqs = charMap(rareWords);
  seqLens = cellfun(@(x) length(x), charSeqs);
  
  if params.debug
    fprintf(2, '# tgtCharCostGrad, %s: %d words\n', gpuInfo(params.gpu), numRareWords);
  end
  
  % return vars
  numChars = sum(seqLens) + numRareWords; % we also predict tgt eos, so add numRareWords
  totalCharCost = 0;
  charGrad.W_soft = 0;
  charGrad.W_tgt = cell(params.charNumLayers, 1);
  initEmb_grad = zeroMatrix([params.lstmSize, numRareWords], params.isGPU, params.dataType);
  % there are also charGrad.W_emb_tgt_char, charGrad.indices_tgt_char
  % accumulated at the end
  if numRareWords == 0
    return;
  end
  
  % tmp vars
  % preallocation: when printing out logs, we often use less than <
  % numRareWords columns.
  grad_W_emb_char_total = zeroMatrix([params.lstmSize, numRareWords*2], params.isGPU, params.dataType);
  indices_char_total = zeros(numRareWords*2, 1);
  charCount = 0;
  
  
  if params.debug
    fprintf(2, '  after init, %s\n', gpuInfo(params.gpu));
  end
  
  if numRareWords > 0
    charParams = params;
    charParams.numLayers = params.charNumLayers;
    
    % tgtWordStates
    decLen = size(input, 2);
    rareWordCount = 0;
    if params.charFeedOpt
      assert(params.attnFunc > 0); % only support attention for now
      tgtWordStates_prev = zeroMatrix([2*params.lstmSize, numRareWords], params.isGPU, params.dataType);
    else
      tgtWordStates = zeroMatrix([params.lstmSize, numRareWords], params.isGPU, params.dataType);
    end
    for tt=1:decLen  
      rareIndices = find(rareFlags(:, tt));
      
      if params.charFeedOpt
        assert(params.attnFunc > 0); % only support attention for now
        tgtWordStates_prev(:, rareWordCount+1:rareWordCount+length(rareIndices)) = attnInfos{tt}.input(:, rareIndices);
      else
        tgtWordStates(:, rareWordCount+1:rareWordCount+length(rareIndices)) = decStates{tt}{end}.softmax_h(:, rareIndices);
      end
      
      rareWordCount = rareWordCount + length(rareIndices);
    end
    assert(rareWordCount == numRareWords);

    if params.charFeedOpt % a separate transformer for attention!
      tgtWordStates = hiddenLayerForward(model.W_h_char, tgtWordStates_prev, params.nonlinear_f); 
    end
    
    % sort if the number of rare words is large
    if numRareWords > params.batchSize
      [seqLens, sortedIndices] = sort(seqLens);
      charSeqs = charSeqs(sortedIndices);
    else
      sortedIndices = 1:numRareWords;
    end

    % split into batches
    numBatches = floor((numRareWords - 1) / params.batchSize) + 1;
    prevBatchSize = -1;
    rareWordCount = 0;
    for ii=1:numBatches
      startId = (ii-1)*params.batchSize + 1;
      endId = ii*params.batchSize;
      if endId > numRareWords
        endId = numRareWords;
      end
      curBatchSize = endId - startId + 1;
      
      % init: update setting with respect to new mini-batch size only
      if curBatchSize ~= prevBatchSize
        charParams.curBatchSize = curBatchSize;
        zeroBatch = zeroMatrix([charParams.lstmSize, charParams.curBatchSize], params.isGPU, params.dataType);
        charInitState = cell(charParams.numLayers, 1);
        for ll=1:charParams.numLayers % layer
          charInitState{ll}.h_t = zeroBatch;
          charInitState{ll}.c_t = zeroBatch;
        end
      end
      
      % feed hidden state from word-level RNN
      charInitState{1}.h_t = tgtWordStates(:, sortedIndices(startId:endId)); % IMPORTANT: don't put this in the above if!
      
      % prepare data
      [charBatch, charMask, maxLen, ~] = rightPad(charSeqs(startId:endId), seqLens(startId:endId), params.tgtCharEos, params.tgtCharSos);
      if params.debug
        fprintf(2, '  batch %d, %s: curBatchSize %d, maxLen %d\n', ii, gpuInfo(params.gpu), curBatchSize, maxLen);
      end
      
      %% forward %%
      charRnnFlags = struct('decode', 1, 'test', isTest, 'attn', 0, 'feedInput', 0, 'charSrcRep', 0, 'charTgtGen', 0);
      [charStates, ~, ~] = rnnLayerForward(model.W_tgt_char, model.W_emb_tgt_char, charInitState, charBatch, charMask, charParams, charRnnFlags, [], [], []);
  
      %% softmax
      charTgtOutput = [charBatch(:, 2:end) params.tgtCharEos*ones(charParams.curBatchSize, 1)];
      [cost_char, grad_W_soft_char, topGrads] = softmaxCostGrad(charStates, model.W_soft_char, charTgtOutput, charMask, params, isTest);
      
      % cost
      totalCharCost = totalCharCost + params.charWeight * cost_char;
      
      %% backprop %%
      % we will scale by charWeight at the very end %
      if isTest==0
        % W_soft
        if ii==1
          charGrad.W_soft = grad_W_soft_char;
        else
          charGrad.W_soft = charGrad.W_soft + grad_W_soft_char;
        end

        %% char rnn back prop
        % init state
        if curBatchSize ~= prevBatchSize
          zeroGrad = cell(charParams.numLayers, 1);
          for ll=1:charParams.numLayers % layer
            zeroGrad{ll} = zeroBatch;
          end
        end
        [~, dh_char, grad_W_rnn_char, grad_W_emb_char, indices_char, ~, ~, ~] = rnnLayerBackprop(model.W_tgt_char, charStates, charInitState, ...
        topGrads, zeroGrad, zeroGrad, charBatch, charMask, charParams, charRnnFlags, [], [], [], []);
        
        % W_tgt
        if ii==1
          charGrad.W_tgt = grad_W_rnn_char;
        else
          for ll=1:charParams.numLayers
            charGrad.W_tgt{ll} = charGrad.W_tgt{ll} + grad_W_rnn_char{ll};
          end
        end
        
        % char emb
        numDistinctChars = length(indices_char);
        grad_W_emb_char_total(:, charCount+1:charCount+numDistinctChars) = grad_W_emb_char;
        indices_char_total(charCount+1:charCount+numDistinctChars) = indices_char;
        charCount = charCount + numDistinctChars;
        
        % init emb
        initEmb_grad(:, rareWordCount+1:rareWordCount+curBatchSize) = dh_char{1};
        rareWordCount = rareWordCount + curBatchSize;
        
        prevBatchSize = curBatchSize;
      end % end isTest
    end % end for numBatches
    
    %% final accumulation %%
    if isTest==0
      assert(rareWordCount == numRareWords);
      
      grad_W_emb_char_total(:, charCount+1:end) = [];
      indices_char_total(charCount+1:end) = [];
      [charGrad.W_emb_tgt_char, charGrad.indices_tgt_char] = aggregateMatrix(grad_W_emb_char_total, indices_char_total, params.isGPU, params.dataType);

      % due to sorting
      if numRareWords > params.batchSize
        initEmb_grad(:, sortedIndices) = initEmb_grad;
      end
      
      %% scale by charWeight
      if params.charWeight ~= 1
        charGrad.W_soft = params.charWeight * charGrad.W_soft;
        for ll=1:charParams.numLayers
          charGrad.W_tgt{ll} = params.charWeight * charGrad.W_tgt{ll};
        end
        charGrad.W_emb_tgt_char = params.charWeight * charGrad.W_emb_tgt_char;
        initEmb_grad = params.charWeight * initEmb_grad;
      end
      
      if params.charFeedOpt % a separate transformer for attention! 
        [charGrad.initAttnInput, charGrad.W_h_char] = hiddenLayerBackprop(model.W_h_char, initEmb_grad, tgtWordStates_prev, params.nonlinear_f_prime, tgtWordStates);
      else
        charGrad.initEmb = initEmb_grad;
        charGrad.initAttnInput = [];
      end
      
      if params.debug
        fprintf(2, '  end tgtCharCostGrad, %s, charCount %d\n', gpuInfo(params.gpu), charCount);
      end
    end
  end % end if numRareWords
  
  charGrad.rareFlags = rareFlags;
  charGrad.numRareWords = numRareWords;
end