function [totalCharCost, charGrad, numChars, rareFlags, numRareWords] = tgtCharCostGrad(W_soft_char, W_rnn, W_emb, input, tgtHidVecs, charMap, params, isTest)
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
  charGrad.initEmb = zeroMatrix([params.lstmSize, numRareWords], params.isGPU, params.dataType);
  % there are also charGrad.W_emb_tgt_char, charGrad.indices_tgt_char
  % accumulated at the end
  
  % tmp vars
  % preallocation: when printing out logs, we often use less than <
  % numRareWords columns.
  grad_W_emb_char_total = zeroMatrix([params.lstmSize, numRareWords*2], params.isGPU, params.dataType);
  indices_char_total = zeros(numRareWords*2, 1);
  charCount = 0;
  rareWordCount = 0;
  
  if params.debug
    fprintf(2, '  after init, %s\n', gpuInfo(params.gpu));
  end
  
  if numRareWords > 0
    charParams = params;
    charParams.numLayers = params.charNumLayers;
  
    % tgtHidVecs
    assert(isequal(size(input), [size(tgtHidVecs, 2), size(tgtHidVecs, 3)]));
    tgtHidVecs = reshape(tgtHidVecs, params.lstmSize, []);
    tgtHidIndices = find(rareFlags(:));

    % sort if the number of rare words is large
    if numRareWords > params.batchSize
      [seqLens, sortedIndices] = sort(seqLens);
      charSeqs = charSeqs(sortedIndices);
      tgtHidIndices = tgtHidIndices(sortedIndices);
    end

    % split into batches
    numBatches = floor((numRareWords - 1) / params.batchSize) + 1;
    prevBatchSize = -1;
    for ii=1:numBatches
      startId = (ii-1)*params.batchSize + 1;
      endId = ii*params.batchSize;
      if endId > numRareWords
        endId = numRareWords;
      end
      curBatchSize = endId - startId + 1;
      
      % update setting with respect to new mini-batch size
      if curBatchSize ~= prevBatchSize
        charParams.curBatchSize = curBatchSize;
        % charInitState = createZeroState(charParams);
        zeroBatch = zeroMatrix([params.lstmSize, curBatchSize], params.isGPU, params.dataType);
        charInitState = cell(params.numLayers, 1);
        for ll=1:params.numLayers % layer
          % feed hidden state from word-level RNN
          charInitState{ll}.h_t = tgtHidVecs(:, tgtHidIndices(startId:endId));
          charInitState{ll}.c_t = zeroBatch;
        end
      end
      charRnnFlags = struct('decode', 1, 'test', isTest, 'attn', 0, 'feedInput', 0, 'charSrcRep', 0, 'charTgtGen', 0);
      
      % prepare data
      [charBatch, charMask, maxLen, ~] = rightPad(charSeqs(startId:endId), seqLens(startId:endId), params.tgtCharEos, params.tgtCharSos);

      if params.debug
        fprintf(2, '  batch %d, %s: curBatchSize %d, maxLen %d\n', ii, gpuInfo(params.gpu), curBatchSize, maxLen);
      end
      
      %% forward %%
      [charStates, ~, ~] = rnnLayerForward(W_rnn, W_emb, charInitState, charBatch, charMask, charParams, charRnnFlags, [], [], []);
      
      if params.debug
        fprintf(2, '    after forward, %s\n', gpuInfo(params.gpu));
      end
  
      %% softmax
      charTgtOutput = [charBatch(:, 2:end) params.tgtCharEos*ones(charParams.curBatchSize, 1)];
      [cost_char, grad_W_soft_char, topGrads_char] = softmaxCostGrad(charStates, W_soft_char, charTgtOutput, charMask, params, isTest);
      
      if params.debug
        fprintf(2, '    after softmax, %s\n', gpuInfo(params.gpu));
      end
      
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
          
          if params.debug
            charBatch
            charMask
          end
        end
        [~, dh_char, grad_W_rnn_char, grad_W_emb_char, indices_char, ~, ~, ~] = rnnLayerBackprop(W_rnn, charStates, charInitState, ...
        topGrads_char, zeroGrad, zeroGrad, charBatch, charMask, charParams, charRnnFlags, [], [], []);
        % initEmb_batch = charRnnGrad.initEmb;
        for ll=1:charParams.numLayers
          if ll == 1
            initEmb_batch = dh_char{ll};
          else
            initEmb_batch = initEmb_batch + dh_char{ll};
          end
        end
        
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
        charGrad.initEmb(:, rareWordCount+1:rareWordCount+curBatchSize) = initEmb_batch;
        rareWordCount = rareWordCount + curBatchSize;
        
        prevBatchSize = curBatchSize;
        
        if params.debug
          fprintf(2, '    after backprop, %s\n', gpuInfo(params.gpu));
        end
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
        charGrad.initEmb(:, sortedIndices) = charGrad.initEmb;
      end
      
      %% scale by charWeight
      if params.charWeight ~= 1
        charGrad.W_soft = params.charWeight * charGrad.W_soft;
        for ll=1:charParams.numLayers
          charGrad.W_tgt{ll} = params.charWeight * charGrad.W_tgt{ll};
        end
        charGrad.W_emb_tgt_char = params.charWeight * charGrad.W_emb_tgt_char;
        charGrad.initEmb = params.charWeight * charGrad.initEmb;
      end
      
      if params.debug
        fprintf(2, '  end tgtCharCostGrad, %s, charCount %d\n', gpuInfo(params.gpu), charCount);
      end
    end
  end % end if numRareWords
end