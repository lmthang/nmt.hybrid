function [costs, grad, numChars] = lstmCostGrad(model, trainData, params, isTest)
%%%
%
% Compute cost/grad for LSTM. 
% When params.predictPos>0, returns costs.pos and costs.word
% If isTest==1, this method only computes cost (for testing purposes).
%
% Thang Luong @ 2014, 2015, <lmthang@stanford.edu>
%
%%%

  %%%%%%%%%%%%
  %%% INIT %%%
  %%%%%%%%%%%%
  curBatchSize = size(trainData.tgtInput, 1);
  if params.isBi
    srcMaxLen = trainData.srcMaxLen;
  else
    srcMaxLen = 1;
  end
  
  params.curBatchSize = curBatchSize;
  params.srcMaxLen = srcMaxLen;

  [params] = setAttnParams(params);
  [grad, params] = initGrad(model, params);
  zeroBatch = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  
  % init states
  [zeroState] = createZeroState(params);
  
  % init costs
  costs = initCosts(params);
    
  % char
  if params.charOpt
    % tgt
    if params.charTgtGen
      trainData.origTgtOutput = trainData.tgtOutput;
      trainData.origTgtInput = trainData.tgtInput;
    end
    trainData.tgtOutput(trainData.tgtOutput > params.tgtCharShortList) = params.tgtUnk;
    trainData.tgtInput(trainData.tgtInput > params.tgtCharShortList) = params.tgtUnk;
  end
  
  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  %% encoder
  lastEncState = zeroState;
  if params.isBi
    [encStates, lastEncState, encRnnFlags, trainData, srcCharData] = encoderLayerForward(model, zeroState, trainData, params, isTest);
  end
  
  %% decoder
  decRnnFlags = struct('decode', 1, 'test', isTest, 'attn', params.attnFunc, 'feedInput', params.feedInput, 'charSrcRep', params.charSrcRep, ...
      'charTgtGen', params.charTgtGen, 'initEmb', []);  
  [decStates, trainData, attnInfos] = rnnLayerForward(model.W_tgt, model.W_emb_tgt, lastEncState, trainData.tgtInput, trainData.tgtMask, ...
    params, decRnnFlags, trainData, model, []);
  
  %% softmax
  [costs.word, grad.W_soft, dec_top_grads] = softmaxCostGrad(decStates, model.W_soft, trainData.tgtOutput, trainData.tgtMask, ...
    params, isTest);
  costs.total = costs.word;
  
  
  %% tgt char foward / backprop %%
  numChars = 0;
  if params.charTgtGen
    [costs.char, charGrad, numChars, tgtCharRareFlags, tgtNumRareWords] = tgtCharCostGrad(model.W_soft_char, model.W_tgt_char, model.W_emb_tgt_char, ...
      trainData.origTgtOutput, trainData.tgtHidVecs, params.tgtCharMap, params, isTest);
    costs.total = costs.total + costs.char;
    
    if isTest==0
      % W_soft_char
      grad.W_soft_char = grad.W_soft_char + charGrad.W_soft;

      % W_tgt_char
      for ll=1:params.charNumLayers
        grad.W_tgt_char{ll} = grad.W_tgt_char{ll} + charGrad.W_tgt{ll};
      end

      grad.W_emb_tgt_char = charGrad.W_emb_tgt_char;
      grad.indices_tgt_char = charGrad.indices_tgt_char;
      
      % add top grads from tgt char
      rareCount = 0;
      assert(length(dec_top_grads) == size(tgtCharRareFlags, 2));
      for tt=1:length(dec_top_grads)
        rareIndices = find(tgtCharRareFlags(:, tt));
        dec_top_grads{tt}(:, rareIndices) = dec_top_grads{tt}(:, rareIndices) + charGrad.initEmb(:, rareCount+1:rareCount+length(rareIndices));
        rareCount = rareCount + length(rareIndices);
      end
      assert(rareCount == tgtNumRareWords);
      clear charGrad;
      if params.debug
        fprintf(2, '# after clearing charGrad, %s\n', gpuInfo(params.gpu));
      end
    end
  end
  
  if isTest==1 % don't compute grad
    return;
  end
  
  %%%%%%%%%%%%%%%%%%%%%
  %%% BACKWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%%
  % h_t and c_t gradients accumulate over time per layer
  zeroGrad = cell(params.numLayers, 1); 
  for ll=1:params.numLayers
    zeroGrad{ll} = zeroBatch;
  end
  
  %% decoder
  [dc, dh, grad.W_tgt, grad.W_emb_tgt, grad.indices_tgt, attnGrad, grad.srcHidVecs, ~] = rnnLayerBackprop(model.W_tgt, ...
    decStates, lastEncState, dec_top_grads, zeroGrad, zeroGrad, trainData.tgtInput, trainData.tgtMask, params, decRnnFlags, ...
    attnInfos, trainData, model);
  if params.attnFunc % copy attention grads 
    [grad] = copyStruct(attnGrad, grad);
  end
  
  %% encoder
  if params.isBi
    enc_top_grads = cell(srcMaxLen - 1, 1);
    for tt=1:params.numSrcHidVecs % attention
      enc_top_grads{tt} = grad.srcHidVecs(:,:,tt);
    end
    
    [~, ~, grad.W_src, grad.W_emb_src, grad.indices_src, ~, ~, charGrad] = rnnLayerBackprop(model.W_src, encStates, zeroState, ...
    enc_top_grads, dc, dh, trainData.srcInput, trainData.srcMask, params, encRnnFlags, attnInfos, trainData, model);
  
    % char backprop
    if params.charSrcRep
      [grad.W_src_char, grad.W_emb_src_char, grad.indices_src_char] = srcCharLayerBackprop(model.W_src_char, srcCharData, charGrad);
    end
  end

    
  % remove unused variables
  if params.attnFunc>0
    grad = rmfield(grad, 'srcHidVecs');
  end
end

function [grad, params] = initGrad(model, params)
  %% grad
  for ii=1:length(params.varsDenseUpdate)
    field = params.varsDenseUpdate{ii};
    if iscell(model.(field))
      for jj=1:length(model.(field)) % cell, like W_src, W_tgt
        grad.(field){jj} = zeroMatrix(size(model.(field){jj}), params.isGPU, params.dataType);
      end
    else
      grad.(field) = zeroMatrix(size(model.(field)), params.isGPU, params.dataType);
    end
  end
  
  %% backprop to src hidden states for attention and positional models
  if params.attnFunc>0
    % we extract trainData.srcHidVecs later, which contains all src hidden states, lstmSize * curBatchSize * numSrcHidVecs 
    grad.srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
  end
end


%     % char rnn
%     [tgtCharData] = tgtCharLayerForward(model.W_tgt_char, model.W_emb_tgt_char, trainData.origTgtOutput, trainData.tgtHidVecs, params.tgtCharMap, ...
%           params, isTest);
%     numChars = tgtCharData.numChars;
%     
%     % char softmax
%     costs.char = 0.0;
%     grad_init_emb = zeroMatrix([params.lstmSize, tgtCharData.numRareWords], params.isGPU, params.dataType);
%     grad.W_emb_tgt_char = zeroMatrix([params.lstmSize, tgtCharData.numRareWords*10], params.isGPU, params.dataType);
%     grad.indices_tgt_char = zeros(tgtCharData.numRareWords*10, 1);
%     charCount = 0;
%     rareWordCount = 0;
%     for ii=1:tgtCharData.numBatches
%       batchCharData = tgtCharData.batches{ii};
%       charTgtOutput = [batchCharData.batch(:, 2:end) params.tgtCharEos*ones(batchCharData.numSeqs, 1)];
%       [cost_char, grad_W_soft_char, topGrads_char] = softmaxCostGrad(batchCharData.states, model.W_soft_char, charTgtOutput, batchCharData.mask, ...
%         params, isTest);
%     
%       % cost
%       costs.char = costs.char + params.charWeight * cost_char;
%       
%       % backprop
%       if isTest==0
%         % W_soft_char
%         if ii==1
%           grad.W_soft_char = params.charWeight * grad_W_soft_char;
%         else
%           grad.W_soft_char = grad.W_soft_char + params.charWeight * grad_W_soft_char;
%         end
% 
%         % topGrads_char
%         for tt=1:length(topGrads_char)
%           topGrads_char{tt} = params.charWeight * topGrads_char{tt};
%         end
% 
%         % char back prop
%         [grad_W_tgt_char, grad_W_emb_tgt_char_batch, indices_tgt_char_batch, grad_init_emb_batch] = tgtCharLayerBackprop(...
%           model.W_tgt_char, batchCharData, topGrads_char);
%         
%         % W_tgt_char
%         if ii==1
%           grad.W_tgt_char = grad_W_tgt_char;
%         else
%           for ll=1:length(grad.W_tgt_char)
%             grad.W_tgt_char{ll} = grad.W_tgt_char{ll} + grad_W_tgt_char{ll};
%           end
%         end
%         
%         % char emb
%         numDistinctChars = length(indices_tgt_char_batch);
%         grad.W_emb_tgt_char(:, charCount+1:charCount+numDistinctChars) = grad_W_emb_tgt_char_batch;
%         grad.indices_tgt_char(charCount+1:charCount+numDistinctChars) = indices_tgt_char_batch;
%         charCount = charCount + numDistinctChars;
%         
%         % init emb
%         grad_init_emb(:, rareWordCount+1:rareWordCount+batchCharData.params.curBatchSize) = grad_init_emb_batch;
%         rareWordCount = rareWordCount + batchCharData.params.curBatchSize;
%       end
%     end % for batch
%     
%     % test
%     if isTest==0
%       grad.W_emb_tgt_char(:, charCount+1:end) = [];
%       grad.indices_tgt_char(charCount+1:end) = [];
%       [grad.W_emb_tgt_char, grad.indices_tgt_char] = aggregateMatrix(grad.W_emb_tgt_char, grad.indices_tgt_char, params.isGPU, params.dataType);
% 
%       % due to sorting
%       grad_init_emb(:, tgtCharData.sortedIndices) = grad_init_emb;
%     end
%     
%     clear topGrads_char;
%     clear grad_W_tgt_char;
%     clear grad_W_emb_tgt_char_batch;
%     clear grad_init_emb_batch;
%     clear indices_tgt_char_batch;
%     
%     costs.total = costs.total + costs.char;