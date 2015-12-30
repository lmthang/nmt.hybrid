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
  else % monolingual
    srcMaxLen = 1;
  end
  
  params.curBatchSize = curBatchSize;
  params.srcMaxLen = srcMaxLen;

  [params] = setAttnParams(params);
  [grad, params] = initGrad(model, params);
  zeroBatch = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  
  % initState
  [zeroState] = createZeroState(params);
  
  % init costs
  costs = initCosts();
    
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
  
  
  %% char
  numChars = 0;
  if params.charTgtGen
    % char rnn
    [tgtCharData] = tgtCharLayerForward(model.W_tgt_char, model.W_emb_tgt_char, trainData.origTgtOutput, trainData.tgtHidVecs, params.tgtCharMap, ...
          params, isTest);
    numChars = sum(tgtCharData.mask(:));
    % char softmax
    charTgtOutput = [tgtCharData.batch(:, 2:end) params.tgtCharEos*ones(tgtCharData.numSeqs, 1)];
    [costs.char, grad.W_soft_char, topGrads_char] = softmaxCostGrad(tgtCharData.states, model.W_soft_char, charTgtOutput, tgtCharData.mask, ...
    params, isTest);
    costs.total = costs.total + costs.char;
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
  
  %% char
  if params.charTgtGen
    % TODO: remember in tgtCharLayerBackprop, ignore sos
    % char back prop
    [grad.W_tgt_char, grad.W_emb_tgt_char, grad.indices_tgt_char, grad_init_emb] = tgtCharLayerBackprop(model.W_tgt_char, tgtCharData, topGrads_char);
    
    % add top grads from tgt char
    count = 0;
    assert(length(dec_top_grads) == size(tgtCharData.rareFlags, 2));
    for tt=1:length(dec_top_grads)
      rareIndices = find(tgtCharData.rareFlags(:, tt));
      dec_top_grads{tt}(:, rareIndices) = dec_top_grads{tt}(:, rareIndices) + grad_init_emb(:, count+1:count+length(rareIndices));
      count = count + length(rareIndices);
    end
    assert(count == tgtCharData.numSeqs);
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