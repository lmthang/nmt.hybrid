function [costs, grad] = lstmCostGrad(model, trainData, params, isTest)
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
  isChar = params.charShortList;
  if isChar
    % src
    if params.isBi
      [srcCharData] = charForward(model.W_src_char, model.W_emb_src_char, trainData.srcInput, params.srcCharMap, ...
        params.srcVocabSize, params.charShortList, params, isTest);
    end
    
    % tgt
    [tgtCharData] = charForward(model.W_tgt_char, model.W_emb_tgt_char, trainData.tgtInput, params.tgtCharMap, ...
        params.tgtVocabSize, params.charShortList, params, isTest);
  else
    srcCharData = [];
    tgtCharData = [];
  end
  
  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  %% encoder
  lastEncState = zeroState;
  if params.isBi
    encRnnFlags = struct('decode', 0, 'test', isTest, 'attn', params.attnFunc, 'feedInput', 0, 'char', isChar);
    [encStates, trainData, ~] = rnnLayerForward(model.W_src, model.W_emb_src, zeroState, trainData.srcInput, trainData.srcMask, ...
      params, encRnnFlags, trainData, model, srcCharData);
    lastEncState = encStates{end};
    
    % feed input
    if params.feedInput
      lastEncState{end}.softmax_h = zeroBatch;
    end
  end
  
  %% decoder
  decRnnFlags = struct('decode', 1, 'test', isTest, 'attn', params.attnFunc, 'feedInput', params.feedInput, 'char', isChar);
  [decStates, ~, attnInfos] = rnnLayerForward(model.W_tgt, model.W_emb_tgt, lastEncState, trainData.tgtInput, trainData.tgtMask, ...
    params, decRnnFlags, trainData, model, tgtCharData);
  
  %% softmax
  [costs.total, grad.W_soft, dec_top_grads] = softmaxCostGrad(decStates, model.W_soft, trainData.tgtOutput, trainData.tgtMask, ...
    params, isTest);
  costs.word = costs.total;
  
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
  [dc, dh, grad.W_tgt, grad.W_emb_tgt, grad.indices_tgt, attnGrad, grad.srcHidVecs, charGrad] = rnnLayerBackprop(model.W_tgt, ...
    decStates, lastEncState, dec_top_grads, zeroGrad, zeroGrad, trainData.tgtInput, trainData.tgtMask, params, decRnnFlags, ...
    attnInfos, trainData, model);
  if params.attnFunc % copy attention grads 
    [grad] = copyStruct(attnGrad, grad);
  end
  % char backprop
  if isChar
    [grad.W_char_tgt, grad.W_emb_tgt_char, grad.indices_tgt_char] = charBackprop(model.W_tgt_char, tgtCharData, charGrad);
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
    if isChar
      [grad.W_char_src, grad.W_emb_src_char, grad.indices_src_char] = charBackprop(model.W_src_char, srcCharData, charGrad);
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

function [charData] = charForward(W_rnn, W_emb, input, charMap, vocabSize, charShortList, params, isTest)  
  charData.rareFlags = input > charShortList;
  rareWords = unique(input(charData.rareFlags));
  
  charData.params = params;
  charData.params.numLayers = params.charNumLayers;
  charData.params.curBatchSize = length(rareWords);
  
  [charData.states, charData.batch, charData.mask, charData.maxLen, charData.numSeqs] = char2wordReps(W_rnn, W_emb, ...
    rareWords, charMap, charData.params, isTest);
  charData.rareWordMap = sparse(rareWords, 1, 1:length(rareWords), vocabSize, 1);
  charData.rareWordReps = charData.states{end}{end}.h_t;
end

function [grad_W_rnn, grad_W_emb, emb_indices] = charBackprop(W_rnn, charData, charGrad)
  assert(length(charGrad.indices) == charData.numSeqs);
  topGrads = cell(charData.maxLen, 1);
  topGrads{end} = charGrad.embs(:, charData.rareWordMap(charGrad.indices));
  
  % init state
  params = charData.params;
  zeroBatch = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  zeroState = cell(params.numLayers, 1);
  zeroGrad = cell(params.numLayers, 1);
  for ll=1:params.numLayers % layer
    zeroState{ll}.h_t = zeroBatch;
    zeroState{ll}.c_t = zeroBatch;
    zeroGrad{ll} = zeroBatch;
  end
  
  charRnnFlags = struct('decode', 0, 'attn', 0, 'feedInput', 0, 'char', 0);
  [~, ~, grad_W_rnn, grad_W_emb, emb_indices, ~, ~, ~] = rnnLayerBackprop(W_rnn, charData.states, zeroState, ...
  topGrads, zeroGrad, zeroGrad, charData.batch, charData.mask, charData.params, charRnnFlags, [], [], []);
end