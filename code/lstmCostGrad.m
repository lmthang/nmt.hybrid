function [costs, grad, charInfo] = lstmCostGrad(model, trainData, params, isTest)
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
  charInfo.numChars = 0;
  if params.charTgtGen
    [costs.char, tgtCharGrad, charInfo.numChars] = tgtCharCostGrad(decStates, attnInfos, model, trainData.origTgtOutput, params.tgtCharMap, params, isTest);
    costs.total = costs.total + costs.char;
    if isTest==0
      if tgtCharGrad.numRareWords>0
        grad.W_soft_char = tgtCharGrad.W_soft;
        grad.W_tgt_char = tgtCharGrad.W_tgt;
        grad.W_emb_tgt_char = tgtCharGrad.W_emb_tgt_char;
        grad.indices_tgt_char = tgtCharGrad.indices_tgt_char;
        
        if params.charFeedOpt
          assert(params.attnFunc>0);
          grad.W_h_char = tgtCharGrad.W_h_char;
        else
          % add top grads from tgt char
          rareCount = 0;
          decLen = length(dec_top_grads);
          for tt=1:decLen
            rareIndices = find(tgtCharGrad.rareFlags(:, tt));
            dec_top_grads{tt}(:, rareIndices) = dec_top_grads{tt}(:, rareIndices) + tgtCharGrad.initEmb(:, rareCount+1:rareCount+length(rareIndices));
            rareCount = rareCount + length(rareIndices);
          end
          assert(rareCount == tgtCharGrad.numRareWords);
        end
      else
        grad.indices_tgt_char = [];
      end
      
      % clear tgtCharGrad;
      if params.debug
        fprintf(2, '# after clearing tgtCharGrad, %s\n', gpuInfo(params.gpu));
      end
    end
  else
    tgtCharGrad = [];
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
    attnInfos, trainData, model, tgtCharGrad);
  if params.attnFunc % copy attention grads 
    [grad] = copyStruct(attnGrad, grad);
  end
  
  %% encoder
  if params.isBi
    enc_top_grads = cell(srcMaxLen - 1, 1);
    for tt=1:params.numSrcHidVecs % attention
      enc_top_grads{tt} = grad.srcHidVecs(:,:,tt);
    end
    
    [~, ~, grad.W_src, grad.W_emb_src, grad.indices_src, ~, ~, srcCharGrad] = rnnLayerBackprop(model.W_src, encStates, zeroState, ...
    enc_top_grads, dc, dh, trainData.srcInput, trainData.srcMask, params, encRnnFlags, attnInfos, trainData, model, []);
  
    % char backprop
    if params.charSrcRep
      if srcCharData.numRareWords > 0
        [grad.W_src_char, grad.W_emb_src_char, grad.indices_src_char] = srcCharLayerBackprop(model.W_src_char, srcCharData, srcCharGrad);
      else
        grad.indices_src_char = [];
      end
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
