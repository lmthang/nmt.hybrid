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
  tgtMaxLen = trainData.tgtMaxLen;
  curBatchSize = size(trainData.input, 1);
  if params.isBi
    srcMaxLen = trainData.srcMaxLen;
    trainData.srcTotalWordCount = sum(trainData.srcLens);
  else % monolingual
    srcMaxLen = 1;
  end
  T = srcMaxLen+tgtMaxLen-1;
  
  input = trainData.input;
  inputMask = trainData.inputMask;
  trainData.tgtTotalWordCount = sum(trainData.tgtLens);
  
  trainData.isTest = isTest;
  trainData.T = T;
  trainData.srcMaxLen = srcMaxLen;
  trainData.curBatchSize = curBatchSize;
  
  params.curBatchSize = curBatchSize;
  params.srcMaxLen = srcMaxLen;
  params.T = T;
  [grad, params] = initGrad(model, params);
  zeroState = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  
  % lstm states over time
  lstmStates = cell(T, 1);
  % initState
  prevState = cell(params.numLayers, 1);
  for ll=1:params.numLayers % layer
    prevState{ll}.h_t = zeroState;
    prevState{ll}.c_t = zeroState;
  end
  
  % init costs
  costs = initCosts();
  
  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  
  % attention
  if params.attnFunc>0
    if params.attnGlobal % global
      trainData.srcMaskedIds = [];
    end
    trainData.srcHidVecsOrig = zeroMatrix([params.lstmSize, curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
  end
  
  % prepare mask
  maskInfo = cell(T, 1);
  for tt=1:T
    maskInfo{tt}.mask = inputMask(:, tt)'; % curBatchSize * 1
    maskInfo{tt}.unmaskedIds = find(maskInfo{tt}.mask);
    maskInfo{tt}.maskedIds = find(~maskInfo{tt}.mask);
  end
  
  %% encoder
  encLen = srcMaxLen - 1;
  if params.isBi
    isDecoder = 0;
    [lstmStates(1:encLen), ~] = rnnLayerForward(encLen, model.W_src, model.W_emb_src, prevState, input(:, 1:encLen), maskInfo(1:encLen), params, isTest, isDecoder, trainData, model);
    prevState = lstmStates{encLen};
    
    % attention
    if params.attnFunc>0
      % Record src hidden states
      for tt=1:params.numSrcHidVecs
        trainData.srcHidVecsOrig(:, :, tt) = lstmStates{tt}{params.numLayers}.h_t;
      end
      
      if params.attnGlobal == 0 % local
        trainData.srcHidVecs = trainData.srcHidVecsOrig;
      end
    end
  end
  
  %% decoder
  decLen = T - srcMaxLen + 1;
  isDecoder = 1;
  [lstmStates(encLen+1:T), attnInfos] = rnnLayerForward(decLen, model.W_tgt, model.W_emb_tgt, prevState, input(:, encLen+1:T), maskInfo(encLen+1:T), params, isTest, isDecoder, trainData, model);
  
  %% softmax
  [costs.total, grad.W_soft, grad_softmax_all] = softmaxCostGrad(decLen, lstmStates(encLen+1:T), attnInfos, model.W_soft, trainData.tgtOutput, maskInfo(encLen+1:T), params, isTest);
  costs.word = costs.total;
  
  if isTest==1 % don't compute grad
    return;
  end
  
  %%%%%%%%%%%%%%%%%%%%%
  %%% BACKWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%%
  % h_t and c_t gradients accumulate over time per layer
  dh = cell(params.numLayers, 1);
  dc = cell(params.numLayers, 1); 
  for ll=params.numLayers:-1:1 % layer
    dh{ll} = zeroState;
    dc{ll} = zeroState;
  end

  % emb grad
  if params.isBi
    allEmbGrads_src = zeroMatrix([params.lstmSize, trainData.srcTotalWordCount], params.isGPU, params.dataType);
    allEmbIndices_src = zeros(trainData.srcTotalWordCount, 1);
    wordCount_src = 0;
  end
  % update the decoder
  allEmbGrads_tgt = zeroMatrix([params.lstmSize, trainData.tgtTotalWordCount], params.isGPU, params.dataType);
  allEmbIndices_tgt = zeros(trainData.tgtTotalWordCount, 1);
  wordCount_tgt = 0;
  
  % NOTE: IMPORTANT for tt first, then for ll in other for attn3,4, pos2 models to work
  W_layers = model.W_tgt;
  for tt=T:-1:1 % time
    unmaskedIds = maskInfo{tt}.unmaskedIds;
    maskedIds = maskInfo{tt}.maskedIds;
    tgtPos = tt-srcMaxLen+1;
    
    % switch to encoder. NOTE: we assume W_layers has been set to 
    if tt==(srcMaxLen-1)
      W_layers = model.W_src;
    end
    
    %% softmax_h -> h_t: at the top layer
    if (tt>=srcMaxLen)
      if params.attnFunc
        % softmax_h -> h_t
        h2sInfo = attnInfos{tgtPos};
        [grad_tgt_ht, attnGrad, grad_srcHidVecs] = attnLayerBackprop(model, grad_softmax_all{tgtPos}, trainData, h2sInfo, params, maskInfo{tt});
        if params.assert
          assert(computeSum(grad_tgt_ht(:, maskedIds), params.isGPU)==0);
        end
      
        fields = fieldnames(attnGrad);
        for ii=1:length(fields)
          field = fields{ii};
          if tt==T
            grad.(field) = attnGrad.(field);
          else
            grad.(field) = grad.(field) + attnGrad.(field);
          end
        end

        % attention models: srcHidVecs
        if params.attnFunc
          if params.attnGlobal 
            grad.srcHidVecs = grad.srcHidVecs + grad_srcHidVecs;
          else
            grad.srcHidVecs = reshape(grad.srcHidVecs, params.lstmSize, []);
            grad_srcHidVecs = reshape(grad_srcHidVecs, params.lstmSize, []);
            grad.srcHidVecs(:, h2sInfo.linearIdAll) = grad.srcHidVecs(:, h2sInfo.linearIdAll) + grad_srcHidVecs(:, h2sInfo.linearIdSub);
            grad.srcHidVecs = reshape(grad.srcHidVecs, [params.lstmSize, params.curBatchSize, params.numSrcHidVecs]);
          end
        end
      else
        grad_tgt_ht = grad_softmax_all{tgtPos};
      end
      
      % get signals from the softmax layer
      dh{params.numLayers} = dh{params.numLayers} + grad_tgt_ht;
    end
    
    % attention/pos models: get feedback from grad.srcHidVecs
    if tt<=params.numSrcHidVecs 
      dh{params.numLayers} = dh{params.numLayers} + grad.srcHidVecs(:,:,tt);
    end
    
    for ll=params.numLayers:-1:1 % layer
      W = W_layers{ll};
      
      %% cell backprop
      if tt==1
        c_t_1 = [];
      else
        c_t_1 = lstmStates{tt-1}{ll}.c_t; %all_c_t{ll, tt-1};
      end
      c_t = lstmStates{tt}{ll}.c_t; %all_c_t{ll, tt};
      %lstm = lstms{ll, tt};
      
      [lstm_grad] = lstmLayerBackprop(W, lstmStates{tt}{ll}, c_t, c_t_1, dc{ll}, dh{ll}, tt, zeroState, maskedIds, params); % ll, tt, srcMaxLen, 
      
      dc{ll} = lstm_grad.dc;
      
      % assert
      if params.assert
        assert(computeSum(lstm_grad.input(:, maskedIds), params.isGPU)==0);
      end
      
      %% grad.W_src / grad.W_tgt
      if (tt>=srcMaxLen)
        grad.W_tgt{ll} = grad.W_tgt{ll} + lstm_grad.W;
      else
        grad.W_src{ll} = grad.W_src{ll} + lstm_grad.W;
      end
      
      %% input grad: lstm_grad.input = [x_t; h_t] for normal models
      % NOTE: important, here we only do assignment to initialize the hidden grad at layer ll for the previous time step. 
      % Later, when we go back one time step, we will accumulate.
      dh{ll} = lstm_grad.input(end-params.lstmSize+1:end, :); 
      if ll==1 % collect embedding grad
        % feed softmax vector
        if params.feedInput && tt>srcMaxLen % for tt==srcMaxLen, we feed zero vector
          grad_softmax_all{tgtPos-1} = grad_softmax_all{tgtPos-1} + lstm_grad.input(params.lstmSize+1:2*params.lstmSize, :);
        end
        
        %% emb grad
        embIndices = input(unmaskedIds, tt)';
        embGrad = lstm_grad.input(1:params.lstmSize, unmaskedIds);
        numWords = length(embIndices);
        if (tt<srcMaxLen)
          allEmbIndices_src(wordCount_src+1:wordCount_src+numWords) = embIndices;
          allEmbGrads_src(:, wordCount_src+1:wordCount_src+numWords) = embGrad;
          wordCount_src = wordCount_src + numWords;
        else
          % update the decoder
          allEmbIndices_tgt(wordCount_tgt+1:wordCount_tgt+numWords) = embIndices;
          allEmbGrads_tgt(:, wordCount_tgt+1:wordCount_tgt+numWords) = embGrad;
          wordCount_tgt = wordCount_tgt + numWords;
        end
      else % pass down hidden state grad to the below layer
        dh{ll-1}(:, unmaskedIds) = dh{ll-1}(:, unmaskedIds) + lstm_grad.input(1:params.lstmSize, unmaskedIds);
      end
    end % end for layer
  end % end for time
  
  % grad W_emb
  if params.isBi
    allEmbGrads_src(:, wordCount_src+1:end) = [];
    allEmbIndices_src(wordCount_src+1:end) = [];
    [grad.W_emb_src, grad.indices_src] = aggregateMatrix(allEmbGrads_src, allEmbIndices_src, params.isGPU, params.dataType);
  end

  % update the decoder
  allEmbGrads_tgt(:, wordCount_tgt+1:end) = [];
  allEmbIndices_tgt(wordCount_tgt+1:end) = [];
  [grad.W_emb_tgt, grad.indices_tgt] = aggregateMatrix(allEmbGrads_tgt, allEmbIndices_tgt, params.isGPU, params.dataType);
    
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
    params.numSrcHidVecs = params.srcMaxLen-1;
    assert(params.numSrcHidVecs<params.T);
    
    if params.attnGlobal % global
      params.numAttnPositions = params.numSrcHidVecs;
    else % local
      params.numAttnPositions = 2*params.posWin + 1;
    end
    
    % we extract trainData.srcHidVecs later, which contains all src hidden states, lstmSize * curBatchSize * numSrcHidVecs 
    grad.srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
  else
    params.numSrcHidVecs = 0;
  end
end
