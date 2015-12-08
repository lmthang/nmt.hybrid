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
  curBatchSize = size(trainData.tgtInput, 1);
  if params.isBi
    srcMaxLen = trainData.srcMaxLen;
    trainData.srcTotalWordCount = sum(trainData.srcLens);
  else % monolingual
    srcMaxLen = 1;
  end
  T = srcMaxLen+tgtMaxLen-1;
  
  %input = trainData.input;
  %inputMask = trainData.inputMask;
  trainData.tgtTotalWordCount = sum(trainData.tgtLens);
  
  trainData.isTest = isTest;
  trainData.T = T;
  trainData.srcMaxLen = srcMaxLen;
  trainData.curBatchSize = curBatchSize;
  
  params.curBatchSize = curBatchSize;
  params.srcMaxLen = srcMaxLen;
  params.T = T;
  [grad, params] = initGrad(model, params);
  zeroBatch = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  
  % initState
  zeroState = cell(params.numLayers, 1);
  for ll=1:params.numLayers % layer
    zeroState{ll}.h_t = zeroBatch;
    zeroState{ll}.c_t = zeroBatch;
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
  
%   % prepare mask
%   maskInfos = cell(T, 1);
%   for tt=1:T
%     maskInfos{tt}.mask = inputMask(:, tt)'; % curBatchSize * 1
%     maskInfos{tt}.unmaskedIds = find(maskInfos{tt}.mask);
%     maskInfos{tt}.maskedIds = find(~maskInfos{tt}.mask);
%   end
  
  %% encoder
  lastEncState = zeroState;
  encLen = srcMaxLen - 1;
  
  if params.isBi
    isDecoder = 0;
    [srcMaskInfos] = prepareMask(trainData.srcMask);
    [encStates, ~] = rnnLayerForward(encLen, model.W_src, model.W_emb_src, zeroState, trainData.srcInput, srcMaskInfos, ...
      params, isTest, isDecoder, trainData, model);
    lastEncState = encStates{end};
    
    % attention
    if params.attnFunc>0
      % Record src hidden states
      for tt=1:params.numSrcHidVecs
        trainData.srcHidVecsOrig(:, :, tt) = encStates{tt}{params.numLayers}.h_t;
      end
      
      if params.attnGlobal == 0 % local
        trainData.srcHidVecs = trainData.srcHidVecsOrig;
      end
    end
  end
  
  %% decoder
  decLen = T - srcMaxLen + 1;
  isDecoder = 1;
  [tgtMaskInfos] = prepareMask(trainData.tgtMask);
  [decStates, attnInfos] = rnnLayerForward(decLen, model.W_tgt, model.W_emb_tgt, lastEncState, trainData.tgtInput, tgtMaskInfos, ...
    params, isTest, isDecoder, trainData, model);
  
  %% softmax
  [costs.total, grad.W_soft, dec_top_grads] = softmaxCostGrad(decLen, decStates, attnInfos, model.W_soft, trainData.tgtOutput, tgtMaskInfos, params, isTest);
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
    dh{ll} = zeroBatch;
    dc{ll} = zeroBatch;
  end
  
  % decoder
  isDecoder = 1;
  [dc, dh, grad.W_tgt, grad.W_emb_tgt, grad.indices_tgt, attnGrad, grad.srcHidVecs] = rnnLayerBackprop(decLen, model.W_tgt, ...
    decStates, lastEncState, ...
    dec_top_grads, dc, dh, trainData.tgtInput, tgtMaskInfos, params, isDecoder, attnInfos, trainData, model);
  if params.attnFunc % copy attention grads 
    [grad] = copyStruct(attnGrad, grad);
  end
    
  % encoder
  if params.isBi
    enc_top_grads = cell(encLen, 1);
    for tt=1:params.numSrcHidVecs % attention
      enc_top_grads{tt} = grad.srcHidVecs(:,:,tt);
    end
    
    isDecoder = 0;
    [~, ~, grad.W_src, grad.W_emb_src, grad.indices_src, ~, ~] = rnnLayerBackprop(encLen, model.W_src, encStates, zeroState, ...
    enc_top_grads, dc, dh, trainData.srcInput, srcMaskInfos, params, isDecoder, attnInfos, trainData, model);
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


%   % h_t and c_t gradients accumulate over time per layer
%   % emb grad
%   if params.isBi
%     allEmbGrads_src = zeroMatrix([params.lstmSize, trainData.srcTotalWordCount], params.isGPU, params.dataType);
%     allEmbIndices_src = zeros(trainData.srcTotalWordCount, 1);
%     wordCount_src = 0;
%   end
%   % update the decoder
%   allEmbGrads_tgt = zeroMatrix([params.lstmSize, trainData.tgtTotalWordCount], params.isGPU, params.dataType);
%   allEmbIndices_tgt = zeros(trainData.tgtTotalWordCount, 1);
%   wordCount_tgt = 0;
%   
%   assert(params.numSrcHidVecs < srcMaxLen)
%   
%   % NOTE: IMPORTANT for tt first, then for ll in other for attn3,4, pos2 models to work
%   W_layers = model.W_tgt;
%   isDecoder = 1;
%   for tt=T:-1:1 % time
%     unmaskedIds = maskInfos{tt}.unmaskedIds;
%     tgtPos = tt-srcMaxLen+1;
%     
%     % switch to encoder. NOTE: we assume W_layers has been set to 
%     if tt==(srcMaxLen-1)
%       W_layers = model.W_src;
%       isDecoder = 0;
%     end
%     
%     %% softmax_h -> h_t: at the top layer
%     if (tt>=srcMaxLen)
%       if params.attnFunc
%         % softmax_h -> h_t
%         h2sInfo = attnInfos{tgtPos};
%         [grad_tgt_ht, attnGrad, grad_srcHidVecs] = attnLayerBackprop(model, grad_softmax_all{tgtPos}, trainData, h2sInfo, params, maskInfos{tt});
%       
%         fields = fieldnames(attnGrad);
%         for ii=1:length(fields)
%           field = fields{ii};
%           if tt==T
%             grad.(field) = attnGrad.(field);
%           else
%             grad.(field) = grad.(field) + attnGrad.(field);
%           end
%         end
% 
%         % attention models: srcHidVecs
%         if params.attnFunc
%           if params.attnGlobal 
%             grad.srcHidVecs = grad.srcHidVecs + grad_srcHidVecs;
%           else
%             grad.srcHidVecs = reshape(grad.srcHidVecs, params.lstmSize, []);
%             grad.srcHidVecs(:, h2sInfo.linearIdAll) = grad.srcHidVecs(:, h2sInfo.linearIdAll) + grad_srcHidVecs(:, h2sInfo.linearIdSub);
%             grad.srcHidVecs = reshape(grad.srcHidVecs, [params.lstmSize, params.curBatchSize, params.numSrcHidVecs]);
%           end
%         end
%       else
%         grad_tgt_ht = grad_softmax_all{tgtPos};
%       end
%     end
%     
%     %% grad from the top
%     if (tt>=srcMaxLen)
%       topHidGrad = grad_tgt_ht;
%     else
%       topHidGrad = zeroBatch;
%     end
%     % attention: get feedback from grad.srcHidVecs
%     if tt<=params.numSrcHidVecs 
%       topHidGrad = topHidGrad + grad.srcHidVecs(:,:,tt);
%     end
%     
%     if tt>1
%       prevState = lstmStates{tt-1};
%     else
%       prevState = zeroState;
%     end
%     
%     %% multi-layer RNN backprop
%     [dc, dh, d_emb, d_W_rnn, d_feed_input] = rnnStepLayerBackprop(W_layers, prevState, lstmStates{tt}, topHidGrad, dc, dh, maskInfos{tt}, params, isDecoder);
%     
%     %% grad
%     for ll=params.numLayers:-1:1 % layer
%       if (tt>=srcMaxLen)
%         grad.W_tgt{ll} = grad.W_tgt{ll} + d_W_rnn{ll};
%       else
%         grad.W_src{ll} = grad.W_src{ll} + d_W_rnn{ll};
%       end
%     end
%     
%     % feed softmax vector
%     if params.feedInput && tt>srcMaxLen % for tt==srcMaxLen, we feed zero vector
%       grad_softmax_all{tgtPos-1} = grad_softmax_all{tgtPos-1} + d_feed_input;
%     end
% 
%     % emb grad
%     embIndices = input(unmaskedIds, tt)';
%     embGrad = d_emb(1:params.lstmSize, unmaskedIds);
%     numWords = length(embIndices);
%     if (tt<srcMaxLen)
%       allEmbIndices_src(wordCount_src+1:wordCount_src+numWords) = embIndices;
%       allEmbGrads_src(:, wordCount_src+1:wordCount_src+numWords) = embGrad;
%       wordCount_src = wordCount_src + numWords;
%     else
%       % update the decoder
%       allEmbIndices_tgt(wordCount_tgt+1:wordCount_tgt+numWords) = embIndices;
%       allEmbGrads_tgt(:, wordCount_tgt+1:wordCount_tgt+numWords) = embGrad;
%       wordCount_tgt = wordCount_tgt + numWords;
%     end
%   end % end for time
%   
%   % grad W_emb
%   if params.isBi
%     allEmbGrads_src(:, wordCount_src+1:end) = [];
%     allEmbIndices_src(wordCount_src+1:end) = [];
%     [grad.W_emb_src, grad.indices_src] = aggregateMatrix(allEmbGrads_src, allEmbIndices_src, params.isGPU, params.dataType);
%   end
% 
%   % update the decoder
%   allEmbGrads_tgt(:, wordCount_tgt+1:end) = [];
%   allEmbIndices_tgt(wordCount_tgt+1:end) = [];
%   [grad.W_emb_tgt, grad.indices_tgt] = aggregateMatrix(allEmbGrads_tgt, allEmbIndices_tgt, params.isGPU, params.dataType);