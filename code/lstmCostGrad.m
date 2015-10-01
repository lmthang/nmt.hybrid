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
  if params.isBi
    srcMaxLen = trainData.srcMaxLen;
    T = srcMaxLen+tgtMaxLen-1;
    trainData.numInputWords_src = sum(sum(trainData.srcMask(:, 1:end-1)));
  else
    srcMaxLen = 1;
    T = tgtMaxLen;
  end
  input = trainData.input;
  inputMask = trainData.inputMask;
  curBatchSize = size(input, 1);
  trainData.numInputWords_tgt = sum(sum(trainData.tgtMask));
  
  trainData.isTest = isTest;
  trainData.T = T;
  trainData.srcMaxLen = srcMaxLen;
  trainData.curBatchSize = curBatchSize;
  
  params.curBatchSize = curBatchSize;
  params.srcMaxLen = srcMaxLen;
  params.T = T;
  [grad, params] = initGrad(model, params);
  zeroState = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  
  % current h_t and c_t over time
  h_t = cell(params.numLayers, 1);
  all_c_t = cell(params.numLayers, T);
  lstms = cell(params.numLayers, T); % each cell contains intermediate results for that timestep needed for backprop
  for ll=1:params.numLayers
    h_t{ll} = zeroMatrix([params.lstmSize, curBatchSize], params.isGPU, params.dataType);
    for tt=1:T
      all_c_t{ll, tt} = zeroMatrix([params.lstmSize, curBatchSize], params.isGPU, params.dataType);
    end
  end
  
  % init costs
  costs = initCosts(params);
  
  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  trainData.maskInfo = cell(T, 1);
  grad_softmax_all = cell(tgtMaxLen, 1);
  h2sInfoAll = cell(tgtMaxLen, 1);
  softmax_h = zeroState;
  
  % attentional model
  if params.attnFunc   
    if params.attnGlobal % global
      if params.attnOpt==0 % no src compare
        startAttnId = 1;
        endAttnId = params.numSrcHidVecs;
        startHidId = params.numAttnPositions-params.numSrcHidVecs+1;
        endHidId = params.numAttnPositions;
      end
      trainData.srcMaskedIds = [];
    end
  end
  if params.attnFunc>0
    trainData.srcHidVecsOrig = zeroMatrix([params.lstmSize, curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
  end
  
  % Note: IMPORTANT. For attention-based models or positional models, it it
  % important to build the top hidden states first before moving to the
  % next time step. So DO NOT swap these for loops.
  if params.isBi
    W_layers = model.W_src;
    W_emb = model.W_emb_src;
  end
  for tt=1:T % time
    tgtPos = tt-srcMaxLen+1;
    
    % switch to decoder mode
    if tt==srcMaxLen 
      W_layers = model.W_tgt;
      W_emb = model.W_emb_tgt;
    end
    
    for ll=1:params.numLayers % layer
      W = W_layers{ll};
      
      %% Input
      % previous-time input
      if tt==1 % first time step
        h_t_1 = zeroState;
        c_t_1 = zeroState;
      else
        h_t_1 = h_t{ll};
        c_t_1 = all_c_t{ll, tt-1};
      end

      % current-time input
      if ll==1 % first layer
        % prepare mask
        trainData.maskInfo{tt}.mask = inputMask(:, tt)'; % curBatchSize * 1
        trainData.maskInfo{tt}.unmaskedIds = find(trainData.maskInfo{tt}.mask);
        trainData.maskInfo{tt}.maskedIds = find(~trainData.maskInfo{tt}.mask);
        curMask = trainData.maskInfo{tt};
        
        if tt>=srcMaxLen % decoder input
          x_t = getLstmDecoderInput(input(:, tt)', W_emb, softmax_h, params);
        else
          x_t = W_emb(:, input(:, tt));
        end
        
        
      else % subsequent layer, use the previous-layer hidden state
        x_t = h_t{ll-1};
      end
      
      % masking
      x_t(:, curMask.maskedIds) = 0; 
      h_t_1(:, curMask.maskedIds) = 0;
      c_t_1(:, curMask.maskedIds) = 0;
      
      %% Core LSTM: input -> h_t
      [lstms{ll, tt}, h_t{ll}, all_c_t{ll, tt}] = lstmUnit(W, x_t, h_t_1, c_t_1, ll, tt, srcMaxLen, params, isTest); 
      % assert
      if params.assert
        assert(computeSum(h_t{ll}(:, curMask.maskedIds), params.isGPU)==0);
      end
      
      %% Loss
      if tt>=srcMaxLen && ll==params.numLayers % decoding phase, tgtPos>=1
        %% predicting positions
        trainData.posMask = curMask;
        
        %% predicting words
        % h_t -> softmax_h
        if params.attnFunc
          % TODO: save memory here, h2sInfo.input only keeps track of srcHidVecs or attnVecs, but not h_t.
          [softmax_h, h2sInfoAll{tgtPos}] = attnLayerForward(h_t{ll}, params, model, trainData, tgtPos);
        else
          softmax_h = h_t{ll};
        end
        
        % softmax_h -> loss
        predWords = trainData.tgtOutput(:, tgtPos)';
        [cost, probs, scores, scoreIndices] = softmaxLayerForward(model.W_soft, softmax_h, predWords, curMask);
        costs.total = costs.total + cost;
        costs.word = costs.word + cost;
                
        % backprop: loss -> softmax_h
        if isTest==0
          % loss -> softmax_h
          [grad_W_soft, grad_softmax_all{tgtPos}] = softmaxLayerBackprop(model.W_soft, softmax_h, probs, scoreIndices);
          grad.W_soft = grad.W_soft + grad_W_soft;
        end
      end
      
    
      %% Record src hidden states
      if tt<=params.numSrcHidVecs && ll==params.numLayers && params.attnFunc>0
        trainData.srcHidVecsOrig(:, :, tt) = h_t{ll};
        
        if tt==params.numSrcHidVecs
          [trainData] = updateDataSrcVecs(trainData, params);
        end
      end
      
      
      % assert
      if params.assert
        assert(computeSum(h_t{ll}(:, curMask.maskedIds), params.isGPU)==0);
        assert(computeSum(all_c_t{ll, tt}(:, curMask.maskedIds), params.isGPU)==0);
        
        if tt>=srcMaxLen && ll==params.numLayers
          assert(computeSum(scores(:, curMask.maskedIds), params.isGPU)==0);
          
          if isTest==0
            assert(computeSum(grad_softmax_all{tgtPos}(:, curMask.maskedIds), params.isGPU)==0);
          end
        end
      end
    end % end for t
  end
  
  if isTest==1 % don't compute grad
    return;
  end
  trainData = rmfield(trainData, 'posMask');
  
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
    allEmbGrads_src = zeroMatrix([params.lstmSize, trainData.numInputWords_src], params.isGPU, params.dataType);
    allEmbIndices_src = zeros(trainData.numInputWords_src, 1);
    wordCount_src = 0;
  end
  % update the decoder
  allEmbGrads_tgt = zeroMatrix([params.lstmSize, trainData.numInputWords_tgt], params.isGPU, params.dataType);
  allEmbIndices_tgt = zeros(trainData.numInputWords_tgt, 1);
  wordCount_tgt = 0;
  
  % NOTE: IMPORTANT for tt first, then for ll in other for attn3,4, pos2 models to work
  for tt=T:-1:1 % time
    curMask = trainData.maskInfo{tt};
    unmaskedIds = curMask.unmaskedIds;
    maskedIds = curMask.maskedIds;
    tgtPos = tt-srcMaxLen+1;
    
    % switch to encoder. NOTE: we assume W_layers has been set to 
    if tt==(srcMaxLen-1)
      W_layers = model.W_src;
    end
    
    %% softmax_h -> h_t: at the top layer
    if (tt>=srcMaxLen)
      if params.attnFunc
        % softmax_h -> h_t
        h2sInfo = h2sInfoAll{tgtPos};
        [grad_tgt_ht, attnGrad, grad_srcHidVecs] = attnLayerBackprop(model, grad_softmax_all{tgtPos}, trainData, h2sInfo, params, curMask);
        if params.assert
          assert(computeSum(grad_tgt_ht(:, curMask.maskedIds), params.isGPU)==0);
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
            if params.attnOpt==0 % fixed
              grad.srcHidVecs(:, :, startAttnId:endAttnId) = grad.srcHidVecs(:, :, startAttnId:endAttnId) + grad_srcHidVecs(:, :, startHidId:endHidId);
            else % variable
              grad.srcHidVecs = grad.srcHidVecs + grad_srcHidVecs;
            end
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
        c_t_1 = all_c_t{ll, tt-1};
      end
      c_t = all_c_t{ll, tt};
      lstm = lstms{ll, tt};
      
      [lstm_grad] = lstmUnitGrad(W, lstm, c_t, c_t_1, dc{ll}, dh{ll}, ll, tt, srcMaxLen, zeroState, maskedIds, params);
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
        %% those models that feed additional info into lstm input
        % same-length decoder
        if tt>=srcMaxLen
          if tgtPos<=params.numSrcHidVecs
            grad.srcHidVecs(:, unmaskedIds, tgtPos) = grad.srcHidVecs(:, unmaskedIds, tgtPos) + lstm_grad.input(params.lstmSize+1:2*params.lstmSize, unmaskedIds);
          end
        end

        % feed softmax vector
        if params.softmaxFeedInput && tt>srcMaxLen % for tt==srcMaxLen, we feed zero vector
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
    
    if params.attnGlobal
      if params.attnOpt==0 % for attnOpt==1, we use variable-length alignment vectors
        params.numAttnPositions = params.maxSentLen-1;
      else % global, content-based alignments
        params.numAttnPositions = params.numSrcHidVecs;
      end
    else % local
      params.numAttnPositions = 2*params.posWin + 1;
    end
    
    % we extract trainData.srcHidVecs later, which contains all src hidden states, lstmSize * curBatchSize * numSrcHidVecs 
    grad.srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
  else
    params.numSrcHidVecs = 0;
  end
end
