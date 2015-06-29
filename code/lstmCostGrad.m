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
  allInputInfo = cell(tgtMaxLen, 1);
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
        
%         trainData.alignMask = zeroMatrix([params.numAttnPositions, params.curBatchSize], params.isGPU, params.dataType);
%         trainData.alignMask(startHidId:endHidId, :) = trainData.srcMask(:, 1:params.numSrcHidVecs)';
        
%         trainData.alignMask = oneMatrix([params.numAttnPositions, params.curBatchSize], params.isGPU, params.dataType);
      else
%         trainData.alignMask = trainData.srcMask(:, 1:params.numSrcHidVecs)'; % numSrcHidVecs * curBatchSize
        
%         trainData.alignMask = oneMatrix([params.numSrcHidVecs, params.curBatchSize], params.isGPU, params.dataType);
      end
      trainData.srcMaskedIds = [];
%       trainData.srcMaskedIds = find(trainData.alignMask==0);
    else % local
      if params.posSignal % unsupervised alignment
        grad_ht_pos_all = cell(tgtMaxLen, 1);
      end
    end
  end
  if (params.attnFunc>0 || params.sameLength==1)
    trainData.srcHidVecsOrig = zeroMatrix([params.lstmSize, curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
  end
  
  % Note: IMPORTANT. For attention-based models or positional models, it it
  % important to build the top hidden states first before moving to the
  % next time step. So DO NOT swap these for loops.
  if params.isBi
    W_layers = model.W_src;
    if params.tieEmb
      W_emb = model.W_emb_tie;
    else
      W_emb = model.W_emb_src;
    end
  end
  for tt=1:T % time
    tgtPos = tt-srcMaxLen+1;
    
    % switch to decoder mode
    if tt==srcMaxLen 
      W_layers = model.W_tgt;
      if params.tieEmb==0 % separate embeddings
        W_emb = model.W_emb_tgt;
      end
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
          [x_t, allInputInfo{tgtPos}] = getLstmDecoderInput(input(:, tt)', tgtPos, W_emb, softmax_h, trainData, zeroState, params); %, curMask);
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
        if params.posSignal
          [cost_pos, posGrad, trainData] = posSignalCostGrad(model, h_t{ll}, tgtPos, curMask, trainData, params, isTest);
          costs.total = costs.total + cost_pos;
          costs.pos = costs.pos + cost_pos;
  
          if isTest==0
            grad_ht_pos_all{tgtPos} = posGrad.ht;
            grad.v_pos = grad.v_pos + posGrad.v_pos;
            grad.W_pos = grad.W_pos + posGrad.W_pos;
          end 
        else
          trainData.posMask = curMask;
        end
        
        
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
      if tt<=params.numSrcHidVecs && ll==params.numLayers && (params.attnFunc>0 || params.sameLength==1)
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
  if params.tieEmb % tie embeddings
    trainData.numInputWords_tie = trainData.numInputWords_tgt + trainData.numInputWords_src;
    allEmbGrads_tie = zeroMatrix([params.lstmSize, trainData.numInputWords_tie], params.isGPU, params.dataType);
    allEmbIndices_tie = zeros(trainData.numInputWords_tie, 1);
    wordCount_tie = 0;
  else
    if params.isBi
      allEmbGrads_src = zeroMatrix([params.lstmSize, trainData.numInputWords_src], params.isGPU, params.dataType);
      allEmbIndices_src = zeros(trainData.numInputWords_src, 1);
      wordCount_src = 0;
    end
    % update the decoder
    if params.epoch>=params.decodeUpdateEpoch
      allEmbGrads_tgt = zeroMatrix([params.lstmSize, trainData.numInputWords_tgt], params.isGPU, params.dataType);
      allEmbIndices_tgt = zeros(trainData.numInputWords_tgt, 1);
      wordCount_tgt = 0;
    end
  end
  
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
        % get signals from the softmax layer for positions
        if params.posSignal
          dh{params.numLayers} = dh{params.numLayers} + grad_ht_pos_all{tgtPos};
        end
        
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
        % update the decoder
        if params.epoch>=params.decodeUpdateEpoch || params.decodeUpdateOpt==1 % for decodeUpdateOpt==1, always update
          grad.W_tgt{ll} = grad.W_tgt{ll} + lstm_grad.W;
        end
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
        if params.sameLength==1 && tt>=srcMaxLen
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
        if params.tieEmb % tie embeddings
          allEmbIndices_tie(wordCount_tie+1:wordCount_tie+numWords) = embIndices;
          allEmbGrads_tie(:, wordCount_tie+1:wordCount_tie+numWords) = embGrad;
          wordCount_tie = wordCount_tie + numWords;
        else % separate embeddings
          if (tt<srcMaxLen)
            allEmbIndices_src(wordCount_src+1:wordCount_src+numWords) = embIndices;
            allEmbGrads_src(:, wordCount_src+1:wordCount_src+numWords) = embGrad;
            wordCount_src = wordCount_src + numWords;
          else
            % update the decoder
            if params.epoch>=params.decodeUpdateEpoch
              allEmbIndices_tgt(wordCount_tgt+1:wordCount_tgt+numWords) = embIndices;
              allEmbGrads_tgt(:, wordCount_tgt+1:wordCount_tgt+numWords) = embGrad;
              wordCount_tgt = wordCount_tgt + numWords;
            end
          end
        end
      else % pass down hidden state grad to the below layer
        dh{ll-1}(:, unmaskedIds) = dh{ll-1}(:, unmaskedIds) + lstm_grad.input(1:params.lstmSize, unmaskedIds);
      end
    end % end for layer
  end % end for time
  
  % grad W_emb
  if params.tieEmb % tie embeddings
    allEmbGrads_tie(:, wordCount_tie+1:end) = [];
    allEmbIndices_tie(wordCount_tie+1:end) = [];
    [grad.W_emb_tie, grad.indices_tie] = aggregateMatrix(allEmbGrads_tie, allEmbIndices_tie, params.isGPU, params.dataType);
  else
    if params.isBi
      allEmbGrads_src(:, wordCount_src+1:end) = [];
      allEmbIndices_src(wordCount_src+1:end) = [];
      [grad.W_emb_src, grad.indices_src] = aggregateMatrix(allEmbGrads_src, allEmbIndices_src, params.isGPU, params.dataType);
    end

    % update the decoder
    if params.epoch>=params.decodeUpdateEpoch
      allEmbGrads_tgt(:, wordCount_tgt+1:end) = [];
      allEmbIndices_tgt(wordCount_tgt+1:end) = [];
      [grad.W_emb_tgt, grad.indices_tgt] = aggregateMatrix(allEmbGrads_tgt, allEmbIndices_tgt, params.isGPU, params.dataType);
    end
  end
    
  % remove unused variables
  if params.attnFunc>0 %|| params.posModel>=2
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
  if params.attnFunc>0 || params.sameLength==1
    params.numSrcHidVecs = params.srcMaxLen-1;
    assert(params.numSrcHidVecs<params.T);
    
    % we extract trainData.srcHidVecs later, which contains all src hidden states, lstmSize * curBatchSize * numSrcHidVecs 
    grad.srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
  else
    params.numSrcHidVecs = 0;
  end
end

%           %% position predictions
%           if params.predictPos==1 % regression
%             
%           elseif params.predictPos==2 % classification
%             % h_t -> h_pos=f(W_pos*h_t)
%             h_pos = hiddenLayerForward(model.W_pos, h_t{ll}, params.nonlinear_f);
%           
%             % h_t_pos -> position loss
%             softmaxPositions = curPosOutput-params.startPosId+1; %+params.maxRelDist+1; % curPosOutput is in [-params.maxRelDist, params.maxRelDist]. the value params.maxRelDist+1 in curPosOutput marks eos.
%             [cost_pos, probs_pos, ~, scoreIndices_pos] = softmaxLayerForward(model.W_softPos, h_pos, softmaxPositions, trainData.posMask);
%             costs.total = costs.total + params.posWeight*cost_pos;
%             costs.pos = costs.pos + params.posWeight*cost_pos;
% 
%             % backprop: position loss -> h_pos
%             if isTest==0
%               % loss -> h_pos
%               [grad_W_soft, grad_h_pos] = softmaxLayerBackprop(model.W_softPos, h_pos, probs_pos, scoreIndices_pos);
%               grad_h_pos = params.posWeight*grad_h_pos;
%               grad_W_soft = params.posWeight*grad_W_soft;
%               
%               % h_pos -> h_t
%               [grad_ht_pos_all{tgtPos}, grad_W_pos] = hiddenLayerBackprop(model.W_pos, grad_h_pos, h_t{ll}, params.nonlinear_f_prime, h_pos);
% 
%               % update grads
%               grad.W_softPos = grad.W_softPos + grad_W_soft;
%               
%               % assert
%               if params.assert
%                 assert(sum(sum(abs(grad_h_pos(:, trainData.posMask.maskedIds))))==0);
%               end
%             end
%             
%             %% TODO: do argmax positions here when isTest==1
%             trainData.positions = (tgtPos+params.zeroPosId) - curPosOutput; %  = tgtPos - (curPosOutput-zeroPosId)  = srcPos = tgtPos - relative distance
%           end          


%           %% null predictions
%           if params.predictNull
%             nullMask.mask = curMask.mask & curPosOutput==params.nullPosId;
%             nullMask.unmaskedIds = find(nullMask.mask);
%             
%             if ~isempty(nullMask.unmaskedIds)
%               nullMask.maskedIds = find(~nullMask.mask);
% 
%               % h_t -> h_null=f(W_null*h_t)
%               h_null = hiddenLayerForward(model.W_null, h_t{ll}, params.nonlinear_f);
% 
%               % h_null -> softmax
%               nullLabels = nullMask.mask+1; % 1: non-null, 2: null
%               [cost_null, probs_null, ~, scoreIndices_null] = softmaxLayerForward(model.W_softNull, h_null, nullLabels, nullMask);
%               costs.total = costs.total + params.nullWeight*cost_null;
%               costs.null = costs.null + params.nullWeight*cost_null;
% 
%               if isTest==0 % backprop
%                 % loss -> h_null
%                 [grad_W_softNull, grad_h_null] = softmaxLayerBackprop(model.W_softNull, h_null, probs_null, scoreIndices_null);
%                 grad_h_null = params.nullWeight*grad_h_null;
%                 grad_W_softNull = params.nullWeight*grad_W_softNull;
% 
%                 % h_null -> h_t
%                 [grad_ht, grad_W_null] = hiddenLayerBackprop(model.W_null, grad_h_null, h_t{ll}, params.nonlinear_f_prime, h_null);
% 
%                 % update grads
%                 grad.W_softNull = grad.W_softNull + grad_W_softNull;
%                 grad.W_null = grad.W_null + grad_W_null;
%                 grad_ht_pos_all{tgtPos} = grad_ht_pos_all{tgtPos} + grad_ht;
%                 
%                 % assert
%                 if params.assert
%                   assert(sum(sum(abs(grad_h_null(:, nullMask.maskedIds))))==0);
%                   assert(sum(sum(abs(grad_ht(:, nullMask.maskedIds))))==0);
%                 end
%               end
%             end % if ~isempty
%           end % end predictNull

  
%   %% costs
%   if params.isGPU    
%     costs.total = gather(costs.total);
%     costs.word = gather(costs.word);
%     if params.predictPos
%       costs.pos = gather(costs.pos);
%       if predictNull
%         costs.null = gather(costs.null);
%       end
%     end
%   end

%% Positional models %%
%             % round positions
%             if isTest % use predicted positions at test time
%               trainData.positions = floor(scales.*trainData.srcLens) + 1;
%             else % use gold positions, at training time
%             end


%           if params.predictPos % use unsupervised alignments  
%           else % soft attention
%             if params.attnGlobal==0 % relative
%               startAttnId = h2sInfo.startAttnId;
%               endAttnId = h2sInfo.endAttnId;
%               startHidId = h2sInfo.startHidId;
%               endHidId = h2sInfo.endHidId;
%             end
%             grad.srcHidVecs(:, :, startAttnId:endAttnId) = grad.srcHidVecs(:, :, startAttnId:endAttnId) + grad_srcHidVecs(:, :, startHidId:endHidId);
%           end

%           %% decide null or non-null alignment
%           nullFlags = curPosOutput==params.nullPosId;
%           nullLabels = nullFlags+1; % 1: non-null, 2: null
%           [cost_null, probs_null, scores_null, scoreIndices_null] = softmaxLayerForward(model.W_softNull, top_ht, nullLabels, curMask);
%           costs.total = costs.total + params.posWeight*cost_null;
%           costs.null = costs.null + params.posWeight*cost_null;
%           
%           % assert
%           if params.assert
%             assert(sum(sum(abs(scores_null(:, curMask.maskedIds))))==0);
%           end
%           
%           if isTest==0 % backprop
%             % loss -> h_t_pos
%             [grad_W_softNull, grad_ht] = softmaxLayerBackprop(model.W_softNull, top_ht, probs_null, scoreIndices_null);
%             
%             % assert
%             if params.assert
%               assert(sum(sum(abs(grad_ht(:, curMask.maskedIds))))==0);
%             end
%             
%             % update grads
%             grad_ht_pos_all{tgtPos} = params.posWeight*grad_ht;
%             if tt==srcMaxLen
%               grad.W_softNull = params.posWeight*grad_W_softNull;
%             else
%               grad.W_softNull = grad.W_softNull + params.posWeight*grad_W_softNull;
%             end
%           end
%           top_ht(:, nullFlags) = 0; % ignore <p_n>

%             if params.oldSrcVecs % old
%               grad.srcHidVecs(h2sInfo.linearIndices) = grad.srcHidVecs(h2sInfo.linearIndices) + grad_srcHidVecs(h2sInfo.attnLinearIndices);
%             else % new
%             end

%   % positional models
%   if params.posModel>0
%     if params.posModel==2
%       trainData.srcPosInfo = cell(tgtMaxLen, 1);
%       W_tgt_combined = [model.W_tgt{1} model.W_tgt_pos];
%     end
%     
%     % positional model 3, include more src embeddings
%     if params.posModel==3 
%       trainData.numInputWords_src = floor(trainData.numInputWords_src + trainData.numInputWords_tgt*0.5);
%     end
%   end

%           % pos model predict words
%           if params.posModel==2 && mod(tgtPos, 2)==0 
%             W = W_tgt_combined;
%           end

%         if params.posModel>=1
%           if mod(tgtPos, 2)==1 % positions
%             predWords = predWords - params.startPosId + 1;
%             matrixName = 'W_softPos';
%           else
%             matrixName = 'W_soft';
%           end
%         end

%         if params.posModel>=0 % separate out pos/word perplexities
%           if mod(tgtPos, 2)==0
%             costs.word = costs.word + cost;
%           else
%             costs.pos = costs.pos + cost;
%           end
%         end

%       if params.posModel>=1 && mod(tgtPos, 2)==1 % positions
%         isPredictPos = 1;
%       else % words
%         isPredictPos = 0;
%       end
      
%         % positional model 3, predict words
%         if params.posModel==3 && isPredictPos==0
%           grad.srcHidVecs(h2sInfo.linearIndices) = grad.srcHidVecs(h2sInfo.linearIndices) + reshape(grad_srcHidVecs(:, curMask.unmaskedIds), 1, []);
%         end

%           % positional
%           if params.posModel==2 && ll==1 && mod(tgtPos, 2)==0 % predict words
%             grad.W_tgt{ll} = grad.W_tgt{ll} + lstm_grad.W(:, 1:2*params.lstmSize);
%             grad.W_tgt_pos = grad.W_tgt_pos + lstm_grad.W(:, 2*params.lstmSize+1:end);
%           else
%           end

%         % pos model 2 lstm_grad.input = = [x_t; s_t; h_t]
%         if params.posModel==2 && tt>=srcMaxLen && mod(tgtPos, 2)==0 % predict words
%           inputInfo = allInputInfo{tgtPos};
%           linearIndices = inputInfo.srcPosLinearIndices;
%           grad.srcHidVecs(linearIndices) = grad.srcHidVecs(linearIndices) + reshape(lstm_grad.input(params.lstmSize+1:2*params.lstmSize, unmaskedIds), 1, []);          
%         end


%% softmax
%   %%%%%%%%%%%%%%%
%   %%% SOFTMAX %%%
%   %%%%%%%%%%%%%%%
%   [costs, softmaxGrad, grad_tgt_ht] = hid2lossCostGrad(model, params, trainData, softmaxVecAll, h2sInfoAll); %all_h_t(params.numLayers, :));
%   if isTest==1 % don't compute grad
%     return;
%   else
%     if isstruct(softmaxGrad)
%       fields = fieldnames(softmaxGrad);
%       for ii=1:length(fields)
%         field = fields{ii};
%         grad.(field) = softmaxGrad.(field);
%       end
%     end
%   end
    
%   % attention feed input
%   elseif params.attnFeedInput
%     if params.attnRelativePos % relative
%       [attnSrcHidVecs, decodeInfo.startAttnId, decodeInfo.endAttnId, decodeInfo.startHidId, decodeInfo.endHidId] = buildSrcHidVecs(trainData.srcHidVecs, trainData.srcMaxLen, tgtPos, params);
%     else % absolute
%       attnSrcHidVecs = absAttnSrcHidVecs;
%     end
% 
%     % attnForward: h_t -> attnVecs (used the top previous hidden state)
%     [attnVecs, decodeInfo.alignWeights] = attnLayerForward(model.W_a, all_h_t{params.numLayers, tt-1}, attnSrcHidVecs, trainData.maskInfo{tt}.mask);
%     x_t = [W_emb(:, decodeInput); attnVecs];

%         % attn feed input
%         if params.attnFeedInput && tt>=srcMaxLen
%           if params.attnRelativePos % relative
%             attnSrcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
%             startAttnId = decodeInfo.startAttnId;
%             endAttnId = decodeInfo.endAttnId;
%             startHidId = decodeInfo.startHidId;
%             endHidId = decodeInfo.endHidId;
%             attnSrcHidVecs(:, :, startHidId:endHidId) = trainData.srcHidVecs(:, :, startAttnId:endAttnId);
%           else % absolute
%             attnSrcHidVecs = trainData.absSrcHidVecs;
%           end
%           
%           % grad_attn -> grad_ht, grad_W_a, grad_srcHidVecs
%           % grad_attn_ht will be used in time tt-1, at the top layer
%           prev_ht = all_h_t{params.numLayers, tt-1};
%           prev_ht(:, maskedIds) = 0;
%           [grad_attn_ht, grad_W_a, grad_srcHidVecs] = attnLayerBackprop(model.W_a, lstm_grad.input(params.lstmSize+1:2*params.lstmSize, :), prev_ht, ...
%             params, decodeInfo.alignWeights, attnSrcHidVecs);
%           
%           % update srcHidVecs
%           grad.srcHidVecs(:, :, startAttnId:endAttnId) = grad.srcHidVecs(:, :, startAttnId:endAttnId) + grad_srcHidVecs(:, :, startHidId:endHidId);
%           
%           % update W_a
%           grad.W_a = grad.W_a + grad_W_a;
%           
%           % update the top hidden layer of the previous time step
%           dh{params.numLayers} = dh{params.numLayers} + grad_attn_ht;
%         end % if attnFeedInput

%% class-based softmax %%
% in main method, backprop
%   % grad W_soft_inclass
%   if params.numClasses>0
%     grad.classIndices = otherSoftmaxGrads.classIndices;
%     grad.W_soft_inclass = otherSoftmaxGrads.W_soft_inclass(:, :, grad.classIndices);
%   end

%% Unused    
%   if params.separateEmb==1  
%   else
%     trainData.numInputWords = sum(sum(inputMask));
%   end

%     % separate embs
%     if params.separateEmb==1
%     else
%       trainData.numInputWords = floor(trainData.numInputWords * 1.5);
%     end

%   % separate emb
%   if params.separateEmb==1 
%   else
%     W_emb = model.W_emb;
%   end

%       % separate emb
%       if params.separateEmb==1   
%       end

%   % separate embs
%   if params.separateEmb==1
%   else
%     allEmbGrads = zeroMatrix([params.lstmSize, trainData.numInputWords], params.isGPU, params.dataType);
%     allEmbIndices = zeros(trainData.numInputWords, 1);
%     wordCount = 0;
%   end

%         % separate embs
%         if params.separateEmb==1
%         else
%           allEmbIndices(wordCount+1:wordCount+numWords) = embIndices;
%           allEmbGrads(:, wordCount+1:wordCount+numWords) = embGrad;
%           wordCount = wordCount + numWords;
%         end

%   if params.separateEmb==1 % % separate embs
%   else
%     allEmbGrads(:, wordCount+1:end) = [];
%     allEmbIndices(wordCount+1:end) = [];
%     [grad.W_emb, grad.indices] = aggregateMatrix(allEmbGrads, allEmbIndices, params.isGPU, params.dataType);
%   end
