function [costs, grad] = lstmCostGrad(model, trainData, params, isTest)
%%%
%
% Compute cost/grad for LSTM. 
% When params.posModel>0, returns costs.pos and costs.word
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
  
  params.curBatchSize = curBatchSize;
  params.srcMaxLen = srcMaxLen;
  params.T = T;
  [grad, params] = initGrad(model, params);
  zeroState = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  
  % all h_t, c_t over time
  all_h_t = cell(params.numLayers, T);
  all_c_t = cell(params.numLayers, T);
  lstms = cell(params.numLayers, T); % each cell contains intermediate results for that timestep needed for backprop
  for ll=1:params.numLayers
    for tt=1:T
      all_h_t{ll, tt} = zeroMatrix([params.lstmSize, curBatchSize], params.isGPU, params.dataType);
      all_c_t{ll, tt} = zeroMatrix([params.lstmSize, curBatchSize], params.isGPU, params.dataType);
    end
  end
  
  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  trainData.maskInfo = cell(T, 1);
  allDecodeInfo = cell(tgtMaxLen, 1);
  softmaxVecAll = cell(tgtMaxLen, 1);
  h2sInfoAll = cell(tgtMaxLen, 1);
  % positional models
  if params.posModel>0
    if params.posModel==2
      trainData.srcPosInfo = cell(tgtMaxLen, 1);
      W_tgt_combined = [model.W_tgt{1} model.W_tgt_pos];
    end
    
    % positional model 3, include more src embeddings
    if params.posModel==3 
      trainData.numInputWords_src = floor(trainData.numInputWords_src + trainData.numInputWords_tgt*0.5);
    end
  end
  
  % attentional model
%   if params.attnFeedInput
%     if params.attnRelativePos
%       trainData.startAttnIds = zeros(tgtMaxLen, 1);
%       trainData.endAttnIds = zeros(tgtMaxLen, 1);
%       trainData.startHidIds = zeros(tgtMaxLen, 1);
%       trainData.endHidIds = zeros(tgtMaxLen, 1); 
%     else % absolute pos
%       startAttnId = 1;
%       endAttnId = params.numSrcHidVecs;
%       startHidId = params.numAttnPositions-params.numSrcHidVecs+1;
%       endHidId = params.numAttnPositions;
%     end
%   end
  absAttnSrcHidVecs = [];
  
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
      
      % previous-time input
      if tt==1 % first time step
        h_t_1 = zeroState;
        c_t_1 = zeroState;
      else
        c_t_1 = all_c_t{ll, tt-1};
        h_t_1 = all_h_t{ll, tt-1};
      end

      % current-time input
      if ll==1 % first layer
        % prepare mask
        trainData.maskInfo{tt}.mask = inputMask(:, tt)'; % curBatchSize * 1
        trainData.maskInfo{tt}.unmaskedIds = find(trainData.maskInfo{tt}.mask);
        trainData.maskInfo{tt}.maskedIds = find(~trainData.maskInfo{tt}.mask);
        
        if tt>=srcMaxLen % decoder input
          [x_t, allDecodeInfo{tgtPos}] = getDecoderInput(input(:, tt), tt, tgtPos, model, W_emb, all_h_t, trainData, absAttnSrcHidVecs, zeroState, params);
          
          % pos model predict words
          if params.posModel==2 && mod(tgtPos, 2)==0 
            W = W_tgt_combined;
          end
        else
          x_t = W_emb(:, input(:, tt));
        end
      else % subsequent layer, use the previous-layer hidden state
        x_t = all_h_t{ll-1, tt}; % lstm{ll-1, t}.h_t;
      end
      
      % masking
      x_t(:, trainData.maskInfo{tt}.maskedIds) = 0; 
      h_t_1(:, trainData.maskInfo{tt}.maskedIds) = 0;
      c_t_1(:, trainData.maskInfo{tt}.maskedIds) = 0;
      
      %% Core LSTM
      [lstms{ll, tt}, all_h_t{ll, tt}, all_c_t{ll, tt}] = lstmUnit(W, x_t, h_t_1, c_t_1, ll, tt, srcMaxLen, params, isTest); 
      
      % softmax_h
%       if tt>=srcMaxLen && ll==params.numLayers % decoding phase, tgtPos>=1
%         [softmaxVecAll{tgtPos}, h2sInfoAll{tgtPos}] = hid2softLayerForward(all_h_t{ll, tt}, params, model, trainData, trainData.maskInfo{tt}, tgtPos);
%       end
      
      % attention-based or positional-based models or same-length decoder: keep track of all the src hidden states
      if tt==params.numSrcHidVecs && ll==params.numLayers && (params.attnFunc>0 || params.posModel>=2 || params.sameLength==1)
        trainData.srcHidVecs = reshape([all_h_t{params.numLayers, 1:params.numSrcHidVecs}], params.lstmSize, curBatchSize, params.numSrcHidVecs);
        
%         if params.attnFeedInput && params.attnRelativePos==0 % absolute positions
%           absAttnSrcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
%           absAttnSrcHidVecs(:, :, startHidId:endHidId) = reshape([all_h_t{params.numLayers, 1:params.numSrcHidVecs}], params.lstmSize, curBatchSize, params.numSrcHidVecs);
%         end
      end
      
      % assert
      if params.assert
        assert(sum(sum(abs(all_h_t{ll, tt}(:, trainData.maskInfo{tt}.maskedIds))))==0);
        assert(sum(sum(abs(all_c_t{ll, tt}(:, trainData.maskInfo{tt}.maskedIds))))==0);
      end
    end % end for t
  end
  
  
  %%%%%%%%%%%%%%%
  %%% SOFTMAX %%%
  %%%%%%%%%%%%%%%
  [costs, softmaxGrad, grad_tgt_ht] = hid2lossCostGrad(model, params, trainData, all_h_t(params.numLayers, :));
  if isTest==1 % don't compute grad
    return;
  else
    if isstruct(softmaxGrad)
      fields = fieldnames(softmaxGrad);
      for ii=1:length(fields)
        field = fields{ii};
        grad.(field) = softmaxGrad.(field);
      end
    end
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
    unmaskedIds = trainData.maskInfo{tt}.unmaskedIds;
    maskedIds = trainData.maskInfo{tt}.maskedIds;
    tgtPos = tt-srcMaxLen+1;
    if tt>=srcMaxLen
      decodeInfo = allDecodeInfo{tgtPos};
    end
    
    % switch to encoder. NOTE: we assume W_layers has been set to 
    if tt==(srcMaxLen-1)
      W_layers = model.W_src;
    end
    for ll=params.numLayers:-1:1 % layer
      W = W_layers{ll};
      
      %% hidden state grad
      if ll==params.numLayers
        if (tt>=srcMaxLen) % get signals from the softmax layer
          dh{ll} = dh{ll} + grad_tgt_ht{tgtPos};
        end
          
        if tt<=params.numSrcHidVecs % attention/pos models: get feedback from grad.srcHidVecs
          dh{ll} = dh{ll} + grad.srcHidVecs(:,:,tt);
        end
      end

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
        assert(sum(sum(abs(lstm_grad.input(:, maskedIds))))==0);
      end
      
      %% grad.W_src / grad.W_tgt
      if (tt>=srcMaxLen)
        % update the decoder
        if params.epoch>=params.decodeUpdateEpoch || params.decodeUpdateOpt==1 % for decodeUpdateOpt==1, always update
          % positional
          if params.posModel==2 && ll==1 && mod(tgtPos, 2)==0 % predict words
            grad.W_tgt{ll} = grad.W_tgt{ll} + lstm_grad.W(:, 1:2*params.lstmSize);
            grad.W_tgt_pos = grad.W_tgt_pos + lstm_grad.W(:, 2*params.lstmSize+1:end);
          else
            grad.W_tgt{ll} = grad.W_tgt{ll} + lstm_grad.W;
          end
        end
      else
        grad.W_src{ll} = grad.W_src{ll} + lstm_grad.W;
      end
      
      %% input grad: lstm_grad.input = [x_t; h_t] for normal models
      % NOTE: important, here we only do assignment to initialize the hidden grad at layer ll for the previous time step. 
      % Later, when we go back one time step, we will accumulate.
      dh{ll} = lstm_grad.input(end-params.lstmSize+1:end, :); 
      if ll==1 % collect embedding grad
        embIndices = input(unmaskedIds, tt)';
        embGrad = lstm_grad.input(1:params.lstmSize, unmaskedIds);
        
        % same-length decoder
        if params.sameLength==1 && tt>=srcMaxLen
          if tgtPos<=params.numSrcHidVecs
            grad.srcHidVecs(:, unmaskedIds, tgtPos) = grad.srcHidVecs(:, unmaskedIds, tgtPos) + lstm_grad.input(params.lstmSize+1:2*params.lstmSize, unmaskedIds);
          end
        end
        % pos model 2 lstm_grad.input = = [x_t; s_t; h_t]
        if params.posModel==2 && tt>=srcMaxLen && mod(tgtPos, 2)==0 % predict words
          linearIndices = decodeInfo.srcPosLinearIndices;
          grad.srcHidVecs(linearIndices) = grad.srcHidVecs(linearIndices) + reshape(lstm_grad.input(params.lstmSize+1:2*params.lstmSize, unmaskedIds), 1, []);          
        end

        % attn feed input
%         if params.attnFeedInput && tt>=srcMaxLen
%           if params.attnRelativePos % relative
%             attnSrcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
%             startAttnId = decodeInfo.startAttnId;
%             endAttnId = decodeInfo.endAttnId;
%             startHidId = decodeInfo.startHidId;
%             endHidId = decodeInfo.endHidId;
%             attnSrcHidVecs(:, :, startHidId:endHidId) = trainData.srcHidVecs(:, :, startAttnId:endAttnId);
%           else % absolute
%             attnSrcHidVecs = absAttnSrcHidVecs;
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
        
        numWords = length(embIndices);
        
        % emb grad
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
  if params.attnFunc>0 || params.posModel>=2
    grad = rmfield(grad, 'srcHidVecs');
  end
  
  % gather data from GPU
  if params.isGPU    
    % costs
    costs.total = gather(costs.total);
    if params.posModel>0
      costs.pos = gather(costs.pos);
      costs.word = gather(costs.word);
    end
  end
end

function [x_t, decodeInfo] = getDecoderInput(decodeInput, tt, tgtPos, model, W_emb, all_h_t, trainData, absAttnSrcHidVecs, zeroState, params)
  decodeInfo = [];
  
  
  % same-length decoder
  if params.sameLength==1
    if tgtPos>params.numSrcHidVecs
      x_t = [W_emb(:, decodeInput); zeroState];
    else
      x_t = [W_emb(:, decodeInput); trainData.srcHidVecs(:, :, params.numSrcHidVecs-tgtPos+1)];
    end
    
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
    
  % positionl models 2: at the first level, we use additional src information
  elseif params.posModel==2 && mod(tgtPos, 2)==0 % predict words
    [s_t, decodeInfo.srcPosLinearIndices] = buildSrcPosVecs(tgtPos, params, trainData, trainData.tgtOutput(:, tgtPos)', trainData.maskInfo{tt});
    x_t = [W_emb(:, decodeInput); s_t];
  else
    x_t = W_emb(:, decodeInput);
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
  if params.attnFunc>0 || params.posModel>=2 || params.sameLength==1
    params.numSrcHidVecs = params.srcMaxLen-1;
    assert(params.numSrcHidVecs<params.T);
    
    % we extract trainData.srcHidVecs later, which contains all src hidden states, lstmSize * curBatchSize * numSrcHidVecs 
    grad.srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
  else
    params.numSrcHidVecs = 0;
  end
end


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
