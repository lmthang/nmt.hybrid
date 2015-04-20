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
  else
    srcMaxLen = 1;
    T = tgtMaxLen;
  end
  input = trainData.input;
  inputMask = trainData.inputMask;
  curBatchSize = size(input, 1);
  
  % separate embs
  if params.separateEmb==1
    trainData.numInputWords_src = sum(sum(trainData.srcMask));
    trainData.numInputWords_tgt = sum(sum(trainData.tgtMask(:, 2:end)));
  else
    trainData.numInputWords = sum(sum(inputMask));
  end
  
  trainData.isTest = isTest;
  trainData.T = T;
  
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
  
  % positional model 3, include more src embeddings
  if params.posModel==3 
    % separate embs
    if params.separateEmb==1
      trainData.numInputWords_src = floor(trainData.numInputWords_src + trainData.numInputWords_tgt*0.5);
    else
      trainData.numInputWords = floor(trainData.numInputWords * 1.5);
    end
  end
  
  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  trainData.maskInfo = cell(T, 1);
  if params.posModel==2
    trainData.srcPosInfo = cell(T, 1);
    W_tgt_combined = [model.W_tgt{1} model.W_tgt_pos];
  end
  
  if params.attnFunc==3 || params.attnFunc==4
    attnVecs = cell(tgtMaxLen, 1);
  end
  
  % separate emb
  if params.separateEmb==1 
    W_emb = model.W_emb_src;
  else
    W_emb = model.W_emb;
  end
  
  % Note: IMPORTANT. For attention-based models or positional models, it it
  % important to build the top hidden states first before moving to the
  % next time step. So DO NOT swap these for loops.
  W_layers = model.W_src;
  for tt=1:T % time
    tgtPos = tt-srcMaxLen+1;
    
    % switch to decoder mode
    if tt==srcMaxLen 
      W_layers = model.W_tgt;
    
      % separate emb
      if params.separateEmb==1 
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
        x_t = W_emb(:, input(:, tt));
        
        % prepare mask
        trainData.maskInfo{tt}.mask = inputMask(:, tt)'; % curBatchSize * 1
        trainData.maskInfo{tt}.unmaskedIds = find(trainData.maskInfo{tt}.mask);
        trainData.maskInfo{tt}.maskedIds = find(~trainData.maskInfo{tt}.mask);
        
        % attention model 3, 4
        if (params.attnFunc==3 || params.attnFunc==4) && tt>=(srcMaxLen-1) && tt<T

          % attnForward: h_t -> attnVecs (used the previous hidden state
          [attnVecs{tgtPos}, hid2softData.alignWeights] = attnLayerForward(model, all_h_t{ll, tt-1}, batchData.srcHidVecs, trainData.maskInfo{tt});
          x_t = [x_t; attnVecs{tgtPos}];
        end
        
        % positionl models 2: at the first level, we use additional src information
        if params.posModel==2 && tt>=srcMaxLen && mod(tgtPos, 2)==0 % predict words
          [s_t, trainData.srcPosInfo{tt}.linearIndices] = buildSrcPosVecs(tt, params, trainData, trainData.tgtOutput(:, tgtPos)', trainData.maskInfo{tt});
          x_t = [x_t; s_t];
          W = W_tgt_combined;
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
      
      % attention-based or positional-based models: keep track of all the src hidden states
      if tt==params.numSrcHidVecs && ll==params.numLayers && (params.attnFunc>0 || params.posModel>=2)
        % attention model 3, 4
        if (params.attnFunc==3 || params.attnFunc==4) && tt<T
          if params.assert
            assert(tt==(srcMaxLen-1));
          end
          
          if params.attnFunc==3
            startHidId = params.numAttnPositions-params.numSrcHidVecs+1;
            endHidId = params.numAttnPositions;
            batchData.srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
            batchData.srcHidVecs(:, :, startHidId:endHidId) = reshape([all_h_t{params.numLayers, 1:params.numSrcHidVecs}], params.lstmSize, curBatchSize, params.numSrcHidVecs);
          elseif params.attnFunc==4
            [startAttnId, endAttnId, startHidId, endHidId] = buildSrcHidVecs(srcMaxLen, tgtPos, params);
            batchData.srcHidVecs(:, :, startHidId:endHidId) = trainData.srcHidVecs(:, :, startAttnId:endAttnId);
          end
        else
          trainData.srcHidVecs = reshape([all_h_t{ll, 1:params.numSrcHidVecs}], params.lstmSize, curBatchSize, params.numSrcHidVecs);
        end
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
  [costs, softmaxGrad, grad_tgt_ht] = softmaxCostGrad(model, params, trainData, all_h_t(params.numLayers, :));
  if isTest==1 % don't compute grad
    return;
  else
    fields = fieldnames(softmaxGrad);
    for ii=1:length(fields)
      field = fields{ii};
      grad.(field) = softmaxGrad.(field);
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
  
  % separate embs
  if params.separateEmb==1
    allEmbGrads_src = zeroMatrix([params.lstmSize, trainData.numInputWords_src], params.isGPU, params.dataType);
    allEmbIndices_src = zeros(trainData.numInputWords_src, 1);
    wordCount_src = 0;
    
    % update the decoder
    if params.epoch>=params.epochUpdateDecoder
      allEmbGrads_tgt = zeroMatrix([params.lstmSize, trainData.numInputWords_tgt], params.isGPU, params.dataType);
      allEmbIndices_tgt = zeros(trainData.numInputWords_tgt, 1);
      wordCount_tgt = 0;
    end
  else
    allEmbGrads = zeroMatrix([params.lstmSize, trainData.numInputWords], params.isGPU, params.dataType);
    allEmbIndices = zeros(trainData.numInputWords, 1);
    wordCount = 0;
  end
  
  for tt=T:-1:1 % time
    unmaskedIds = trainData.maskInfo{tt}.unmaskedIds;
    tgtPos = tt-srcMaxLen+1;
    
    for ll=params.numLayers:-1:1 % layer
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
      
      [lstm_grad] = lstmUnitGrad(model, lstm, c_t, c_t_1, dc{ll}, dh{ll}, ll, tt, srcMaxLen, zeroState, params);
      dc{ll} = lstm_grad.dc;
      
      %% grad.W_src / grad.W_tgt
      if (tt>=srcMaxLen)
        % update the decoder
        if params.epoch>=params.epochUpdateDecoder
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
      dh{ll} = lstm_grad.input(end-params.lstmSize+1:end, :);
      if ll==1 % collect embedding grad
        embIndices = input(unmaskedIds, tt)';
        embGrad = lstm_grad.input(1:params.lstmSize, unmaskedIds);
        
        % pos model 2 lstm_grad.input = = [x_t; s_t; h_t]
        if params.posModel==2 && tt>=srcMaxLen && mod(tgtPos, 2)==0 % predict words
          linearIndices = trainData.srcPosInfo{tt}.linearIndices;
          grad.srcHidVecs(linearIndices) = grad.srcHidVecs(linearIndices) + reshape(lstm_grad.input(params.lstmSize+1:2*params.lstmSize, unmaskedIds), 1, []);          
        end

        numWords = length(embIndices);
        
        % separate embs
        if params.separateEmb==1
          if (tt<srcMaxLen)
            allEmbIndices_src(wordCount_src+1:wordCount_src+numWords) = embIndices;
            allEmbGrads_src(:, wordCount_src+1:wordCount_src+numWords) = embGrad;
            wordCount_src = wordCount_src + numWords;
          else
            % update the decoder
            if params.epoch>=params.epochUpdateDecoder
              allEmbIndices_tgt(wordCount_tgt+1:wordCount_tgt+numWords) = embIndices;
              allEmbGrads_tgt(:, wordCount_tgt+1:wordCount_tgt+numWords) = embGrad;
              wordCount_tgt = wordCount_tgt + numWords;
            end
          end
        else
          allEmbIndices(wordCount+1:wordCount+numWords) = embIndices;
          allEmbGrads(:, wordCount+1:wordCount+numWords) = embGrad;
          wordCount = wordCount + numWords;
        end
      else % pass down hidden state grad to the below layer
        dh{ll-1}(:, unmaskedIds) = dh{ll-1}(:, unmaskedIds) + lstm_grad.input(1:params.lstmSize, unmaskedIds);
      end
    end % end for layer
  end % end for time
  
  % grad W_emb
  if params.separateEmb==1 % % separate embs
    allEmbGrads_src(:, wordCount_src+1:end) = [];
    allEmbIndices_src(wordCount_src+1:end) = [];
    [grad.W_emb_src, grad.indices_src] = aggregateMatrix(allEmbGrads_src, allEmbIndices_src, params.isGPU, params.dataType);
    
    % update the decoder
    if params.epoch>=params.epochUpdateDecoder
      allEmbGrads_tgt(:, wordCount_tgt+1:end) = [];
      allEmbIndices_tgt(wordCount_tgt+1:end) = [];
      [grad.W_emb_tgt, grad.indices_tgt] = aggregateMatrix(allEmbGrads_tgt, allEmbIndices_tgt, params.isGPU, params.dataType);
    end
  else
    allEmbGrads(:, wordCount+1:end) = [];
    allEmbIndices(wordCount+1:end) = [];
    [grad.W_emb, grad.indices] = aggregateMatrix(allEmbGrads, allEmbIndices, params.isGPU, params.dataType);
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

function [grad, params] = initGrad(model, params)
  %% grad
  for ii=1:length(params.varsSelected)
    field = params.varsSelected{ii};
    if iscell(model.(field))
      for jj=1:length(model.(field)) % cell, like W_src, W_tgt
        grad.(field){jj} = zeroMatrix(size(model.(field){jj}), params.isGPU, params.dataType);
      end
    else
      grad.(field) = zeroMatrix(size(model.(field)), params.isGPU, params.dataType);
    end
  end
  
  %% backprop to src hidden states for attention and positional models
  if params.attnFunc>0 || params.posModel>=2
    params.numSrcHidVecs = params.srcMaxLen-1;
    assert(params.numSrcHidVecs<=params.T);
    
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


