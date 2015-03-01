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
  input = trainData.input;
  inputMask = trainData.inputMask;
  
  srcMaxLen = trainData.srcMaxLen;
  tgtMaxLen = trainData.tgtMaxLen;
  curBatchSize = size(input, 1);
  
  T = srcMaxLen+tgtMaxLen-1;
  trainData.numInputWords = sum(sum(inputMask));
  if params.posModel==1 % positional models, include src embeddings
    trainData.numInputWords = trainData.numInputWords * 2;
  end
  
  if params.embCPU && params.isGPU % only put part of the emb matrix onto GPU
    input_embs = model.W_emb(:, input);
    input_embs = gpuArray(input_embs); % load input embeddings onto GPUs
  end
  
  trainData.isTest = isTest;
  trainData.curBatchSize = curBatchSize;
  trainData.T = T;
  
  [grad] = initGrad(model, params);
  zeroState = zeroMatrix([params.lstmSize, trainData.curBatchSize], params.isGPU, params.dataType);
  
  
  % collect all emb grads and aggregate later
  allEmbGrads = zeroMatrix([params.lstmSize, trainData.numInputWords], params.isGPU, params.dataType);
  allEmbIndices = zeros(trainData.numInputWords, 1);

  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  lstm = cell(params.numLayers, T); % each cell contains intermediate results for that timestep needed for backprop
  
  %% backprop to src hidden states for attention and positional models
  if params.attnFunc>0
    params.numSrcHidVecs = params.maxSentLen;
  elseif params.posModel==2 % add an extra <s_eos> to the src side
    params.numSrcHidVecs = srcMaxLen-2;
  else
    params.numSrcHidVecs = 0;
  end
  % lstmSize * curBatchSize * numSrcHidVecs 
  grad.srcHidVecs = zeroMatrix([params.lstmSize, curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
  
  % lstmSize * curBatchSize * T 
  trainData.topHidVecs = zeroMatrix([params.lstmSize, curBatchSize, T], params.isGPU, params.dataType);
  maskInfo = cell(T, 1);
  for ll=1:params.numLayers % layer
    for t=1:T % time
      %% decide encoder/decoder
      if (t>=srcMaxLen) % decoder
        W = model.W_tgt{ll};
      else % encoder
        W = model.W_src{ll};
      end
      
      %% previous-time input
      if t==1 % first time step
        h_t_1 = zeroState;
        c_t_1 = zeroState;
      else 
        c_t_1 = lstm{ll, t-1}.c_t;
        
        if ll==params.numLayers
          h_t_1 = trainData.topHidVecs(:, :, t-1);
        else
          h_t_1 = lstm{ll, t-1}.h_t;
        end
      end

      %% current-time input
      if ll==1 % first layer
        if params.embCPU && params.isGPU
          x_t = input_embs(:, ((t-1)*curBatchSize+1):t*curBatchSize);
        else
          x_t = model.W_emb(:, input(:, t));
        end

        % prepare mask
        maskInfo{t}.mask = inputMask(:, t)'; % curBatchSize * 1
        maskInfo{t}.unmaskedIds = find(maskInfo{t}.mask);
        maskInfo{t}.maskedIds = find(~maskInfo{t}.mask);
      else % subsequent layer, use the previous-layer hidden state
        x_t = lstm{ll-1, t}.h_t;
      end
      
      %% positional models: get source info
      if (params.posModel==1 || params.posModel==2) && t>=srcMaxLen && ll==1 % for positional models, at the first level, we use additional src information
        [s_t, posIds, nullIds, eosIds, colIndices, posEmbIndices] = buildSrcPosVecs(t, model, params, trainData, maskInfo{t});
        x_t = [x_t; s_t];
      end
      
      %% masking
      x_t(:, maskInfo{t}.maskedIds) = 0; 
      h_t_1(:, maskInfo{t}.maskedIds) = 0;
      c_t_1(:, maskInfo{t}.maskedIds) = 0;
      
      %% Core LSTM
      [lstm{ll, t}, cell_h_t] = lstmUnit(W, x_t, h_t_1, c_t_1, ll, t, srcMaxLen, params, isTest);
      
      %% positional model: store info
      if (params.posModel==1 || params.posModel==2) && t>=srcMaxLen && ll==1 % for positional models
        if params.posModel==1
          lstm{ll, t}.embIndices = posEmbIndices;
        elseif params.posModel==2
          lstm{ll, t}.posIds = posIds;
          lstm{ll, t}.nullIds = nullIds;
          lstm{ll, t}.eosIds = eosIds;
          lstm{ll, t}.colIndices = colIndices;
        end
      end
      
      %% attention mechanism: keep track of src hidden states at the top level
      if ll==params.numLayers
        trainData.topHidVecs(:, :, t) = cell_h_t;
      else
        lstm{ll, t}.h_t = cell_h_t;
      end
        
      
    end % end for t
  end
  
  %%%%%%%%%%%%%%%
  %%% SOFTMAX %%%
  %%%%%%%%%%%%%%%
  [costs, softmaxGrad, topGradHt] = softmaxCostGrad(model, params, trainData);
  if isTest==1 % don't compute grad
    return;
  else
    % softmax grads
    fields = fieldnames(softmaxGrad);
    for ii=1:length(fields)
      field = fields{ii};
      grad.(field) = grad.(field) + softmaxGrad.(field);
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
  
  wordCount = 0;
  for t=T:-1:1 % time
    unmaskedIds = maskInfo{t}.unmaskedIds;
    numWords = length(unmaskedIds);
    
    for ll=params.numLayers:-1:1 % layer
      %% hidden state grad
      if ll==params.numLayers
        if (t>=srcMaxLen) || (params.posModel>0 && t==(srcMaxLen-1)) % get signals from the softmax layer
          dh{ll} = dh{ll} + topGradHt(:, :, t);
        end
        if t<params.numSrcHidVecs % attention/pos models: get feedback from grad.srcHidVecs
          dh{ll} = dh{ll} + grad.srcHidVecs(:,:,t);
        end
      end

      %% cell backprop
      [lstm_grad] = lstmUnitGrad(model, lstm, dc{ll}, dh{ll}, ll, t, srcMaxLen, zeroState, params);
      % lstm_grad.input = [x_t; h_t] for normal models, = [x_t; s_t; h_t] for positional models
      dc{ll} = lstm_grad.dc;
      dh{ll} = lstm_grad.input(end-params.lstmSize+1:end, :);

      %% grad.W_src / grad.W_tgt
      if (t>=srcMaxLen)
        grad.W_tgt{ll} = grad.W_tgt{ll} + lstm_grad.W;
      else
        grad.W_src{ll} = grad.W_src{ll} + lstm_grad.W;
      end
      
      %% input grad
      embGrad = lstm_grad.input(1:params.lstmSize, unmaskedIds);
      if ll==1 % collect embedding grad
        allEmbIndices(wordCount+1:wordCount+numWords) = input(unmaskedIds, t);
        allEmbGrads(:, wordCount+1:wordCount+numWords) = embGrad;
        wordCount = wordCount + numWords;

        if params.posModel>0 && t>=srcMaxLen% positional models
          posEmbGrad = lstm_grad.input(params.lstmSize+1:2*params.lstmSize, :);
          if params.posModel==1 % update embeddings
            allEmbIndices(wordCount+1:wordCount+numWords) = lstm{ll, t}.embIndices;
            allEmbGrads(:, wordCount+1:wordCount+numWords) = posEmbGrad(:, unmaskedIds);
            wordCount = wordCount + numWords;
          elseif params.posModel==2
            % update embs of <p_n> and <p_eos>
            tmpIndices = [lstm{ll, t}.nullIds lstm{ll, t}.eosIds];
            numWords = length(tmpIndices);
            allEmbIndices(wordCount+1:wordCount+numWords) = [params.nullPosId*ones(1, length(lstm{ll, t}.nullIds)) params.eosPosId*ones(1, length(lstm{ll, t}.eosIds))];
            allEmbGrads(:, wordCount+1:wordCount+numWords) = posEmbGrad(:, tmpIndices);
            wordCount = wordCount + numWords;
            
            % update src hidden states
            if ~isempty(lstm{ll, t}.posIds)
              posIds = lstm{ll, t}.posIds;
              colIndices = lstm{ll, t}.colIndices;
              numPositions = length(posIds);
              xIds = repmat(1:params.lstmSize, 1, numPositions);
              yIds = repmat(posIds, params.lstmSize, 1);
              yIds = yIds(:)';
              zIds = repmat(colIndices, params.lstmSize, 1);
              zIds = zIds(:)';
              linearIndices = sub2ind(size(trainData.topHidVecs), xIds, yIds, zIds);
              grad.srcHidVecs(linearIndices) = grad.srcHidVecs(linearIndices) + reshape(posEmbGrad(:, posIds), 1, numPositions*params.lstmSize);
            end
          end
        end
      else % pass down hidden state grad to the below layer
        dh{ll-1}(:, unmaskedIds) = dh{ll-1}(:, unmaskedIds) + embGrad;
      end
    end % end for layer
  end % end for time
   
  % grad W_emb
  if params.posModel>0
    allEmbGrads(:, wordCount+1:end) = [];
    allEmbIndices(wordCount+1:end) = [];
  else
    if params.assert
      assert(wordCount == trainData.numInputWords);
      if params.numClasses>0
        assert(allClassCount <= trainData.numInputWords);
      end
    end
  end
  [grad.W_emb, grad.indices] = aggregateMatrix(allEmbGrads, allEmbIndices, params.isGPU, params.dataType);
    
  % remove unused variables
  if params.attnFunc>0 || params.posModel==2
    grad = rmfield(grad, 'srcHidVecs');
  end
  
  if params.isGPU
    if params.embCPU
      grad.W_emb = double(gather(grad.W_emb));
    end
    
    % costs
    costs.total = gather(costs.total);
    if params.posModel>0
      costs.pos = gather(costs.pos);
      costs.word = gather(costs.word);
    end
  end
end

function [grad] = initGrad(model, params)
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
end

  
  % global opt
  %if params.globalOpt==1
  %  srcSentEmbs = sum(reshape(input_embs(:, 1:curBatchSize*srcMaxLen), params.lstmSize*curBatchSize, srcMaxLen), 2); % sum
  %  srcSentEmbs = bsxfun(@rdivide, reshape(srcSentEmbs, params.lstmSize, curBatchSize), trainData.srcLens');
  %end

%     % grad_ht
%     for t=(srcMaxLen-1):T
%       % words
%       if (t>=srcMaxLen)
%         lstm{params.numLayers, t}.grad_ht = otherGrad.word_ht(:, (t-srcMaxLen)*curBatchSize+1:(t-srcMaxLen+1)*curBatchSize);
%       end
%       
%       % positions
%       if params.posModel>0 && t<T
%         range = (t-srcMaxLen+1)*curBatchSize+1:(t-srcMaxLen+2)*curBatchSize;
%         if t==(srcMaxLen-1)
%           lstm{params.numLayers, t}.grad_ht = otherGrad.pos_ht(:, range);
%         else
%           lstm{params.numLayers, t}.grad_ht = lstm{params.numLayers, t}.grad_ht + otherGrad.pos_ht(:, range);
%         end
%       end
%     end


%   if params.numClasses>0 % this might cause memory problem. We can actually do aggregation as we go
%     allClassGrads = zeroMatrix([params.softmaxSize*params.classSize, trainData.numInputWords], params.isGPU, params.dataType);
%     allClassIndices = zeros(trainData.numInputWords, 1);
%     allClassCount = 0;
%   end

%   if params.numClasses>0 % class-based softmax
%     allClassGrads(:, allClassCount+1:end) = [];
%     allClassIndices(allClassCount+1:end) = [];
%     [grad.W_soft_inclass, grad.classIndices] = aggregateMatrix(allClassGrads, allClassIndices, params.isGPU, params.dataType);
%     grad.W_soft_inclass = reshape(grad.W_soft_inclass, [params.classSize params.softmaxSize length(grad.classIndices)]);
%   end

%       if params.numClasses>0 % class-based softmax
%         numClassIds = length(classSoftmax.indices);
%         allClassIndices(allClassCount+1:allClassCount+numClassIds) = classSoftmax.indices;
%         allClassGrads(:, allClassCount+1:allClassCount+numClassIds) = classSoftmax.W_soft_inclass;
%         allClassCount = allClassCount + numClassIds;
%       end
