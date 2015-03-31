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
  trainData.numInputWords = sum(sum(inputMask));
  
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
  
  if params.posModel==3 % positional models, include more src embeddings
    trainData.numInputWords = floor(trainData.numInputWords * 1.5);
  end
  
  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  trainData.maskInfo = cell(T, 1);
  if params.posModel==2
    trainData.srcPosInfo = cell(T, 1);
    W_tgt_combined = [model.W_tgt{1} model.W_tgt_pos];
  end
  
  % Note: IMPORTANT. For attention-based models or positional models, it it
  % important to build the top hidden states first before moving to the
  % next time step. So DO NOT swap these for loops.
  for t=1:T % time
    tgtPos = t-srcMaxLen+1;
    
    for ll=1:params.numLayers % layer
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
        c_t_1 = all_c_t{ll, t-1};
        h_t_1 = all_h_t{ll, t-1};
      end

      %% current-time input
      if ll==1 % first layer
        x_t = model.W_emb(:, input(:, t));
        
        % prepare mask
        trainData.maskInfo{t}.mask = inputMask(:, t)'; % curBatchSize * 1
        trainData.maskInfo{t}.unmaskedIds = find(trainData.maskInfo{t}.mask);
        trainData.maskInfo{t}.maskedIds = find(~trainData.maskInfo{t}.mask);
        
        %% positionl models 2: at the first level, we use additional src information
        if params.posModel==2 && t>=srcMaxLen && mod(tgtPos, 2)==0 % predict words
          [s_t, trainData.srcPosInfo{t}.linearIndices] = buildSrcPosVecs(t, params, trainData, trainData.tgtOutput(:, tgtPos)', trainData.maskInfo{t});
          x_t = [x_t; s_t];
          W = W_tgt_combined;
        end
      else % subsequent layer, use the previous-layer hidden state
        x_t = all_h_t{ll-1, t}; % lstm{ll-1, t}.h_t;
      end
      
      
      %% masking
      x_t(:, trainData.maskInfo{t}.maskedIds) = 0; 
      h_t_1(:, trainData.maskInfo{t}.maskedIds) = 0;
      c_t_1(:, trainData.maskInfo{t}.maskedIds) = 0;
      
      %% Core LSTM
      [lstms{ll, t}, all_h_t{ll, t}, all_c_t{ll, t}] = lstmUnit(W, x_t, h_t_1, c_t_1, ll, t, srcMaxLen, params, isTest); 
      
      % attention-based or positional-based models: keep track of all the src hidden states
      if t>=srcMaxLen && ll==params.numLayers && (params.attnFunc>0 || params.posModel>=2)
        trainData.srcHidVecs = reshape([all_h_t{params.numLayers, 1:params.numSrcHidVecs}], params.lstmSize, curBatchSize, params.numSrcHidVecs);
      end
    end % end for t
  end
  
  
  %%%%%%%%%%%%%%%
  %%% SOFTMAX %%%
  %%%%%%%%%%%%%%%
  [costs, softmaxGrad, otherSoftmaxGrads] = softmaxCostGrad(model, params, trainData, all_h_t(params.numLayers, :));
  if isTest==1 % don't compute grad
    return;
  else
    % collect all emb grads and aggregate later
    allEmbGrads = zeroMatrix([params.lstmSize, trainData.numInputWords], params.isGPU, params.dataType);
    allEmbIndices = zeros(trainData.numInputWords, 1);
    wordCount = 0;
  
    % softmax grads
    fields = fieldnames(softmaxGrad);
    for ii=1:length(fields)
      field = fields{ii};
      grad.(field) = softmaxGrad.(field); %grad.(field) + softmaxGrad.(field);
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
  
  for t=T:-1:1 % time
    unmaskedIds = trainData.maskInfo{t}.unmaskedIds;
    tgtPos = t-srcMaxLen+1;
    
    for ll=params.numLayers:-1:1 % layer
      %% hidden state grad
      if ll==params.numLayers
        if (t>=srcMaxLen) % get signals from the softmax layer
          dh{ll} = dh{ll} + otherSoftmaxGrads.ht{:, t-srcMaxLen+1};
        end

        if t<=params.numSrcHidVecs % attention/pos models: get feedback from grad.srcHidVecs
          dh{ll} = dh{ll} + grad.srcHidVecs(:,:,t);
        end
      end

      %% cell backprop
      if t==1
        c_t_1 = [];
      else
        c_t_1 = all_c_t{ll, t-1};
      end
      c_t = all_c_t{ll, t};
      lstm = lstms{ll, t};
      
      [lstm_grad] = lstmUnitGrad(model, lstm, c_t, c_t_1, dc{ll}, dh{ll}, ll, t, srcMaxLen, zeroState, params);
      dc{ll} = lstm_grad.dc;
      
      %% grad.W_src / grad.W_tgt
      if (t>=srcMaxLen)
        % positional
        if params.posModel==2 && ll==1 && mod(tgtPos, 2)==0 % predict words
          grad.W_tgt{ll} = grad.W_tgt{ll} + lstm_grad.W(:, 1:2*params.lstmSize);
          grad.W_tgt_pos = grad.W_tgt_pos + lstm_grad.W(:, 2*params.lstmSize+1:end);
        else
          grad.W_tgt{ll} = grad.W_tgt{ll} + lstm_grad.W;
        end
      else
        grad.W_src{ll} = grad.W_src{ll} + lstm_grad.W;
      end
      
      %% input grad: lstm_grad.input = [x_t; h_t] for normal models
      dh{ll} = lstm_grad.input(end-params.lstmSize+1:end, :);
      if ll==1 % collect embedding grad
        embIndices = input(unmaskedIds, t)';
        embGrad = lstm_grad.input(1:params.lstmSize, unmaskedIds);
        
        % pos model 2 lstm_grad.input = = [x_t; s_t; h_t]
        if params.posModel==2 && t>=srcMaxLen && mod(tgtPos, 2)==0 % predict words
          linearIndices = trainData.srcPosInfo{t}.linearIndices;
          grad.srcHidVecs(linearIndices) = grad.srcHidVecs(linearIndices) + reshape(lstm_grad.input(params.lstmSize+1:2*params.lstmSize, unmaskedIds), 1, []);          
        end

        numWords = length(embIndices);
        allEmbIndices(wordCount+1:wordCount+numWords) = embIndices;
        allEmbGrads(:, wordCount+1:wordCount+numWords) = embGrad;
        wordCount = wordCount + numWords;
      else % pass down hidden state grad to the below layer
        dh{ll-1}(:, unmaskedIds) = dh{ll-1}(:, unmaskedIds) + lstm_grad.input(1:params.lstmSize, unmaskedIds);
      end
    end % end for layer
  end % end for time
   
  % grad W_soft_inclass
  if params.numClasses>0
    grad.classIndices = otherSoftmaxGrads.classIndices;
    grad.W_soft_inclass = otherSoftmaxGrads.W_soft_inclass(:, :, grad.classIndices);
  end
  
  % grad W_emb
  allEmbGrads(:, wordCount+1:end) = [];
  allEmbIndices(wordCount+1:end) = [];
  [grad.W_emb, grad.indices] = aggregateMatrix(allEmbGrads, allEmbIndices, params.isGPU, params.dataType);
    
  % remove unused variables
  if params.attnFunc>0 || params.posModel>=2
    grad = rmfield(grad, 'srcHidVecs');
  end
  params = rmfield(params, {'curBatchSize', 'srcMaxLen', 'T'});
  
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



    
%     % update src embs from positional models
%     if params.posModel==2
%       numWords = otherSoftmaxGrads.wordCount;
%       allEmbIndices(wordCount+1:wordCount+numWords) = otherSoftmaxGrads.allEmbIndices;
%       allEmbGrads(:, wordCount+1:wordCount+numWords) = otherSoftmaxGrads.allEmbGrads;
%       wordCount = wordCount + numWords;
%     end

%           if params.posModel==1 % word embs
%             embIndices = [embIndices srcPosData(tgtPos).embIndices];
%             embGrad = [embGrad lstm_grad.input(range, unmaskedIds)];
%           elseif params.posModel==2 % embs of <p_n> and <p_eos>
%             embIndices = [embIndices params.nullPosId*ones(1, length(srcPosData(tgtPos).nullIds)) params.eosPosId*ones(1, length(srcPosData(tgtPos).eosIds))];
%             embGrad = [embGrad lstm_grad.input(range, [srcPosData(tgtPos).nullIds srcPosData(tgtPos).eosIds])];
% 
%             % update src hidden states
%             if ~isempty(srcPosData(tgtPos).posIds)
%               [linearIndices] = getTensorLinearIndices(trainData.srcHidVecs, srcPosData(tgtPos).posIds, srcPosData(tgtPos).colIndices);
%               grad.srcHidVecs(linearIndices) = grad.srcHidVecs(linearIndices) + reshape(lstm_grad.input(range, srcPosData(tgtPos).posIds), 1, []);
%             end
%           end


%   if params.embCPU && params.isGPU % only put part of the emb matrix onto GPU
%     input_embs = model.W_emb(:, input);
%     input_embs = gpuArray(input_embs); % load input embeddings onto GPUs
%   end

%         if params.embCPU && params.isGPU
%           x_t = input_embs(:, ((t-1)*curBatchSize+1):t*curBatchSize);
%         else
%         end

%     if params.embCPU
%       grad.W_emb = double(gather(grad.W_emb));
%     end


%     if params.attnFunc==2
%       params.numSrcHidVecs = params.srcMaxLen-1;
% %     elseif params.posModel==2 || params.posModel==3% add an extra <s_eos> to the src side
% %       params.numSrcHidVecs = params.srcMaxLen-2;
%     else
%       params.numSrcHidVecs = params.maxSentLen-1;
%     end

% last h_t, c_t on the src side
%   if params.inputFormat==1
%     last_h_t = cell(1, params.numLayers);
%     last_c_t = cell(1, params.numLayers);
%     last_lstm = cell(1, params.numLayers);
%     for ll=1:params.numLayers
%       last_h_t{ll} = zeroMatrix([params.lstmSize, curBatchSize], params.isGPU, params.dataType);
%       last_c_t{ll} = zeroMatrix([params.lstmSize, curBatchSize], params.isGPU, params.dataType);
%     end
%   end
  
  % topHidVecs: lstmSize * curBatchSize * T 
  %trainData.topHidVecs = zeroMatrix([params.lstmSize, curBatchSize, T], params.isGPU, params.dataType);
  
  
        
%       if params.inputFormat==1
%         if t==(srcMaxLen+1)
%           c_t_1 = last_c_t{ll}(:, :);
%         end
%         if t==srcMaxLen
%           lstm_cell = last_lstm{ll};
%         end
%       end

%         if params.inputFormat==1 && t==(srcMaxLen+1) % left-aligned
%           c_t_1 = last_c_t{ll};
%           h_t_1 = last_h_t{ll};
%         else
%           
%         end

%   if params.inputFormat==1 % left-aligned
%     topHidVecs(:, :, srcMaxLen) = last_h_t{params.numLayers};
%   end

      % for left-aligned input: copy previous states for masked indices
%       if params.inputFormat==1 && t<=srcMaxLen 
%         last_h_t{ll} = all_h_t{ll}(:, :, t);
%         last_c_t{ll} = all_c_t{ll}(:, :, t);
%         last_lstm{ll} = lstm{ll, t};
%         
%         if ~isempty(trainData.maskInfo{t}.maskedIds) 
%           last_h_t{ll}(:, trainData.maskInfo{t}.maskedIds) = all_h_t{ll}(:, trainData.maskInfo{t}.maskedIds, t-1);
%           last_c_t{ll}(:, trainData.maskInfo{t}.maskedIds) = all_c_t{ll}(:, trainData.maskInfo{t}.maskedIds, t-1);
%           
%           last_lstm{ll}.input(:, trainData.maskInfo{t}.maskedIds) = lstm{ll, t-1}.input(:, trainData.maskInfo{t}.maskedIds);
%           last_lstm{ll}.i_gate(:, trainData.maskInfo{t}.maskedIds) = lstm{ll, t-1}.i_gate(:, trainData.maskInfo{t}.maskedIds);
%           last_lstm{ll}.f_gate(:, trainData.maskInfo{t}.maskedIds) = lstm{ll, t-1}.f_gate(:, trainData.maskInfo{t}.maskedIds);
%           last_lstm{ll}.o_gate(:, trainData.maskInfo{t}.maskedIds) = lstm{ll, t-1}.o_gate(:, trainData.maskInfo{t}.maskedIds);
%           last_lstm{ll}.a_signal(:, trainData.maskInfo{t}.maskedIds) = lstm{ll, t-1}.a_signal(:, trainData.maskInfo{t}.maskedIds);
%           last_lstm{ll}.f_c_t(:, trainData.maskInfo{t}.maskedIds) = lstm{ll, t-1}.f_c_t(:, trainData.maskInfo{t}.maskedIds);
%           if params.dropout<1 % store dropout mask
%             last_lstm{ll}.dropoutMask(:, trainData.maskInfo{t}.maskedIds) = lstm{ll, t-1}.dropoutMask(:, trainData.maskInfo{t}.maskedIds);
%           end
%         end
%       end


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
