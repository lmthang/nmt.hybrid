function [costs, grad] = lstmCostGrad(model, trainData, params, isTest)
%%%
%
% Compute cost/grad for LSTM. 
% When params.posModel>0, returns costs.pos and costs.word
% If isTest==1, this method only computes cost (for testing purposes).
%
% Thang Luong @ 2014, <lmthang@stanford.edu>
%
%%%

  %%%%%%%%%%%%
  %%% INIT %%%
  %%%%%%%%%%%%
  input = trainData.input;
  inputMask = trainData.inputMask;
  srcMaxLen = trainData.srcMaxLen;
  tgtMaxLen = trainData.tgtMaxLen;
  tgtOutput = trainData.tgtOutput;
  
  % positional models
  if params.posModel>0
    srcPos = trainData.srcPos;
  end
  
  T = srcMaxLen+tgtMaxLen-1;
  curBatchSize = size(input, 1);
  trainData.numInputWords = sum(sum(inputMask));
  if params.posModel==1 % positional models, include src embeddings
    trainData.numInputWords = trainData.numInputWords * 2;
  end
  indices = zeros(trainData.numInputWords, 1);
  if params.embCPU && params.isGPU % only put part of the emb matrix onto GPU
    input_embs = model.W_emb(:, input);
    input_embs = gpuArray(input_embs); % load input embeddings onto GPUs
  end
  
  trainData.isTest = isTest;
  trainData.curBatchSize = curBatchSize;
  [grad, zeroState, costs, emb] = initGrad(model, params, trainData);

  
  % global opt
  %if params.globalOpt==1
  %  srcSentEmbs = sum(reshape(input_embs(:, 1:curBatchSize*srcMaxLen), params.lstmSize*curBatchSize, srcMaxLen), 2); % sum
  %  srcSentEmbs = bsxfun(@rdivide, reshape(srcSentEmbs, params.lstmSize, curBatchSize), trainData.srcLens');
  %end

  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  lstm = cell(params.numLayers, T); % each cell contains intermediate results for that timestep needed for backprop
  
  % attention mechanism
  if params.attnFunc>0
    % arrange the tensor this way so as to use bsxfun for alignWeights later in which the last
    % dimension corresponds to a singleton dimension in alignWeights.
    trainData.srcAlignStates = zeroMatrix([params.lstmSize, curBatchSize, params.maxSentLen], params.isGPU, params.dataType);
    grad.srcAlignStates = zeroMatrix([params.lstmSize, curBatchSize, params.maxSentLen], params.isGPU, params.dataType);
  end
  
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
        h_t_1 = lstm{ll, t-1}.h_t; 
        c_t_1 = lstm{ll, t-1}.c_t;
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
      
      %% masking
      x_t(:, maskInfo{t}.maskedIds) = 0; 
      h_t_1(:, maskInfo{t}.maskedIds) = 0;
      c_t_1(:, maskInfo{t}.maskedIds) = 0;
     
      %% dropout
      if params.dropout<1 && isTest==0
        if ~params.isGradCheck
          if params.isGPU
            dropoutMask = (rand(size(x_t), 'gpuArray')<params.dropout)/params.dropout;
          else
            dropoutMask = (rand(size(x_t))<params.dropout)/params.dropout;
          end
        else % for gradient check use the same mask
          dropoutMask = params.dropoutMask;
        end
        x_t = x_t.*dropoutMask;
      end
      
      %% lstm cell
      if params.posModel>0 && t>=srcMaxLen && ll==1 % for positional models, at the first level, we use additional src information
        [s_t, posIds, nullIds, eosIds, embIndices] = buildSrcPosVecs(t, model, params, trainData, maskInfo{t});
        lstm{ll, t} = lstmUnit(W, x_t, h_t_1, c_t_1, params, isTest, s_t);
        
        lstm{ll, t}.posIds = posIds;
        lstm{ll, t}.nullIds = nullIds;
        lstm{ll, t}.eosIds = eosIds;
        if params.posModel==1
          lstm{ll, t}.embIndices = embIndices;
        end
      else % normal
        lstm{ll, t} = lstmUnit(W, x_t, h_t_1, c_t_1, params, isTest);
      end
      
      % store dropout mask
      if params.dropout<1 && isTest==0
        lstm{ll, t}.dropoutMask = dropoutMask;
      end
      
      %% attention mechanism: keep track of src hidden states at the top level
      if params.attnFunc>0 && ll==params.numLayers && (t<srcMaxLen)
        trainData.srcAlignStates(:, 1:curBatchSize, t) = lstm{ll, t}.h_t;
      end
        
      %% prediction at the top layer
      % for positional models, we start predict positions from (srcMaxLen-1) and stop at (T-1)
      if ll==params.numLayers && (t>=srcMaxLen || (t>=(srcMaxLen-1) && params.posModel>0)) 
        % predict positions
        if (params.posModel>0 && t<T)
          predictedPositions = srcPos(:, t-srcMaxLen+2)'- (params.startPosId-1);
          [pos_cost, pos_softmaxGrad, pos_grad_ht] = softmaxCostGrad('W_softPos', lstm{ll, t}.h_t, predictedPositions, model, params, trainData, maskInfo{t});
          costs.total = costs.total + pos_cost;
          costs.pos = costs.pos + pos_cost;
        end
        
        % predict words
        if (t>=srcMaxLen)
          predictedWords = tgtOutput(:, t-srcMaxLen+1)';
          [word_cost, word_softmaxGrad, word_grad_ht] = softmaxCostGrad('W_soft', lstm{ll, t}.h_t, predictedWords, model, params, trainData, maskInfo{t});
          costs.total = costs.total + word_cost;
          costs.word = costs.word + word_cost;
        end

        % grad
        if isTest==0 % compute grads when we are not testing
          % words
          if (t>=srcMaxLen)
            fields = fieldnames(word_softmaxGrad);
            for ii=1:length(fields)
              field = fields{ii};
              grad.(field) = grad.(field) + word_softmaxGrad.(field);
            end

            % grad_ht
            lstm{ll, t}.grad_ht = word_grad_ht;
          end
          
          % positions
          if (params.posModel>0 && t<T)
            fields = fieldnames(pos_softmaxGrad);
            for ii=1:length(fields)
              field = fields{ii};
              grad.(field) = grad.(field) + pos_softmaxGrad.(field);
            end
            
            % grad_ht
            if t==(srcMaxLen-1)
              lstm{ll, t}.grad_ht = pos_grad_ht;
            else
              lstm{ll, t}.grad_ht = lstm{ll, t}.grad_ht + pos_grad_ht;
            end
          end
        end
      end
    end
  end
  
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
  
  wordCount = 0;
  for t=T:-1:1 % time
    unmaskedIds = maskInfo{t}.unmaskedIds;
    numWords = length(unmaskedIds);
    
    for ll=params.numLayers:-1:1 % layer
      %% hidden state grad
      if ll==params.numLayers
        if (t>=srcMaxLen) || (params.posModel>0 && t==(srcMaxLen-1)) % get signals from the softmax layer
          dh{ll} = dh{ll} + lstm{ll, t}.grad_ht;
        elseif params.attnFunc>0 % attention model: get feedback from grad.srcAlignStates
          dh{ll} = dh{ll} + grad.srcAlignStates(:,:,t);
        end
      end

      %% cell backprop
      [lstm_grad] = lstmUnitGrad(model, lstm, dc{ll}, dh{ll}, ll, t, srcMaxLen, zeroState, params);
      dc{ll} = lstm_grad.dc;
      dh{ll} = lstm_grad.input(params.lstmSize+1:2*params.lstmSize, :);

      %% grad.W_src / grad.W_tgt
      if (t>=srcMaxLen)
        grad.W_tgt{ll} = grad.W_tgt{ll} + lstm_grad.W;
      else
        grad.W_src{ll} = grad.W_src{ll} + lstm_grad.W;
      end

      %% dropout
      if params.dropout<1
        embGrad = lstm_grad.input(1:params.lstmSize, :).*lstm{ll, t}.dropoutMask;
        embGrad = embGrad(:, unmaskedIds);  
      else
        embGrad = lstm_grad.input(1:params.lstmSize, unmaskedIds);
      end
      
      
      %% input grad
      if ll==1 % collect embedding grad
        indices(wordCount+1:wordCount+numWords) = input(unmaskedIds, t);
        emb(:, wordCount+1:wordCount+numWords) = embGrad;
        wordCount = wordCount + numWords;

        if params.posModel==1 && t>=srcMaxLen% positional models
          indices(wordCount+1:wordCount+numWords) = lstm{ll, t}.embIndices;
          emb(:, wordCount+1:wordCount+numWords) = lstm_grad.input(2*params.lstmSize+1:3*params.lstmSize, unmaskedIds);
          wordCount = wordCount + numWords;
        end
      else % pass down hidden state grad to the below layer
        dh{ll-1}(:, unmaskedIds) = dh{ll-1}(:, unmaskedIds) + embGrad;
      end
    end % end for layer
  end % end for time
   
  % grad W_emb
  if params.posModel>0
    trainData.numInputWords = wordCount;
    emb(:, wordCount+1:end) = [];
    indices(wordCount+1:end) = [];
  end
  [grad.W_emb, grad.indices] = aggregateMatrix(emb, indices, params.isGPU, params.dataType);
  
  if params.attnFunc>0
    grad = rmfield(grad, 'srcAlignStates');
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

function [grad, zero_state, costs, emb] = initGrad(model, params, trainData)
  zero_state = zeroMatrix([params.lstmSize, trainData.curBatchSize], params.isGPU, params.dataType);
  
  %% grad
  for ii=1:length(params.varsNoEmb)
    field = params.varsNoEmb{ii};
    if iscell(model.(field))
      for jj=1:length(model.(field)) % cell, like W_src, W_tgt
        grad.(field){jj} = zeroMatrix(size(model.(field){jj}), params.isGPU, params.dataType);
      end
    else
      grad.(field) = zeroMatrix(size(model.(field)), params.isGPU, params.dataType);
    end
  end

  % emb
  emb = zeroMatrix([params.lstmSize, trainData.numInputWords], params.isGPU, params.dataType);
  
  % costs
  costs.total = zeroMatrix([1, 1], params.isGPU, params.dataType);
  costs.word = zeroMatrix([1, 1], params.isGPU, params.dataType);
  if params.posModel > 0
    costs.pos = zeroMatrix([1, 1], params.isGPU, params.dataType);
  end
end

            
%           probs(scoreIndices) = probs(scoreIndices) - 1; % minus one at predicted words
%                       
%           % grad_softmax_h
%           grad_softmax_h = model.W_soft'* probs;
%           
%           % grad.W_soft
%           grad.W_soft = grad.W_soft + probs*softmax_h';
%           
%           %% softmax compress or attention
%           if params.softmaxDim>0 || params.attnFunc>0 
%             if params.softmaxDim>0 % f(W_h * h_t)
%               % f'(softmax_h).*grad_softmax_h
%               tmpResult = params.nonlinear_f_prime(softmax_h).*grad_softmax_h;
%               
%               % grad.W_h
%               grad.W_h = grad.W_h + tmpResult*lstm{ll, t}.h_t';
% 
%               % grad_ht
%               lstm{ll, t}.grad_ht = model.W_h'*tmpResult;
%             elseif params.attnFunc>0 % f(W_ah*[attn_t; tgt_h_t])
%               [attnGrad] = attnBackprop(model, srcAlignStates, softmax_h, grad_softmax_h, attn_h_concat, alignWeights, alignScores, attnInput, params);
%               
%               % grad.W_ah
%               grad.W_ah = grad.W_ah + attnGrad.W_ah;
% 
%               % grad_ht
%               lstm{ll, t}.grad_ht = attnGrad.ht;
%               
%               % grad.srcAlignStates
%               grad.srcAlignStates = grad.srcAlignStates + attnGrad.srcAlignStates;
%   
%               % grad.W_a
%               grad.W_a = grad.W_a + attnGrad.W_a;
%             end
%           %% normal softmax
%           else 
%             % grad_ht
%             lstm{ll, t}.grad_ht = grad_softmax_h;
%           end          
%         if params.attnFunc==1 || params.attnFunc==2 % premultiply W_a
%           srcAlignStates(:, :, t) = model.W_a * lstm{ll, t}.h_t; % W_a * src_h
%         elseif params.attnFunc==3  
%         end

%   
%   if params.attnFunc==1 % f(tgt_h' * W_a * src_h)
%     % we have premultiplied W_a in srcAlignStates
%     alignWeights = params.nonlinear_f(squeeze(sum(bsxfun(@times, srcAlignStates, h_t)))); % curBatchSize * (srcMaxLen-1)
% 
%     % assert
%     if params.assert
%       results = zeros(curBatchSize, srcMaxLen-1);
%       for iii=1:(srcMaxLen-1)
%         results(:, iii) = transpose(sum(srcAlignStates(:, :, iii).*h_t));
%       end
%       assert(sum(sum(abs(alignWeights-results)))<1e-5);
%     end
%   elseif params.attnFunc==2 % v_a' * f(W_a_tgt * tgt_h +  W_a * src_h)
%     % we have premultiplied W_a in srcAlignStates
%     alignWeights = params.nonlinear_f(bsxfun(@plus, srcAlignStates, model.W_a_tgt*h_t)); % lstmSize * curBatchSize * (srcMaxLen-1)
%     alignWeights = squeeze(sum(bsxfun(@times, alignWeights, model.v_a))); % curBatchSize * (srcMaxLen-1)
% 
%     % assert
%     if params.assert
%       results = zeros(curBatchSize, srcMaxLen-1);
%       tgtAlignState = model.W_a_tgt*h_t;
%       for iii=1:(srcMaxLen-1)
%         results(:, iii) = transpose(model.v_a'*tanh(srcAlignStates(:, :, iii) + tgtAlignState));
%       end
%       assert(sum(sum(abs(alignWeights-results)))<1e-5);
%     end
%   else
%   end
% 
%   if params.assert % curBatchSize * (srcMaxLen-1)
%     assert(size(alignWeights, 1)==curBatchSize);
%     assert(size(alignWeights, 2)==(srcMaxLen-1));
%   end

%     if params.isGPU
%       srcAlignStates = zeros([params.lstmSize, curBatchSize, srcMaxLen-1], params.dataType, 'gpuArray');
%     else
%       srcAlignStates = zeros([params.lstmSize, curBatchSize, srcMaxLen-1]);
%     end

%   if params.isBi
%     grad.W_src = cell(params.numLayers, 1);
%   end
%   grad.W_tgt = cell(params.numLayers, 1);
%  
%   % W_soft
%   if params.softmaxDim>0
%     grad.W_h = zeroMatrix([params.softmaxDim, params.lstmSize], params.isGPU, params.dataType);
%     grad.W_soft = zeroMatrix([params.outVocabSize, params.softmaxDim], params.isGPU, params.dataType);
%   else
%     grad.W_soft = zeroMatrix([params.outVocabSize, params.lstmSize], params.isGPU, params.dataType);
%   end
%   
%   % W_src
%   if params.isBi
%     for ll=1:params.numLayers
%       grad.W_src{ll} = zeroMatrix([4*params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
%     end
%   end
%   
%   % W_tgt
%   for ll=1:params.numLayers
%     grad.W_tgt{ll} = zeroMatrix([4*params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
%   end

  %if params.isGPU
  %  [grad.indices, ~, J] = unique(indices);
  %  numUniqIndices = length(grad.indices);
  %  numEmbGrads = length(indices);
  %  sparseMatrix = zeros(numEmbGrads, numUniqIndices, params.dataType, 'gpuArray');
  %  sparseIndices = sub2ind([numEmbGrads, numUniqIndices], 1:numEmbGrads, J'); 
  %  sparseMatrix(sparseIndices) = ones(numEmbGrads, 1);
  %  grad.W_emb = emb*sparseMatrix;
  %  totalCost = gather(totalCost);
  %else
  %  grad.W_emb = aggregateMatrix(emb, indices, params.inVocabSize);
  %  grad.W_emb = full(grad.W_emb(:, grad.indices));
  %  grad.indices = unique(indices);
  %end

        %scores = model.W_soft * lstm{ll, t}.h_t;  % params.outVocabSize * curBatchSize
        %mx = max(scores);
        %scores = bsxfun(@minus, scores, mx); % subtract max elements 
        %probs = exp(scores); % unnormalized probs 
        %norms = sum(probs); % normalization factors
        %probs = bsxfun(@times, probs, curMask./norms); % normalized probs

    %grad.indices = unique(indices);
    %grad.W_emb = aggregateMatrix(double(gather(emb)), indices, params.inVocabSize);
    %grad.W_emb = gpuArray(full(grad.W_emb(:, grad.indices)));


%        if isFast
%        else
%          scores = model.W_soft * lstm{ll, t}.h_t(:, curMask);  % params.outVocabSize * numWords
%          mx = max(scores);
%          log_probs = bsxfun(@minus, scores, log(sum(exp(bsxfun(@minus, scores, mx)))) + mx); 
%        
%          % select from scores matrix, one number per column
%          scoreIndices = sub2ind([params.outVocabSize, numWords], tgtPredictedWords, 1:numWords); % 1 * numWords
%
%          % cost
%          totalCost = totalCost - sum(log_probs(scoreIndices));
%        end

%          if isFast
%          else
%            probs = exp(log_probs); % out_size * numWords
%            probs(scoreIndices) = probs(scoreIndices) - ones(1, numWords); % minus one at predicted words
%            grad.W_soft = grad.W_soft + probs*lstm{ll, t}.h_t(:, curMask)';
%          end

%        if isFast
%        else
%          dh{ll}(:, mask) = dh{ll}(:, mask) + lstm{ll, t}.grad_ht; 
%        end

  %  grad.W_soft = gather(grad.W_soft);
  %  if params.isBi
  %    for ll=1:params.numLayers
  %      grad.W_src{ll} = gather(grad.W_src{ll});
  %    end
  %  end
  %  
  %  for ll=1:params.numLayers
  %    grad.W_tgt{ll} = gather(grad.W_tgt{ll});
  %  end
%         if params.isGradCheck
%           indices = input(mask, t);
%           emb_grad = lstm_grad.d_xh(1:params.lstmSize, mask);
%           for jj=1:length(indices)
%             grad.W_emb(:, indices(jj)) = grad.W_emb(:, indices(jj)) + emb_grad(:, jj);
%           end
% 
% %           if params.isGPU
% %             emb_grad = double(gather(lstm_grad.d_xh(1:params.lstmSize, mask))); % copy embedding grads to CPU
% %             for jj=1:length(indices)
% %               grad.W_emb(:, indices(jj)) = grad.W_emb(:, indices(jj)) + emb_grad(:, jj);
% %             end
% %             %grad.W_emb = grad.W_emb + aggregateMatrix(emb_grad, indices, params.inVocabSize); %, params.isGPU);
% %           else
% %             emb_grad = lstm_grad.d_xh(1:params.lstmSize, mask);
% %             grad.W_emb = grad.W_emb + aggregateMatrix(emb_grad, indices, params.inVocabSize); %, params.isGPU);
% %           end
%         else
%           
%         end

%log_probs = bsxfun(@minus, scores, simpleLogSumExp(scores));
%log_probs = bsxfun(@minus, scores, logsumexp(scores));
      
%function [norms] = simpleLogSumExp(scores)
%  mx = max(scores);
%  norms = bsxfun(@plus, log(sum(exp(bsxfun(@minus, scores, mx)))), mx);
%end
%
%
%function [clippedValue] = clip(x, thres)
%  if x>thres
%    clippedValue = thres;
%  elseif x<-thres
%    clippedValue = -thres;
%  else
%    clippedValue = x;
%  end
%end
%
%function [next_c] = cellFun(f_t, c_t, i_t, a_t)
% next_c = f_t*c_t + i_t*a_t;
%end

%     h_t = zeros([params.lstmSize, curBatchSize], dataType, 'gpuArray');
%     c_t = zeros([params.lstmSize, curBatchSize], dataType, 'gpuArray');
    
%     h_t = zeros([params.lstmSize, curBatchSize]);
%     c_t = zeros([params.lstmSize, curBatchSize]);

%     %% input, forget, output gates and input signals before applying non-linear functions
%     if t==1
%       ifoa_linear = W*[x_t; zero_state];    
%     else
%       ifoa_linear = W*[x_t; lstm{t-1}.h_t];    
%     end
%     
%     %% cell
%     % GPU note: the below non-linear functions are fast, so no need to use arrayfun
%     ifo_gate = params.nonlinear_gate_f(ifoa_linear(1:3*params.lstmSize, :));
%     lstm{t}.i_gate = ifo_gate(1:params.lstmSize, :);
%     lstm{t}.f_gate = ifo_gate(params.lstmSize+1:2*params.lstmSize, :);
%     lstm{t}.o_gate = ifo_gate(2*params.lstmSize+1:3*params.lstmSize, :);
%     lstm{t}.a_signal = params.nonlinear_f(ifoa_linear(3*params.lstmSize+1:4*params.lstmSize, :)); % note input uses a different activation function
%     if t==1
%       lstm{t}.c_t = lstm{t}.f_gate.*zero_state + lstm{t}.i_gate.*lstm{t}.a_signal; % c_t = f_t * c_{t-1} + i_t * a_t
%     else
%       lstm{t}.c_t = lstm{t}.f_gate.*lstm{t-1}.c_t + lstm{t}.i_gate.*lstm{t}.a_signal; % c_t = f_t * c_{t-1} + i_t * a_t
%     end
%     
%     %% hidden
%     lstm{t}.f_c_t = params.nonlinear_f(lstm{t}.c_t);
%     lstm{t}.h_t = lstm{t}.o_gate.*lstm{t}.f_c_t; % h_t = o_t * g(c_t)
%     
%     % clip
%     if params.isClip
%       if params.isGPU
%        c_t = arrayfun(@clipForward, c_t);
%        h_t = arrayfun(@clipForward, h_t);
%       else
%        c_t(c_t>params.clipForward) = params.clipForward; c_t(c_t<-params.clipForward) = -params.clipForward; % clip: keep memory small
%        h_t(h_t>params.clipForward) = params.clipForward; h_t(h_t<-params.clipForward) = -params.clipForward; % clip: keep hidden state small
%       end
%     end


%   if params.isGPU
%     dh = zeros(params.lstmSize, curBatchSize, dataType, 'gpuArray');
%     dc = zeros(params.lstmSize, curBatchSize, dataType, 'gpuArray');
%   else
%     dh = zeros(params.lstmSize, curBatchSize);
%     dc = zeros(params.lstmSize, curBatchSize);
%   end

%     dc = dc + params.nonlinear_f_prime(lstm{t}.f_c_t).*lstm{t}.o_gate.*dh;
%     
%     di = params.nonlinear_gate_f_prime(lstm{t}.i_gate).*lstm{t}.a_signal.*dc;
%     if t>1
%       df = params.nonlinear_gate_f_prime(lstm{t}.f_gate).*lstm{t-1}.c_t.*dc;
%     else
%       if params.isGPU
%         df = zeros(params.lstmSize, curBatchSize, dataType, 'gpuArray');
%       else
%         df = zeros(params.lstmSize, curBatchSize);
%       end
%     end
% 
%     % arrayfun doesn't work here for GPU
%     do = params.nonlinear_gate_f_prime(lstm{t}.o_gate).*lstm{t}.f_c_t .* dh;
%     da = params.nonlinear_f_prime(lstm{t}.a_signal).*lstm{t}.i_gate.*dc;   
% 
%     %x_t = model.W_emb(:, input(:, t));
%     x_t = input_embs(:, ((t-1)*curBatchSize+1):t*curBatchSize);
%     if (t>=srcMaxLen) % grad tgt
%       W = model.W_tgt;
%       if t==1
%         grad.W_tgt = grad.W_tgt + [di; df; do; da]*[x_t; zeros(params.lstmSize, curBatchSize)]';
%       else
%         grad.W_tgt = grad.W_tgt + [di; df; do; da]*[x_t; lstm{t-1}.h_t]';
%       end
%     else % grad src
%       W = model.W_src;
%       if t==1
%         grad.W_src = grad.W_src + [di; df; do; da]*[x_t; zeros(params.lstmSize, curBatchSize)]';
%       else
%         grad.W_src = grad.W_src + [di; df; do; da]*[x_t; lstm{t-1}.h_t]';
%       end
%     end
%     
%     dx = W(:, 1:params.lstmSize)'*[di; df; do; da];
%     dc = lstm{t}.f_gate.*dc;
%     dh = W(:, params.lstmSize+1:end)'*[di; df; do; da];
%     
%     
%     % clip hidden/cell derivatives
%     if params.isClip
%       if params.isGPU
%        dh = arrayfun(@clipBackward, dh);
%        dc = arrayfun(@clipBackward, dc);
%       else
%        dh(dh>params.clipBackward) = params.clipBackward; dh(dh<-params.clipBackward) = -params.clipBackward;
%        dc(dc>params.clipBackward) = params.clipBackward; dc(dc<-params.clipBackward) = -params.clipBackward;
%       end
%     end

    %% tried arrayfun for GPUs, no speedup
    %if params.isGPU
    %  ifo_gate = arrayfun(params.nonlinear_gate_f, ifoa_linear(1:3*params.lstmSize, :));
    %  a_signal = arrayfun(params.nonlinear_f, ifoa_linear(3*params.lstmSize+1:4*params.lstmSize, :)); % note input uses a different activation function
    %  c_t = arrayfun(@(f,c,i,a)(f*c + i*a), ifo_gate(params.lstmSize+1:2*params.lstmSize, :), c_t, ifo_gate(1:params.lstmSize, :), a_signal);
    %  %c_t = arrayfun(@cellFun, ifo_gate(params.lstmSize+1:2*params.lstmSize, :), c_t, ifo_gate(1:params.lstmSize, :), a_signal);
    %else
    %end
    %if params.isGPU
    %  %ifo_gate = arrayfun(params.nonlinear_gate_f, ifoa_linear(1:3*params.lstmSize, :));
    %  %a_signal = arrayfun(params.nonlinear_f, ifoa_linear(3*params.lstmSize+1:4*params.lstmSize, :)); % note input uses a different activation function
    %  %c_t = arrayfun(@(f,c,i,a)(f*c + i*a), ifo_gate(params.lstmSize+1:2*params.lstmSize, :), c_t, ifo_gate(1:params.lstmSize, :), a_signal);
    %  c_t = arrayfun(@cell_fun, ifo_gate(params.lstmSize+1:2*params.lstmSize, :), c_t, ifo_gate(1:params.lstmSize, :), a_signal);
    %else
    %end

    %if params.isGPU
    %  do = arrayfun(@(x,y,z) (x*y*z), params.nonlinear_gate_f_prime(lstm{t}.o_gate), lstm{t}.f_c_t, dh);
    %  da = arrayfun(@(x,y,z) (x*y*z), params.nonlinear_f_prime(lstm{t}.a_signal), lstm{t}.i_gate, dc);   
    %else

    %end

    %if params.isGPU % copy to CPU
    %  ifo_gate = gather(ifo_gate);
    %  a_signal = gather(a_signal);
    %  f_c_t = gather(f_c_t);
    %  lstm{t}.c_t = gather(c_t);
    %  lstm{t}.h_t = gather(h_t);
    %else
    %end

      %if params.isGPU % copy to CPU
      %  scores = gather(scores);
      %end

        %if isempty(find(isnan(probs), 1))
        %  assert(abs(sum(sum(probs))-numWords) < 0.001, '! Differ sum(sum(probs)) %g vs. numWords %d\n', sum(sum(probs)), numWords); % normal prob distributions
        %end


