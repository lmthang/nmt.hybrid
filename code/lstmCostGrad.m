function [totalCost, grad] = lstmCostGrad(model, trainData, params, isCostOnly)
%%%
%
% Compute cost/grad for LSTM.
% If isCostOnly==1, this method only computes cost (for testing purposes).
%
% Thang Luong @ 2014, <lmthang@stanford.edu>
%
%%%

  %%%%%%%%%%%%
  %%% INIT %%%
  %%%%%%%%%%%%
  input = trainData.input;
  inputMask = trainData.inputMask;
  tgtOutput = trainData.tgtOutput;
  srcMaxLen = trainData.srcMaxLen;
  tgtMaxLen = trainData.tgtMaxLen;
  
  T = srcMaxLen+tgtMaxLen-1;
  curBatchSize = size(input, 1);
  
  numInputWords = sum(sum(inputMask));
  indices = zeros(numInputWords, 1);
  [grad, zero_state, totalCost, emb] = initGrad(params, curBatchSize, numInputWords);

  
  % global opt
  %if params.globalOpt==1
  %  srcSentEmbs = sum(reshape(input_embs(:, 1:curBatchSize*srcMaxLen), params.lstmSize*curBatchSize, srcMaxLen), 2); % sum
  %  srcSentEmbs = bsxfun(@rdivide, reshape(srcSentEmbs, params.lstmSize, curBatchSize), trainData.srcLens');
  %end

  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  timeInfo = cell(T, 1);
  lstm = cell(params.numLayers, T); % each cell contains intermediate results for that timestep needed for backprop
  
  % attention mechanism
  if params.attnOpt>0
    if params.isGPU
      srcAlignStates = zeros([params.lstmSize, curBatchSize, srcMaxLen-1], params.dataType, 'gpuArray');
    else
      srcAlignStates = zeros([params.lstmSize, curBatchSize, srcMaxLen-1]);
    end
  end
  
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
        h_t_1 = zero_state;
        c_t_1 = zero_state;
      else
        h_t_1 = lstm{ll, t-1}.h_t; 
        c_t_1 = lstm{ll, t-1}.c_t;
      end

      %% current-time input
      if ll==1 % first layer
        x_t = model.W_emb(:, input(:, t));

        % prepare mask
        timeInfo{t}.mask = inputMask(:, t)'; % curBatchSize * 1
        timeInfo{t}.unmaskedIds = find(timeInfo{t}.mask);
        timeInfo{t}.maskedIds = find(~timeInfo{t}.mask);
      else % subsequent layer, use the previous-layer hidden state
        x_t = lstm{ll-1, t}.h_t;
      end
      
      % masking
      x_t(:, timeInfo{t}.maskedIds) = 0; 
      h_t_1(:, timeInfo{t}.maskedIds) = 0;
      c_t_1(:, timeInfo{t}.maskedIds) = 0;
     
      %% lstm cell
      lstm{ll, t} = lstmUnit(W, x_t, h_t_1, c_t_1, params);
      
      %% attention mechanism: keep track of src hidden states at the top level
      if params.attnOpt==1 && ll==params.numLayers && (t<srcMaxLen)
        srcAlignStates(:, :, t) = model.W_a * lstm{ll, t}.h_t; % W_a * src_h
      end
        
      %% prediction at the top layer
      if ll==params.numLayers && (t>=srcMaxLen)
        % attention mechanism
        if params.attnOpt==1 % compute alignment vector after the current hidden state has been computed
          [alignWeights] = computeAlignWeights(lstm{ll, t}.h_t, model, srcAlignStates, params, curBatchSize, srcMaxLen);
        end
        
        % softmax
        [probs, scores, norms] = softmax(model.W_soft, lstm{ll, t}.h_t, timeInfo{t}.mask);
        if params.assert
          if params.isGPU
            assert(gather(sum(sum(abs(scores(:, timeInfo{t}.maskedIds)))))==0);
          else
            assert(sum(sum(abs(scores(:, timeInfo{t}.maskedIds))))==0);
          end
        end

        % cost
        tgtPredictedWords = tgtOutput(timeInfo{t}.unmaskedIds, t-srcMaxLen+1)'; % predict tgtOutput[t-srcMaxLen+1]
        scoreIndices = sub2ind([params.outVocabSize, curBatchSize], tgtPredictedWords, timeInfo{t}.unmaskedIds); % 1 * length(tgtPredictedWords)
        totalCost = totalCost - (sum(scores(scoreIndices)) - sum(log(norms).*timeInfo{t}.mask));

        if isCostOnly==0 % compute grad
          % grad.W_soft
          probs(scoreIndices) = probs(scoreIndices) - 1; % minus one at predicted words
          grad.W_soft = grad.W_soft + probs*lstm{ll, t}.h_t';

          % grad_ht
          lstm{ll, t}.grad_ht = model.W_soft'* probs;
        end
      end
    end
  end
  
  if isCostOnly==1 % don't compute grad
    return;
  end
  
  %%%%%%%%%%%%%%%%%%%%%
  %%% BACKWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%%
  % h_t and c_t gradients accumulate over time per layer
  dh = cell(params.numLayers, 1);
  dc = cell(params.numLayers, 1); 
  for ll=params.numLayers:-1:1 % layer
    dh{ll} = zero_state;
    dc{ll} = zero_state;
  end
  
  wordCount = 0;
  for t=T:-1:1 % time
    unmaskedIds = timeInfo{t}.unmaskedIds;

    for ll=params.numLayers:-1:1 % layer
      %% hidden state grad
      if ll==params.numLayers && (t>=srcMaxLen) % get signals from the softmax layer
        dh{ll} = dh{ll} + lstm{ll, t}.grad_ht; % accumulate grads wrt the hidden layer 
      end

      %% cell backprop
      [lstm_grad] = lstmUnitGrad(model, lstm, dc{ll}, dh{ll}, ll, t, srcMaxLen, zero_state, params);
      dc{ll} = lstm_grad.dc;
      dh{ll} = lstm_grad.d_xh(params.lstmSize+1:end, :);

      %% grad.W_src / grad.W_tgt
      if (t>=srcMaxLen)
        grad.W_tgt{ll} = grad.W_tgt{ll} + lstm_grad.W;
      else
        grad.W_src{ll} = grad.W_src{ll} + lstm_grad.W;
      end

      %% input grad
      if ll==1 % collect embedding grad
        numWords = length(unmaskedIds);
        indices(wordCount+1:wordCount+numWords) = input(unmaskedIds, t);
        emb(:, wordCount+1:wordCount+numWords) = lstm_grad.d_xh(1:params.lstmSize, unmaskedIds);
        wordCount = wordCount + numWords;
      else % pass down hidden state grad to the below layer
        dh{ll-1}(:, unmaskedIds) = dh{ll-1}(:, unmaskedIds) + lstm_grad.d_xh(1:params.lstmSize, unmaskedIds);
      end
    end % end for layer
  end % end for time
   
  % grad W_emb
  [grad.W_emb, grad.indices] = aggregateMatrix(emb, indices, params.isGPU, params.dataType);
  if params.isGPU
    totalCost = gather(totalCost);
  end
end

function [alignWeights] = computeAlignWeights(h_t, model, srcAlignStates, params, curBatchSize, srcMaxLen)
  if params.attnFunc==1 % tgt_h' * W_a * src_h
    alignWeights = squeeze(sum(bsxfun(@times, srcAlignStates, h_t))); % curBatchSize * (srcMaxLen-1)

    % assert
    if params.assert
      results = zeros(curBatchSize, srcMaxLen-1);
      for iii=1:(srcMaxLen-1)
        results(:, iii) = transpose(sum(srcAlignStates(:, :, iii).*h_t));
      end
      assert(sum(sum(abs(alignWeights-results)))<1e-5);
    end
  elseif params.attnFunc==2 % v_a' * tanh(W_a_tgt * tgt_h +  W_a * src_h)
    alignWeights = tanh(bsxfun(@plus, srcAlignStates, model.W_a_tgt*h_t)); % lstmSize * curBatchSize * (srcMaxLen-1)
    alignWeights = squeeze(sum(bsxfun(@times, alignWeights, model.v_a))); % curBatchSize * (srcMaxLen-1)

    % assert
    if params.assert
      results = zeros(curBatchSize, srcMaxLen-1);
      tgtAlignState = model.W_a_tgt*h_t;
      for iii=1:(srcMaxLen-1)
        results(:, iii) = transpose(model.v_a'*tanh(srcAlignStates(:, :, iii) + tgtAlignState));
      end
      assert(sum(sum(abs(alignWeights-results)))<1e-5);
    end
  end

  if params.assert % curBatchSize * (srcMaxLen-1)
    assert(size(alignWeights, 1)==curBatchSize);
    assert(size(alignWeights, 2)==(srcMaxLen-1));
  end
end
function [grad, zero_state, totalCost, emb] = initGrad(params, curBatchSize, numInputWords)
  if params.isBi
    grad.W_src = cell(params.numLayers, 1);
  end
  grad.W_tgt = cell(params.numLayers, 1);
 
  if params.isGPU % declare intermediate variables on GPU
    zero_state = zeros([params.lstmSize, curBatchSize], params.dataType, 'gpuArray');
   
    totalCost = zeros(1, 1, params.dataType, 'gpuArray');

    % grad
    grad.W_soft = zeros(params.outVocabSize, params.lstmSize, params.dataType, 'gpuArray');
    
    % W_src
    if params.isBi
      for ll=1:params.numLayers
        grad.W_src{ll} = zeros(4*params.lstmSize, 2*params.lstmSize, params.dataType, 'gpuArray');
      end
    end
    
    % W_tgt
    for ll=1:params.numLayers
      grad.W_tgt{ll} = zeros(4*params.lstmSize, 2*params.lstmSize, params.dataType, 'gpuArray');
    end
    
    emb = zeros(params.lstmSize, numInputWords, params.dataType, 'gpuArray');
  else
    zero_state = zeros(params.lstmSize, curBatchSize);
    totalCost = 0.0;
    
    % grad 
    grad.W_soft = zeros(params.outVocabSize, params.lstmSize);
    
    % W_src
    if params.isBi
      for ll=1:params.numLayers
        grad.W_src{ll} = zeros(4*params.lstmSize, 2*params.lstmSize);
      end
    end
    
    % W_tgt
    for ll=1:params.numLayers
      grad.W_tgt{ll} = zeros(4*params.lstmSize, 2*params.lstmSize);
    end
  
    emb = zeros(params.lstmSize, numInputWords);
  end
end

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

  %input_embs = model.W_emb(:, input);
    %input_embs = gpuArray(input_embs); % load input embeddings onto GPUs
 
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


