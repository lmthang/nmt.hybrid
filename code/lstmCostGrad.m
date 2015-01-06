function [totalCost, grad] = lstmCostGrad(model, trainData, params, isCostOnly)
%%%
%
% Compute cost/grad for LSTM.
% If isCostOnly==1, this method only computes cost (for testing purposes).
%
% Thang Luong @ 2014, <lmthang@stanford.edu>
%
%%%

  input = trainData.input;
  inputMask = trainData.inputMask;
  tgtOutput = trainData.tgtOutput;
  %tgtMask = trainData.tgtMask;
  srcMaxLen = trainData.srcMaxLen;
  tgtMaxLen = trainData.tgtMaxLen;
  
  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  T = srcMaxLen+tgtMaxLen-1;
  curBatchSize = size(input, 1);
  
  % grad that can be computed as we do the forward pass
  lstm = cell(params.numLayers, T); % each cell contains intermediate results for that timestep needed for backprop
  input_embs = model.W_emb(:, input);
  
  % global opt
  if params.globalOpt==1
    srcSentEmbs = sum(reshape(input_embs(:, 1:curBatchSize*srcMaxLen), params.lstmSize*curBatchSize, srcMaxLen), 2); % sum
    srcSentEmbs = bsxfun(@rdivide, reshape(srcSentEmbs, params.lstmSize, curBatchSize), trainData.srcLens');
  end
  
  grad.W_emb = sparse(params.lstmSize, params.inVocabSize); % live on CPU
  if params.isGPU % declare intermediate variables on GPU
    zero_state = zeros([params.lstmSize, curBatchSize], params.dataType, 'gpuArray');
    input_embs = gpuArray(input_embs); % load input embeddings onto GPUs
    
    totalCost = zeros(1, 1, params.dataType, 'gpuArray');

    % grad
    grad.W_soft = zeros(params.outVocabSize, params.lstmSize, params.dataType, 'gpuArray');
    
    % W_src
    if params.isBi
      grad.W_src = cell(params.numLayers, 1);
      for ll=1:params.numLayers
        grad.W_src{ll} = zeros(4*params.lstmSize, 2*params.lstmSize, params.dataType, 'gpuArray');
      end
    end
    
    % W_tgt
    grad.W_tgt = cell(params.numLayers, 1);
    for ll=1:params.numLayers
      grad.W_tgt{ll} = zeros(4*params.lstmSize, 2*params.lstmSize, params.dataType, 'gpuArray');
    end
  else
    zero_state = zeros(params.lstmSize, curBatchSize);
    totalCost = 0.0;
    
    % grad 
    grad.W_soft = zeros(params.outVocabSize, params.lstmSize);
    
    % W_src
    if params.isBi
      grad.W_src = cell(params.numLayers, 1);
      for ll=1:params.numLayers
        grad.W_src{ll} = zeros(4*params.lstmSize, 2*params.lstmSize);
      end
    end
    
    % W_tgt
    grad.W_tgt = cell(params.numLayers, 1);
    for ll=1:params.numLayers
      grad.W_tgt{ll} = zeros(4*params.lstmSize, 2*params.lstmSize);
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
      
      %% input
      if ll==1 % first layer, get input embeddings
        x_t = input_embs(:, ((t-1)*curBatchSize+1):t*curBatchSize);
        x_t(:, ~inputMask(:, t)) = 0; % zero out those zero-id embeddings
      else % subsequent layer, use the hidden state from the previous layer
        x_t = lstm{ll-1, t}.h_t;
      end

      %% lstm cell
      if t==1
        lstm{ll, t} = lstmUnit(W, x_t, zero_state, zero_state, params);
      else
        lstm{ll, t} = lstmUnit(W, x_t, lstm{ll, t-1}.h_t, lstm{ll, t-1}.c_t, params);
      end

      %% prediction at the top layer
      if ll==params.numLayers && (t>=srcMaxLen) 
        softmaxMask = inputMask(:, t); % curBatchSize * 1
        scores = model.W_soft * lstm{ll, t}.h_t(:, softmaxMask);  % params.outVocabSize * num_words

        % normalize, compute in log domain
        mx = max(scores);
        log_probs = bsxfun(@minus, scores, log(sum(exp(bsxfun(@minus, scores, mx)))) + mx); 

        % predict tgtOutput[t-srcMaxLen+1]
        % select from scores matrix, one number per column
        tgt_predicted_words = tgtOutput(softmaxMask, t-srcMaxLen+1)';
        num_words = length(tgt_predicted_words);
        score_indices = sub2ind([params.outVocabSize, num_words], tgt_predicted_words, 1:num_words); % 1 * num_words

        % cost
        totalCost = totalCost - sum(log_probs(score_indices));

        if isCostOnly==0 % compute grad
          % grad.W_soft
          probs = exp(log_probs); % out_size * curBatchSize
          probs(score_indices) = probs(score_indices) - ones(1, num_words); % minus one at predicted words
          grad.W_soft = grad.W_soft + probs*lstm{ll, t}.h_t(:, softmaxMask)';

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
  % intermediate variables
  
  % h_t and c_t gradients accumulate over time per layer
  dh = cell(params.numLayers, 1);
  dc = cell(params.numLayers, 1); 
  for ll=params.numLayers:-1:1 % layer
    dh{ll} = zero_state;
    dc{ll} = zero_state;
  end
  
  for t=T:-1:1 % time
    mask = inputMask(:, t);
    
    for ll=params.numLayers:-1:1 % layer
      %% hidden state grad
      if ll==params.numLayers && (t>=srcMaxLen) % get signals from the softmax layer
        dh{ll}(:, mask) = dh{ll}(:, mask) + lstm{ll, t}.grad_ht; % accumulate grads wrt the hidden layer
      end

      
      %% cell back prob
      if ll==1 % first layer, get input embeddings
        x_t = input_embs(:, ((t-1)*curBatchSize+1):t*curBatchSize);
      else % subsequent layer, use the hidden state from the previous layer
        x_t = lstm{ll-1, t}.h_t;
      end
      [dc{ll}, dh{ll}, lstm_grad] = lstmUnitGrad(model, lstm, x_t, dc{ll}, dh{ll}, ll, t, mask, srcMaxLen, zero_state, params);

      %% grad.W_src / grad.W_tgt
      if (t>=srcMaxLen)
        grad.W_tgt{ll} = grad.W_tgt{ll} + lstm_grad.W_tgt;
      else
        grad.W_src{ll} = grad.W_src{ll} + lstm_grad.W_src;
      end

      %% input grad
      if ll==1 % collect embedding grad
        indices = input(mask, t);
        if params.isGPU
          emb_grad = double(gather(lstm_grad.dx(:, mask))); % copy embedding grads to CPU
        else
          emb_grad = lstm_grad.dx(:, mask);
        end

        grad.W_emb = grad.W_emb + aggregateMatrix(emb_grad, indices, params.inVocabSize);
      else % pass down hidden state grad to the below layer
        dh{ll-1}(:, mask) = dh{ll-1}(:, mask) + lstm_grad.dx(:, mask);
      end
    end % end for layer
  end % end for time
  
  if params.isGPU % copy to CPU
    grad.W_soft = gather(grad.W_soft);
    if params.isBi
      for ll=1:params.numLayers
        grad.W_src{ll} = gather(grad.W_src{ll});
      end
    end
    
    for ll=1:params.numLayers
      grad.W_tgt{ll} = gather(grad.W_tgt{ll});
    end
    totalCost = gather(totalCost);
  end
end

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
        %  assert(abs(sum(sum(probs))-num_words) < 0.001, '! Differ sum(sum(probs)) %g vs. num_words %d\n', sum(sum(probs)), num_words); % normal prob distributions
        %end


