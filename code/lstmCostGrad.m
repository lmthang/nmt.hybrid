function [totalCost, grad] = lstmCostGrad(model, input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen, params, isCostOnly)
%%%
%
% Compute cost/grad for LSTM.
% If isCostOnly==1, this method only computes cost (for testing purposes).
%
% Thang Luong @ 2014, <lmthang@stanford.edu>
%
%%%
  
  %dataType = 'double'; % Note: use double precision for grad check
  dataType = 'single';

  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  T = srcMaxLen+tgtMaxLen-1;
  curBatchSize = size(input, 1);
  
  % grad that can be computed as we do the forward pass
  lstm = cell(1, T); % each cell contains intermediate results for that timestep needed for backprop
  input_embs = model.W_emb(:, input);
  grad.W_emb = sparse(params.lstmSize, params.inVocabSize); % live on CPU
  if params.isGPU % declare intermediate variables on GPU
    h_t = zeros([params.lstmSize, curBatchSize], dataType, 'gpuArray');
    c_t = zeros([params.lstmSize, curBatchSize], dataType, 'gpuArray');
    input_embs = gpuArray(input_embs); % load input embeddings onto GPUs
    
    totalCost = zeros(1, 1, dataType, 'gpuArray');

    % grad
    grad.W_soft = zeros(params.outVocabSize, params.lstmSize, dataType, 'gpuArray');
    grad.W_src = zeros(4*params.lstmSize, 2*params.lstmSize, dataType, 'gpuArray');
    grad.W_tgt = zeros(4*params.lstmSize, 2*params.lstmSize, dataType, 'gpuArray');
  else
    h_t = zeros([params.lstmSize, curBatchSize]);
    c_t = zeros([params.lstmSize, curBatchSize]);
    totalCost = 0.0;
    
    % grad 
    grad.W_soft = zeros(params.outVocabSize, params.lstmSize);
    grad.W_src = zeros(4*params.lstmSize, 2*params.lstmSize);
    grad.W_tgt = zeros(4*params.lstmSize, 2*params.lstmSize);
  end
  
  for t=1:T
    % get input embeddings
    %x_t = model.W_emb(:, input(:, t));
    x_t = input_embs(:, ((t-1)*curBatchSize+1):t*curBatchSize);
    if params.isGradCheck 
      x_t(:, ~inputMask(:, t)) = 0; % for gradient check code, zero out those unused so that gradients of those zero-id embeddings can pass.
    end

    if (t>=srcMaxLen) % start decoding
      W = model.W_tgt;
    else
      W = model.W_src;
    end
    
    %% input, forget, output gates and input signals before applying non-linear functions
    ifoa_linear = W*[x_t; h_t];    
    
    %% cell
    % GPU note: the below non-linear functions are fast, so no need to use arrayfun
    ifo_gate = params.nonlinear_gate_f(ifoa_linear(1:3*params.lstmSize, :));
    lstm{t}.i_gate = ifo_gate(1:params.lstmSize, :);
    lstm{t}.f_gate = ifo_gate(params.lstmSize+1:2*params.lstmSize, :);
    lstm{t}.o_gate = ifo_gate(2*params.lstmSize+1:3*params.lstmSize, :);
    lstm{t}.a_signal = params.nonlinear_f(ifoa_linear(3*params.lstmSize+1:4*params.lstmSize, :)); % note input uses a different activation function
    c_t = lstm{t}.f_gate.*c_t + lstm{t}.i_gate.*lstm{t}.a_signal; % c_t = f_t * c_{t-1} + i_t * a_t
    
    %% hidden
    lstm{t}.f_c_t = params.nonlinear_f(c_t);
    h_t = lstm{t}.o_gate.*lstm{t}.f_c_t; % h_t = o_t * g(c_t)
    
    % clip
    if params.isClip
      if params.isGPU
       c_t = arrayfun(@clipForward, c_t);
       h_t = arrayfun(@clipForward, h_t);
      else
       c_t(c_t>params.clipForward) = params.clipForward; c_t(c_t<-params.clipForward) = -params.clipForward; % clip: keep memory small
       h_t(h_t>params.clipForward) = params.clipForward; h_t(h_t<-params.clipForward) = -params.clipForward; % clip: keep hidden state small
      end
    end
    
    %% save state
    lstm{t}.c_t = c_t;
    lstm{t}.h_t = h_t;
    
    if (t>=srcMaxLen) % predict tgtOutput[t-srcMaxLen+1]
      t_pos = t-srcMaxLen+1;
      softmaxMask = tgtMask(:, t_pos); % curBatchSize * 1
      tgt_predicted_words = tgtOutput(softmaxMask, t_pos)';
      num_words = length(tgt_predicted_words);
      
      scores = model.W_soft * h_t(:, softmaxMask);  % params.outVocabSize * num_words
      mx = max(scores);
      log_probs = bsxfun(@minus, scores, log(sum(exp(bsxfun(@minus, scores, mx)))) + mx); 
      %log_probs = bsxfun(@minus, scores, simpleLogSumExp(scores));
      %log_probs = bsxfun(@minus, scores, logsumexp(scores));
      
      % select from scores matrix, one number per column
      % TODO: optimize this code
      score_indices = sub2ind([params.outVocabSize, num_words], tgt_predicted_words, 1:num_words); % 1 * num_words
      
      % cost
      totalCost = totalCost - sum(log_probs(score_indices));
      
      if isCostOnly==0 % compute grad
        % grad.W_soft
        probs = exp(log_probs); % out_size * curBatchSize
        probs(score_indices) = probs(score_indices) - ones(1, num_words); % minus one at predicted words
        grad.W_soft = grad.W_soft + probs*h_t(:, softmaxMask)';

        % grad_ht
        lstm{t}.grad_ht = model.W_soft'* probs;
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
  if params.isGPU
    dh = zeros(params.lstmSize, curBatchSize, dataType, 'gpuArray');
    dc = zeros(params.lstmSize, curBatchSize, dataType, 'gpuArray');
  else
    dh = zeros(params.lstmSize, curBatchSize);
    dc = zeros(params.lstmSize, curBatchSize);
  end
  for t=T:-1:1
    if (t>=srcMaxLen) % predict tgtOutput[t-srcMaxLen+1]
      t_pos = t-srcMaxLen+1;
      softmaxMask = tgtMask(:, t_pos); % curBatchSize * 1
      dh(:, softmaxMask) = dh(:, softmaxMask) + lstm{t}.grad_ht; % accumulate grads wrt the hidden layer
    end
    
    dc = dc + params.nonlinear_f_prime(lstm{t}.f_c_t).*lstm{t}.o_gate.*dh;
    
    di = params.nonlinear_gate_f_prime(lstm{t}.i_gate).*lstm{t}.a_signal.*dc;
    if t>1
      df = params.nonlinear_gate_f_prime(lstm{t}.f_gate).*lstm{t-1}.c_t.*dc;
    else
      if params.isGPU
        df = zeros(params.lstmSize, curBatchSize, dataType, 'gpuArray');
      else
        df = zeros(params.lstmSize, curBatchSize);
      end
    end

    % arrayfun doesn't work here for GPU
    do = params.nonlinear_gate_f_prime(lstm{t}.o_gate).*lstm{t}.f_c_t .* dh;
    da = params.nonlinear_f_prime(lstm{t}.a_signal).*lstm{t}.i_gate.*dc;   

    %x_t = model.W_emb(:, input(:, t));
    x_t = input_embs(:, ((t-1)*curBatchSize+1):t*curBatchSize);
    if (t>=srcMaxLen) % grad tgt
      W = model.W_tgt;
      if t==1
        grad.W_tgt = grad.W_tgt + [di; df; do; da]*[x_t; zeros(params.lstmSize, curBatchSize)]';
      else
        grad.W_tgt = grad.W_tgt + [di; df; do; da]*[x_t; lstm{t-1}.h_t]';
      end
    else % grad src
      W = model.W_src;
      if t==1
        grad.W_src = grad.W_src + [di; df; do; da]*[x_t; zeros(params.lstmSize, curBatchSize)]';
      else
        grad.W_src = grad.W_src + [di; df; do; da]*[x_t; lstm{t-1}.h_t]';
      end
    end
    
    dx = W(:, 1:params.lstmSize)'*[di; df; do; da];
    dh = W(:, params.lstmSize+1:end)'*[di; df; do; da];
    dc = lstm{t}.f_gate.*dc;
    
    % clip hidden/cell derivatives
    if params.isClip
      if params.isGPU
       dh = arrayfun(@clipBackward, dh);
       dc = arrayfun(@clipBackward, dc);
      else
       dh(dh>params.clipBackward) = params.clipBackward; dh(dh<-params.clipBackward) = -params.clipBackward;
       dc(dc>params.clipBackward) = params.clipBackward; dc(dc<-params.clipBackward) = -params.clipBackward;
      end
    end
    
    % update embeddings
    embMask = inputMask(:, t);
    indices = input(embMask, t);
    if params.isGPU
      emb_grad = double(gather(dx(:, embMask))); % copy embedding grads to CPU
    else
      emb_grad = dx(:, embMask);
    end
    
    grad.W_emb = grad.W_emb + aggregateMatrix(emb_grad, indices, params.inVocabSize);
  end

  if params.isGPU % copy to CPU
    grad.W_soft = gather(grad.W_soft);
    grad.W_src = gather(grad.W_src);
    grad.W_tgt = gather(grad.W_tgt);
    totalCost = gather(totalCost);
  end
end

function [clippedValue] = clipForward(x)
  if x>50
    clippedValue = single(50);
  elseif x<-50
    clippedValue = single(-50);
  else
    clippedValue = x;
  end
end

function [clippedValue] = clipBackward(x)
  if x>1000
    clippedValue = single(1000);
  elseif x<-1000
    clippedValue = single(-1000);
  else
    clippedValue = x;
  end
end

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


