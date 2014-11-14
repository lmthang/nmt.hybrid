function [totalCost, grad] = lstmDecoder(model, input, inputMask, srcMaxLen, params)
%%%
%
% Decode from an LSTM model
%
% Thang Luong @ 2014, <lmthang@stanford.edu>
%
%%%
  
  %dataType = 'double'; % Note: use double precision for grad check
  dataType = 'single';

  curBatchSize = size(input, 1);
  
  %lstm = cell(1, T); % each cell contains intermediate results for that timestep needed for backprop
  input_embs = model.W_emb(:, input);
  if params.isGPU % declare intermediate variables on GPU
    zero_state = zeros([params.lstmSize, curBatchSize], dataType, 'gpuArray');
    input_embs = gpuArray(input_embs); % load input embeddings onto GPUs
  else
    zero_state = zeros([params.lstmSize, curBatchSize]);
  end
  
  W = model.W_src;
  for t=1:srcMaxLen
    % get input embeddings
    %x_t = model.W_emb(:, input(:, t));
    x_t = input_embs(:, ((t-1)*curBatchSize+1):t*curBatchSize);
    if params.isGradCheck 
      x_t(:, ~inputMask(:, t)) = 0; % for gradient check code, zero out those unused so that gradients of those zero-id embeddings can pass.
    end
    
    if t==1
      lstm = lstmUnit(W, x_t, zero_state, zero_state, params);
    else
      lstm = lstmUnit(W, x_t, lstm.h_t, lstm.c_t, params);
    end
  end % for t
  
  % start decoding
  W = model.W_tgt;
  tgtMaxLen = 2*srcMaxLen;
  for t=1:tgtMaxLen
    % prediction
    
  end
  
  if (t>=srcMaxLen) % predict tgtOutput[t-srcMaxLen+1]
    t_pos = t-srcMaxLen+1;
    softmaxMask = tgtMask(:, t_pos); % curBatchSize * 1
    scores = model.W_soft * lstm{t}.h_t(:, softmaxMask);  % params.outVocabSize * num_words

    % normalize, compute in log domain
    mx = max(scores);
    log_probs = bsxfun(@minus, scores, log(sum(exp(bsxfun(@minus, scores, mx)))) + mx); 

    % select from scores matrix, one number per column
    tgt_predicted_words = tgtOutput(softmaxMask, t_pos)';
    num_words = length(tgt_predicted_words);
    score_indices = sub2ind([params.outVocabSize, num_words], tgt_predicted_words, 1:num_words); % 1 * num_words

    % cost
    totalCost = totalCost - sum(log_probs(score_indices));

    if isCostOnly==0 % compute grad
      % grad.W_soft
      probs = exp(log_probs); % out_size * curBatchSize
      probs(score_indices) = probs(score_indices) - ones(1, num_words); % minus one at predicted words
      grad.W_soft = grad.W_soft + probs*lstm{t}.h_t(:, softmaxMask)';

      % grad_ht
      lstm{t}.grad_ht = model.W_soft'* probs;
    end
  end
end