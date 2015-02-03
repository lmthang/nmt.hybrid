function [totalCost, grad] = lstmDecoder(model, input, inputMask, srcMaxLen, params)
%%%
%
% Decode from an LSTM model
%
% Thang Luong @ 2014, <lmthang@stanford.edu>
% Hieu Pham @ 2015, <hyhieu@cs.stanford.edu>
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

  % encode
  lstm = cell(params.numLayers, 1); % lstm can be over written, as we do not need to backprop
  for t=1:srcMaxLen % time
    for ll=1:params.numLayers % layer
      %% encoder W matrix
      W = model.W_src{ll};
      
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
        mask = inputMask(:, t)'; % curBatchSize * 1
        unmaskedIds = find(timeInfo{t}.mask);
        maskedIds = find(~timeInfo{t}.mask);

        % masking
        x_t(:, maskedIds) = 0; 
        h_t_1(:, maskedIds) = 0;
        c_t_1(:, maskedIds) = 0;
      else % subsequent layer, use the previous-layer hidden state
        x_t = lstm{ll-1}.h_t;

        % assert on masking assumptions
        if params.assert
          assert(gather(sum(sum(x_t(:, timeInfo{t}.maskedIds)))) == 0);
          assert(gather(sum(sum(h_t_1(:, timeInfo{t}.maskedIds)))) == 0);
          assert(gather(sum(sum(c_t_1(:, timeInfo{t}.maskedIds)))) == 0);
        end
      end
     
      %% lstm cell
      lstm{ll} = lstmUnit(W, x_t, h_t_1, c_t_1, params);

%       %% prediction at the top layer
%       if ll==params.numLayers && (t>=srcMaxLen) 
%         % softmax
%         [probs, scores, norms] = softmax(model.W_soft, lstm{ll, t}.h_t, timeInfo{t}.mask);
%         if params.assert
%           value = sum(sum(abs(scores(:, timeInfo{t}.maskedIds))));
%           assert(gather(value)==0);
%         end
% 
%         % cost
%         tgtPredictedWords = tgtOutput(timeInfo{t}.unmaskedIds, t-srcMaxLen+1)'; % predict tgtOutput[t-srcMaxLen+1]
%         scoreIndices = sub2ind([params.outVocabSize, curBatchSize], tgtPredictedWords, timeInfo{t}.unmaskedIds); % 1 * length(tgtPredictedWords)
%         totalCost = totalCost - (sum(scores(scoreIndices)) - sum(log(norms).*timeInfo{t}.mask));
% 
%         if isCostOnly==0 % compute grad
%           % grad.W_soft
%           probs(scoreIndices) = probs(scoreIndices) - 1; % minus one at predicted words
%           grad.W_soft = grad.W_soft + probs*lstm{ll, t}.h_t';
% 
%           % grad_ht
%           lstm{ll, t}.grad_ht = model.W_soft'* probs;
%         end
%       end
    end
  end
  
  % start decoding
  W = model.W_tgt;
  tgtMaxLen = 2*srcMaxLen;
  for t=1:tgtMaxLen
    % prediction
    
  end
end
