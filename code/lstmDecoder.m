function [candidates, scores] = lstmDecoder(model, input, inputMask, srcMaxLen, params)
%%%
%
% Decode from an LSTM model
%
% Thang Luong @ 2014, <lmthang@stanford.edu>
% Hieu Pham @ 2015, <hyhieu@cs.stanford.edu>
%
%%%

  beamSize = 12; % perhaps this should be a parameter
  
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
    % prepare mask
    mask = inputMask(:, t)'; % curBatchSize * 1
    unmaskedIds = find(mask);
    maskedIds = find(~mask);

    for ll=1:params.numLayers % layer
      %% encoder W matrix
      W = model.W_src{ll};
      
      %% previous-time input
      if t==1 % first time step
        h_t_1 = zero_state;
        c_t_1 = zero_state;
      else
        h_t_1 = lstm{ll}.h_t; 
        c_t_1 = lstm{ll}.c_t;
      end

      %% current-time input
      if ll==1 % first layer
        x_t = model.W_emb(:, input(:, t));

        % masking
        x_t(:, maskedIds) = 0; 
        h_t_1(:, maskedIds) = 0;
        c_t_1(:, maskedIds) = 0;
      else % subsequent layer, use the previous-layer hidden state
        x_t = lstm{ll-1}.h_t;
      end
     
      %% lstm cell
      lstm{ll} = lstmUnit(W, x_t, h_t_1, c_t_1, params);
    end
  end
  
  % start decoding
  W = model.W_tgt;
  tgtMaxLen = 2*srcMaxLen;
  num = cell(curBatchSize, 1);
  candidates = cell(curBatchSize, 1);
  scores = cell(curBatchSize, 1);
  for currSent = 1:curBatchSize
    curr_lstm.h_t = lstm{ll}.h_t(:, currSent);
    curr_lstm.c_t = lstm{ll}.c_t(:, currSent);

    [num{i}, candidates{i}, scores{i}] = decodeOneSent(model, params, curr_lstm, tgtMaxLen, beamSize);
  end
end

%%%
%
% Beam decoder from an LSTM model, works for only one sentence
%
% Input:
%   - encoded vector of the source sentence
%   - maximum length willing to go
%   - beam size
%
% Output:
%   - candidates: list of candidates
%   - scores: score of the corresponding candidates
%
% Thang Luong @ 2014, <lmthang@stanford.edu>
% Hieu Pham @ 2015, <hyhieu@cs.stanford.edu>
%
%%%
function [candidates, scores] = decodeOneSent(model, params, lstmStart, maxLen, beamSize)
  W_soft = model.W_soft;
  W_emb = model.W_emb;

  num_decoded = 0;
  candidates = cell(0);
  scores = cell(0);

  [bestProbs, bestWords] = beamStep(model, params, lstmStart.h_t, beamSize);
  
  beam.probs = bestProbs;
  beam.history = cell(beamSize, 1);
  beam.lstmState = cell(beamSize, 1);
  for i = 1:beamSize
    beam.history{i} = bestWords(i);
    beam.lstmState{i} = lstmStart;
  end

  
  for T = 1:maxLen
    % init probs
    allBestProbs = zeros(beamSize*beamSize, 1);
    allBestWords = zeros(beamSize*beamSize, 1);

    % compute next probs
    for j = 1:beamSize % this part can be vectorized
      if beam.probs(j) == 0
        continue;
      end
      x_t = W_emb(:, beam.history{i}(end));
      h_t = beam.lstmState{j}.h_t;
      c_t = beam.lstmstate{j}.c_t;

      [bestProbs, bestWords] = beamStep(model, params, h_t, beamSize);
      allBestProbs((j-1)*beamSize+1:j*beamSize) = bestProbs * beam.probs(j);
      allBestWords((j-1)*beamsize+1:j*beamSize) = bestWords;
    end

    % take the best
    [bestVals, bestWords] = sort(allBestProbs, 'descend');
    bestVals = bestVals(1:beamSize);
    bestWords = bestWords(1:beamSize);

    % update beam
    new_beam.probs = zeros(12, 1);
    new_beam.history = cell(beamSize, 1);
    new_beam.lstmState = cell(beamSize, 1);
    for j = 1:beamSize
      val = bestVals(j);
      idx = floor((bestWords(j) - 1) / beamSize) + 1;

      x_t = W_emb(:, bestWords(j));
      h_t = beam.lstmState{idx}.h_t;
      c_t = beam.lstmState{idx}.c_t;

      new_beam.probs(j) = val;
      new_beam.history{j} = [beam.history{idx}, bestWords(j)];
      new_beam.lstmState{j} = lstmUnit(W_emb, x_t, h_t, c_t, params);

      % if we see tgtEos, then we:
      %  - take that finished beam to candidate list
      %  - stop the beam by setting its probs attribute to 0
      if bestWords(j) == params.tgtEos
        num_decoded = num_decoded + 1;
        candidates{num_decoded} = new_beam.history{j};
        scores{num_decoded} = new_beam.probs;

        new_beam.probs(j) = 0;
      end
    end

    % copy new_beam to beam
    beam.probs = new_beam.probs;
    beam.history = new_beam.history;
    beam.lstmState = new_beam.lstmState;
  end
end

function [bestProbs, bestWords] = beamStep(model, params, h, beamSize)
  [probs, ~, ~] = softmax(model.W_soft, h);
  [sortedProbs, sortedWords] = sort(probs(:), 'descend');
  bestWords = sortedWords(1:beamSize);
  bestProbs = probs(bestWords);
end
