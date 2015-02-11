function [candidates, scores] = lstmDecoder(model, input, inputMask, srcMaxLen, params, beamSize)
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

      else % subsequent layer, use the previous-layer hidden state
        x_t = lstm{ll-1}.h_t;
      end
     
      % masking
      x_t(:, maskedIds) = 0; 
      h_t_1(:, maskedIds) = 0;
      c_t_1(:, maskedIds) = 0;

      %% lstm cell
      lstm{ll} = lstmUnit(W, x_t, h_t_1, c_t_1, params);
    end
  end
  
  % start decoding
  candidates = cell(1);
  scores     = cell(1);
  for currSent = 1:1
    if mod(currSent, 20) == 0
      fprintf('Decoded %d sentences.\n', currSent);
    end
    curr_lstm = cell(params.numLayers, 1);
    for ll = 1:params.numLayers
      curr_lstm{ll}.h_t = lstm{ll}.h_t(:, currSent);
      curr_lstm{ll}.c_t = lstm{ll}.c_t(:, currSent);
    end

    [candidates{currSent}, scores{currSent}] = decode_one_sent(model, params, curr_lstm, sum(inputMask(currSent,:)), beamSize);
    % floor(1.5*sum(inputMask(currSent,:)))
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
function [candidates, scores] = decode_one_sent(model, params, lstm_start, max_len, beam_size)
  W_tgt = model.W_tgt;
  W_emb = model.W_emb;
  num_layers = params.numLayers;

  num_decoded = 0;
  candidates = cell(0);
  scores = cell(0);

  beam.probs = cell(beam_size, 1);
  beam.hists = cell(beam_size, 1);
  beam.lstms = cell(beam_size, 1);

  [best_probs, best_words] = next_beam_step(model, params, lstm_start{num_layers}.h_t, beam_size);
  for bb = 1 : beam_size
    beam.probs{bb} = best_probs(bb);
    beam.hists{bb} = best_words(bb);
    beam.lstms{bb} = lstm_start;
  end

  show_beam(beam, beam_size, params);
  for sent_pos = 1 : max_len
    all_best_probs = zeros(beam_size*beam_size, 1);
    all_best_words = zeros(beam_size*beam_size, 1);

    for bb = 1 : beam_size
      if beam.probs{bb} == 0
        continue;
      end

      % accumulate the last word in history
      % to compute all lstm layers
      last_word = beam.hists{bb}(end);
      lstm = cell(num_layers, 1);

      for ll = 1 : num_layers
        if ll == 1
          x_t = W_emb(:, last_word);
        else
          x_t = lstm{ll-1}.h_t;
        end

        h_t = beam.lstms{bb}{ll}.h_t;
        c_t = beam.lstms{bb}{ll}.c_t;

        lstm{ll} = lstmUnit(W_tgt{ll}, x_t, h_t, c_t, params);
      end

      beam.lstms{bb} = lstm;

      % predict the next word
      [best_next_probs, best_next_words] = next_beam_step(model, params, lstm{num_layers}.h_t, beam_size);
      all_best_probs(((bb-1)*beam_size+1) : (bb*beam_size)) = best_next_probs + beam.probs{bb};
      all_best_words(((bb-1)*beam_size+1) : (bb*beam_size)) = best_next_words;
    end

    non_zeros = find(all_best_probs);
    all_best_probs = all_best_probs(non_zeros);
    all_best_words = all_best_words(non_zeros);

    [sorted_best_probs, indices] = sort(all_best_probs, 'descend');
    probs = sorted_best_probs(1 : beam_size);
    words = all_best_words(indices(1 : beam_size));

    % update beam
    new_beam.probs = cell(beam_size, 1);
    new_beam.hists = cell(beam_size, 1);
    new_beam.lstms = cell(beam_size, 1);

    for bb = 1 : beam_size
      last_beam_idx = floor((indices(bb)-1) / beam_size + 1e-9) + 1;
      new_beam.probs{bb} = probs(bb);
      new_beam.hists{bb} = [beam.hists{last_beam_idx}, words(bb)];
      new_beam.lstms{bb} = beam.lstms{last_beam_idx};

      if words(bb) == params.tgtEos
        num_decoded = num_decoded + 1;
        candidates{num_decoded} = new_beam.hists{bb};
        scores{num_decoded} = new_beam.probs{bb};
        new_beam.probs{bb} = 0;
      end
    end

    beam = new_beam;
    show_beam(beam, beam_size, params);
  end

  for bb = 1 : beam_size
    num_decoded = num_decoded + 1;
    candidates{num_decoded} = beam.hists{bb};
    scores{num_decoded} = beam.probs{bb};
  end
end

function [best_probs, best_words] = next_beam_step(model, params, h, beam_size)
  [probs, ~, ~] = softmax(model.W_soft, h);
  [sorted_probs, sorted_words] = sort(probs(:), 'descend');
  best_words = sorted_words(1 : beam_size);
  best_probs = log(sorted_probs(1 : beam_size));
end

function [] = show_beam(beam, beam_size, params)
  fprintf(2, 'current beam state:\n');
  for bb = 1 : beam_size
    fprintf(2, 'hypothesis: ');
    for i = 1 : length(beam.hists{bb})
      fprintf(2, '%s ', params.vocab{beam.hists{bb}(i)});
    end
    fprintf(2, ' -> %f\n', beam.probs{bb});
    for ll = 1 : params.numLayers
      [beam.lstms{bb}{ll}.h_t(1:10), beam.lstms{bb}{ll}.c_t(1:10)]'
      fprintf(2, '------\n');
    end
    fprintf(2, '===================\n');
  end
  fprintf(2, 'end_beam ================================================== end_beam\n');
end
