function [candidates, candScores] = lstmDecoder(model, data, params, beamSize, stackSize)
%%%
%
% Decode from an LSTM model.
%   stackSize: the maximum number of translations we want to get.
% Output:
%   - candidates: list of candidates
%   - scores: score of the corresponding candidates
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
% Hieu Pham @ 2015, <hyhieu@cs.stanford.edu>
%
%%%
  input = data.input;
  inputMask = data.inputMask; 
  srcMaxLen = data.srcMaxLen;
  
  %% init
  curBatchSize = size(input, 1);
  zeroState = zeroMatrix([params.lstmSize, curBatchSize], params.isGPU, params.dataType);

  %%%%%%%%%%%%
  %% encode %%
  %%%%%%%%%%%%
  lstm = cell(params.numLayers, 1); % lstm can be over written, as we do not need to backprop
  for t=1:srcMaxLen % time
    % prepare mask
    mask = inputMask(:, t)'; % curBatchSize * 1
    maskedIds = find(~mask);

    for ll=1:params.numLayers % layer
      %% encoder W matrix
      W = model.W_src{ll};
      
      %% previous-time input
      if t==1 % first time step
        h_t_1 = zeroState;
        c_t_1 = zeroState;
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
      lstm{ll} = lstmUnit(W, x_t, h_t_1, c_t_1, params, 1);
    end
  end
  
  [candidates, candScores] = decodeBatch(model, params, lstm, srcMaxLen*2, beamSize, stackSize, data.sentIndices);
end

%%%
%
% Beam decoder from an LSTM model, works for multiple sentences
%
% Input:
%   - encoded vector of the source sentences
%   - maximum length willing to go
%   - beamSize
%   - stackSize: maximum number of translations collected for one example
%
%%%
function [candidates, candScores] = decodeBatch(model, params, lstmStart, maxLen, beamSize, stackSize, originalSentIndices)
  startTime = clock;
  numLayers = params.numLayers;
  batchSize = size(lstmStart{numLayers}.h_t, 2);
  
  candidates = cell(batchSize, 1);
  candScores = cell(batchSize, 1);
  numDecoded = zeros(batchSize, 1);
  for bb=1:batchSize
    candidates{bb} = cell(stackSize, 1);
    candScores{bb} = zeroMatrix([stackSize, 1], params.isGPU, params.dataType);
  end
  
  % first prediction
  [scores, words] = nextBeamStep(model, lstmStart{numLayers}.h_t, beamSize); % scores, words: beamSize * batchSize
  beamScores = scores(:)'; % 1 * (beamSize*batchSize)
  beamHistory = zeroMatrix([maxLen, batchSize*beamSize], params.isGPU, params.dataType); % maxLen * (batchSize*beamSize) 
  beamHistory(1, :) = words(:); % words for sent 1 go together, then sent 2, ...
  beamStates = cell(numLayers, 1);
  for ll=1:numLayers
    % lstmSize * (batchSize*beamSize): h_t and c_t vectors of each sent are arranged near each other
    beamStates{ll}.c_t = reshape(repmat(lstmStart{ll}.c_t, beamSize, 1),  params.lstmSize, batchSize*beamSize); 
    beamStates{ll}.h_t = reshape(repmat(lstmStart{ll}.h_t, beamSize, 1),  params.lstmSize, batchSize*beamSize); 
  end
  
%   show_beam(beam, beam_size, params);
  decodeCompleteCount = 0; % count how many sentences we have completed collecting the translations
  sentIndices = repmat(1:batchSize, beamSize, 1);
  sentIndices = sentIndices(:)'; % 1 ... 1, 2 ... 2, ...., batchSize ... batchSize . 1 * (beamSize*batchSize)
  for sentPos = 1 : maxLen
    fprintf(2, '%d ', sentPos);

    % compute next lstm hidden states
    words = beamHistory(sentPos, :);
    for ll = 1 : numLayers
      % current input
      if ll == 1
        x_t = model.W_emb(:, words);
      else
        x_t = beamStates{ll-1}.h_t;
      end
      % previous input
      h_t_1 = beamStates{ll}.h_t;
      c_t_1 = beamStates{ll}.c_t;

      beamStates{ll} = lstmUnit(model.W_tgt{ll}, x_t, h_t_1, c_t_1, params, 1);
    end
    
    % predict the next word
    [allBestScores, allBestWords] = nextBeamStep(model, beamStates{numLayers}.h_t, beamSize); % beamSize * (beamSize*batchSize)
    
    % use previous beamScores, 1 * (beamSize*batchSize), update along the first dimentions
    allBestScores = bsxfun(@plus, allBestScores, beamScores);
    allBestScores = reshape(allBestScores, [beamSize*beamSize, batchSize]);
    allBestWords = reshape(allBestWords, [beamSize*beamSize, batchSize]);
    
    % for each sent, select the best beamSize candidates, out of beamSize^2 ones
    [sortedBestScores, indices] = sort(allBestScores, 'descend'); % beamSize^2 * batchSize
    
    %% update scores
    beamScores = sortedBestScores(1:beamSize, :);
    beamScores = beamScores(:)';
    
    %% update history
    % find out which beams these beamSize*batchSize derivations came from
    rowIndices = indices(1:beamSize, :); % beamSize * batchSize
    rowIndices = rowIndices(:)';
    beamIndices = floor((rowIndices-1)/beamSize) + 1; 
    % figure out best next words
    nextWords = allBestWords(sub2ind(size(allBestWords), rowIndices, sentIndices))';
    % overwrite previous history
    colIndices = (sentIndices-1)*beamSize + beamIndices;
    beamHistory(1:sentPos, :) = beamHistory(1:sentPos, colIndices); 
    beamHistory(sentPos+1, :) = nextWords;
    
    %% update lstm states
    for ll=1:numLayers
      % lstmSize * (batchSize*beamSize): h_t and c_t vectors of each sent are arranged near each other
      beamStates{ll}.c_t = beamStates{ll}.c_t(:, colIndices); 
      beamStates{ll}.h_t = beamStates{ll}.h_t(:, colIndices);
    end
 
    %% find out if some derivations reach eos
    eosIndices = find(nextWords == params.tgtEos);
    for ii=1:length(eosIndices)
      eosIndex = eosIndices(ii);
      sentId = sentIndices(eosIndex);
      
      if numDecoded(sentId)<stackSize % haven't collected enough translations
        numDecoded(sentId) = numDecoded(sentId) + 1;
        candidates{sentId}{numDecoded(sentId)} = beamHistory(1:sentPos+1, eosIndex);
        candScores{sentId}(numDecoded(sentId)) = beamScores(eosIndex);
        
        beamScores(eosIndex) = -1e10; % make the beam score small so that it won't make into the candidate list again

        if numDecoded(sentId)==stackSize % done for sentId
          decodeCompleteCount = decodeCompleteCount + 1;
        end
      end
    end
    
    if decodeCompleteCount==batchSize % done decoding the entire batch
      break;
    end
    %showBeam(beam, beamSize, batchSize, params);
  end
  
  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  fprintf(2, '\nDone decoding %d sents, maxLen=%d, speed %f sents/s\n', batchSize, maxLen, batchSize/timeElapsed);
  for sentId=1:batchSize
    if numDecoded(sentId) == 0 % no translations found, output all we have in the beam
      fprintf(2, '! Sent %d: no translations end in eos\n', originalSentIndices(sentId));
      for bb = 1:beamSize
        eosIndex = (bb-1)*beamSize + sentId;
        numDecoded(sentId) = numDecoded(sentId) + 1;
        candidates{sentId}{numDecoded(sentId)} = [beamHistory(1:sentPos+1, eosIndex); params.tgtEos]; % append eos at the end
        candScores{sentId}(numDecoded(sentId)) = beamScores(eosIndex);
      end
    end
    candidates{sentId}(numDecoded(sentId)+1:end) = [];
    candScores{sentId}(numDecoded(sentId)+1:end) = [];
  end
end


function [bestLogProbs, bestWords] = nextBeamStep(model, h, beamSize)
  % return bestLogProbs, bestWords of sizes beamSize * curBatchSize
  [logProbs] = softmaxDecode(model.W_soft*h);
  [sortedLogProbs, sortedWords] = sort(logProbs, 'descend');
  bestWords = sortedWords(1:beamSize, :);
  bestLogProbs = sortedLogProbs(1:beamSize, :);
end

function [logProbs] = softmaxDecode(scores)
%% only compute logProbs
  mx = max(scores);
  scores = bsxfun(@minus, scores, mx); % subtract max elements 
  logProbs = bsxfun(@minus, scores, log(sum(exp(scores))));
end

function [] = showBeam(beam, beam_size, batchSize, params)
  fprintf(2, 'current beam state:\n');
  for bb=1:beam_size
    for jj=1:batchSize
      fprintf(2, 'hypothesis beam %d, example %d\n', bb, jj);
      for ii=1:length(beam.hists{bb}(jj, :))
        fprintf(2, '%s ', params.vocab{beam.hists{bb}(jj, ii)});
      end
      fprintf(2, ' -> %f\n', beam.scores(bb, jj));
      fprintf(2, '===================\n');
    end
  end
  fprintf(2, 'end_beam ================================================== end_beam\n');
end
