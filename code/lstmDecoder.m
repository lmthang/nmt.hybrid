function [candidates, candScores] = lstmDecoder(model, data, params, beamSize, stackSize)
%%%
%
% Decode from an LSTM model.
%   stackSize: the maximum number of translations we want to get.
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
% Thang Luong @ 2015, <lmthang@stanford.edu>
% Hieu Pham @ 2015, <hyhieu@cs.stanford.edu>
%
%%%
function [candidates, candScores] = decodeBatch(model, params, lstmStart, maxLen, beamSize, stackSize, sentIndices)
  W_tgt = model.W_tgt;
  W_emb = model.W_emb;
  numLayers = params.numLayers;
  batchSize = size(lstmStart{numLayers}.h_t, 2);
  
  candidates = cell(batchSize, 1);
  candScores = cell(batchSize, 1);
  for bb=1:batchSize
    candidates{bb} = cell(stackSize, 1);
    candScores{bb} = zeros(stackSize, 1);
  end
  
  numDecoded = zeros(batchSize, 1);
  [beam.scores, words] = nextBeamStep(model, lstmStart{numLayers}.h_t, beamSize); % scores, words: beamSize * batchSize
  beam.lstms = cell(beamSize, 1);
  beam.hists = cell(beamSize, 1);
  for bb=1:beamSize
    beam.lstms{bb} = lstmStart; 
    beam.hists{bb} = words(bb, :)'; % batchSize * sentPos, at the moment sentPos=1
  end
  
%   show_beam(beam, beam_size, params);
  decodeCompleteCount = 0; % count how many sentences we have completed collecting the translations
  for sentPos = 1 : maxLen
    fprintf(2, '%d ', sentPos);
    allBestScores = zeroMatrix([beamSize*beamSize, batchSize], params.isGPU, params.dataType);
    allBestWords = zeroMatrix([beamSize*beamSize, batchSize], params.isGPU, params.dataType);
    allBestBeams = zeroMatrix([beamSize*beamSize, batchSize], params.isGPU, params.dataType);

    for bb = 1 : min(beamSize, length(beam.scores))
      allBestBeams(((bb-1)*beamSize+1) : (bb*beamSize), :) = bb;
      
      % accumulate the last word in history
      % to compute all lstm layers
      words = beam.hists{bb}(:, end);
      lstmCur = cell(numLayers, 1);

      for ll = 1 : numLayers
        % current input
        if ll == 1
          x_t = W_emb(:, words);
        else
          x_t = lstmCur{ll-1}.h_t;
        end
        % previous input
        h_t_1 = beam.lstms{bb}{ll}.h_t;
        c_t_1 = beam.lstms{bb}{ll}.c_t;
      
        lstmCur{ll} = lstmUnit(W_tgt{ll}, x_t, h_t_1, c_t_1, params, 1);
      end

      beam.lstms{bb} = lstmCur;

      % predict the next word
      [bestNextScores, bestNextWords] = nextBeamStep(model, lstmCur{numLayers}.h_t, beamSize);
      allBestScores(((bb-1)*beamSize+1) : (bb*beamSize), :) = bsxfun(@plus, bestNextScores, beam.scores(bb, :));
      allBestWords(((bb-1)*beamSize+1) : (bb*beamSize), :) = bestNextWords;
      
    end % end for bb


    %% update beam
    [sortedBestScores, indices] = sort(allBestScores, 'descend');
    bestIndices = indices(1:beamSize, :);
    bestWords = allBestWords(bestIndices);
    prevBeams = allBestBeams(bestIndices); % this tells us which beam each hypothesis originally comes from. beamSize * batchSize
    % init new beam
    new_beam.scores = sortedBestScores(1:beamSize, :);
    new_beam.hists = cell(beamSize, 1);
    for bb=1:beamSize
      new_beam.hists{bb} = zeros(batchSize, sentPos+1);
      new_beam.lstms{bb} = cell(numLayers, 1);
      for ll=1:numLayers
        new_beam.lstms{bb}{ll}.h_t = zeroMatrix([params.lstmSize, batchSize], params.isGPU, params.dataType);
        new_beam.lstms{bb}{ll}.c_t = zeroMatrix([params.lstmSize, batchSize], params.isGPU, params.dataType);
      end
    end
    % populate new beam
    for jj=1:batchSize
      for bb=1:beamSize
        % combine beam.hists(beamIndices{bb, jj), jj} with bestWords(bb, jj)
        prevBeam = prevBeams(bb, jj);
        word = bestWords(bb, jj);
        new_beam.hists{bb}(jj, :) = [beam.hists{prevBeam}(jj, :), word];
        
        % copy lstm state
        for ll=1:numLayers
          new_beam.lstms{bb}{ll}.h_t(:, jj) = beam.lstms{prevBeam}{ll}.h_t(:, jj);
          new_beam.lstms{bb}{ll}.c_t(:, jj) = beam.lstms{prevBeam}{ll}.c_t(:, jj);
        end
        
        % eos
        if word == params.tgtEos && numDecoded(jj)<stackSize % haven't collected enough translations
          numDecoded(jj) = numDecoded(jj) + 1;
          candidates{jj}{numDecoded(jj)} = new_beam.hists{bb}(jj, :);
          candScores{jj}(numDecoded(jj)) = new_beam.scores(bb, jj);
          
          if numDecoded(jj)==stackSize
            decodeCompleteCount = decodeCompleteCount + 1;
          end
        end
      end
    end
    
    if decodeCompleteCount==batchSize % done decoding the entire batch
      break;
    end
    
    beam = new_beam;
    %showBeam(beam, beamSize, batchSize, params);
  end
  fprintf(2, 'Done! decoding maxLen=%d\n', maxLen);
  for jj=1:batchSize
    if numDecoded(jj) == 0 % no translations found, output all we have in the beam
      fprintf(2, '! Sent %d: no translations end in eos\n', sentIndices(jj));
      for bb = 1:beamSize
        numDecoded(jj) = numDecoded(jj) + 1;
        candidates{jj}{numDecoded(jj)} = [beam.hists{bb}(jj, :) params.tgtEos];
        candScores{jj}(numDecoded(jj)) = beam.scores(bb, jj);
      end
    end
    candidates{jj}(numDecoded(jj)+1:end) = [];
    candScores{jj}(numDecoded(jj)+1:end) = [];
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
