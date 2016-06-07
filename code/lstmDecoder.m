function [candidates, candScores, alignInfo, encoderInfo] = lstmDecoder(models, data, params, prevInfo)
% Decode from an LSTM model.
%   stackSize: the maximum number of translations we want to get.
% Output:
%   - candidates: list of candidates
%   - candScores: score of the corresponding candidates (stackSize * batchSize)
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%   With help from Hieu Pham.

  % backward compatibility: not a cell, single model, put into a cell format
  if ~iscell(models)
    tmpModels = cell(1, 1);
    tmpModels{1} = models;
    tmpModels{1}.params = params;
    models = tmpModels;
  end
  
  beamSize = params.beamSize;
  stackSize = params.stackSize;
  startTime = clock;
  
  %%%%%%%%%%%%
  %% encode %%
  %%%%%%%%%%%%
  [prevStates, modelData, models] = runEncoder(models, data, params, prevInfo);
  
  % reuse encoder
  if params.reuseEncoder
    encoderInfo.prevStates = prevStates;
    encoderInfo.modelData = modelData;
    encoderInfo.srcInput = data.srcInput;
  else
    encoderInfo = [];
  end
  
  %%%%%%%%%%%%
  %% decode %%
  %%%%%%%%%%%%  
  %% init
  minLen = floor(data.srcMinLen*params.minLenRatio);
  if params.forceDecoder
    maxLen = data.tgtMaxLen;
  else
    maxLen = floor(data.srcMaxLen*params.maxLenRatio);
  end
  batchSize = size(data.srcInput, 1);
  fprintf(2, '# Decoding batch of %d sents, minLen=%d, maxLen=%d, %s\n', batchSize, minLen, maxLen, datestr(now));
  fprintf(params.logId, '# Decoding batch of %d sents, minLen=%d, maxLen=%d, %s\n', batchSize, minLen, maxLen, datestr(now));
  
  assert(unique(data.tgtInput(:, 1)) == params.tgtSos);
  [candidates, candScores, alignInfo] = rnnDecoder(models, params, prevStates, minLen, maxLen, beamSize, stackSize, batchSize, ...
    modelData, data, params.tgtSos);

  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  fprintf(2, '  Done, minLen=%d, maxLen=%d, speed %f sents/s, time %.0fs, %s\n', minLen, maxLen, batchSize/timeElapsed, timeElapsed, datestr(now));
  fprintf(params.logId, '  Done, minLen=%d, maxLen=%d, speed %f sents/s, time %.0fs, %s\n', minLen, maxLen, batchSize/timeElapsed, timeElapsed, datestr(now));
end

%%
% Take the initial states, starting symbol, and then decode!
% This method should be reusable.
%%
function [candidates, candScores, alignInfo] = rnnDecoder(models, params, prevStates, minLen, maxLen, beamSize, stackSize, batchSize, ...
  modelData, data, startSos)
  numModels = length(models);
  
  % first decoder timestep
  attnInfos = cell(numModels, 1);
  for mm=1:numModels
    % local monotonic alignment
    if models{mm}.params.attnLocalMono
      modelData{mm}.tgtPos = 1;
    end
    
    decRnnFlags = struct('decode', 1, 'test', 1, 'attn', models{mm}.params.attnFunc, 'feedInput', models{mm}.params.feedInput);
    [prevStates{mm}, attnInfos{mm}] = rnnStepLayerForward(models{mm}.W_tgt, models{mm}.W_emb_tgt(:, repmat(startSos, batchSize, 1)), ...
      prevStates{mm}, modelData{mm}.tgtMask(:, 1), models{mm}.params, decRnnFlags, modelData{mm}, models{mm});
  end
 
  % output alignment
  if params.align
    params.srcMaxLen = data.srcMaxLen;
    [~, firstAlignIdx] = getAlignWeights(attnInfos, data.srcLens, models, params);
  else
    firstAlignIdx = 0;
  end
  
  % decode the rest
  sentIndices = data.startId:(data.startId+batchSize-1);
  [candidates, candScores, alignInfo] = decodeBatch(models, params, prevStates, minLen, maxLen, beamSize, stackSize, batchSize, ...
    sentIndices, modelData, firstAlignIdx, data);
end

%%
% Beam decoder from an LSTM model, works for multiple sentences
% Input:
%   - encoded vector of the source sentences
%   - maximum length willing to go
%   - beamSize
%   - stackSize: maximum number of translations collected for one example
%%
function [candidates, candScores, alignInfo] = decodeBatch(models, params, prevStates, minLen, maxLen, beamSize, stackSize, batchSize, ...
originalSentIndices, modelData, firstAlignIdx, data)
  numElements = batchSize*beamSize;
  
  candidates = cell(batchSize, 1);
  candScores = -1e10*oneMatrix([stackSize, batchSize], params.isGPU, params.dataType); % set to a very small value
  numDecoded = zeros(batchSize, 1);
  for ii=1:batchSize
    candidates{ii} = cell(stackSize, 1);
  end
  % align
  if params.align
    alignInfo = cell(batchSize, 1);
    for ii=1:batchSize
      alignInfo{ii} = cell(stackSize, 1);
    end
  else
    alignInfo = [];
  end
  
  %% first prediction
  numModels = length(models);
  % scores, words: beamSize * batchSize
  if (params.prefixDecoder || params.forceDecoder) && data.prefixLen>=1
    [scores, words] = nextBeamStep(models, prevStates, beamSize, data.prefixSent(1)); 
    
    % these checks are necessary to ensure the init beam code below is correct
    assert(batchSize == 1);
    assert(length(words)==1);
    assert(length(firstAlignIdx)==1);
  else
    % we get (beamSize+1) choices so that we can filter out tgtEos (if any)
    % for good model, this step is not needed
    [scores, words] = nextBeamStep(models, prevStates, beamSize + 1);
    
    % go and find eos
    for ii=1:batchSize
      eosPos = find(words(:, ii) == params.tgtEos, 1);
      if ~isempty(eosPos) % swap with the last element
        scores(eosPos, ii) = scores(beamSize + 1, ii);
        words(eosPos, ii) = words(beamSize + 1, ii);
      end
    end
    % remove the last row
    scores(beamSize+1, :) = [];
    words(beamSize+1, :) = [];
    
    % [scores, words] = nextBeamStep(models, prevStates, beamSize);
  end

  
  %% matrix dimension note
  % note that we order matrices in the following dimension: n * numElements
  % columns correspond to the 1st sent go first, then the 2nd one, until the batchSize-th sent.
  sentIndices = repmat(1:batchSize, [beamSize, 1]);
  sentIndices = sentIndices(:)'; % 1 ... 1, 2 ... 2, ...., batchSize ... batchSize . 1 * (beamSize*batchSize)
  
  %% init beam
  beamScores = scores(:)'; % 1 * numElements
  beamHistory = zeroMatrix([maxLen, numElements], params.isGPU, params.dataType); % maxLen * (numElements) 
  beamHistory(1, :) = words(:); % words for sent 1 go together, then sent 2, ...
  if params.align
    alignHistory = zeroMatrix([maxLen, numElements], params.isGPU, params.dataType); % maxLen * (numElements) 
    for ii=1:batchSize
      alignHistory(1,(ii-1)*beamSize+1:ii*beamSize) = firstAlignIdx(ii);
    end
  end
  
  % replicate
  beamStates = cell(numModels, 1);
  for mm=1:numModels % model    
    beamStates{mm} = cell(models{mm}.params.numLayers, 1);
    for ll=1:models{mm}.params.numLayers % lstmSize * numElements
      beamStates{mm}{ll}.c_t = reshape(repmat(prevStates{mm}{ll}.c_t, beamSize, 1),  models{mm}.params.lstmSize, numElements); 
      beamStates{mm}{ll}.h_t = reshape(repmat(prevStates{mm}{ll}.h_t, beamSize, 1),  models{mm}.params.lstmSize, numElements); 
    end
    
    beamStates{mm}{end}.softmax_h = reshape(repmat(prevStates{mm}{end}.softmax_h, beamSize, 1),  models{mm}.params.lstmSize, numElements); 
  end
  

  oneMask = ones(1, numElements);
  data.srcLens = reshape(repmat(data.srcLens, beamSize, 1), 1, []);

  % attentional / positional models
  for mm=1:numModels % model
    if models{mm}.params.attnFunc
      models{mm}.params.curBatchSize = numElements;
    
      modelData{mm}.curBatchSize = numElements;
      modelData{mm}.srcLens = reshape(repmat(modelData{mm}.srcLens, beamSize, 1), 1, []);
      
      % duplicate srcHidVecs
      modelData{mm}.srcHidVecsOrig = duplicateSrcHidVecs(modelData{mm}.srcHidVecsOrig, batchSize, beamSize);
      
      %% replicate srcMask
      % first transpose to srcMaxLen * batchSize
      % replicate (srcMaxLen*beamSize)*batchSize
      % reshape maxLen*(beamSize*batchSize)
      % then transpose back to (beamSize*batchSize)*maxLen
      srcMaxLen = size(modelData{mm}.srcMask, 2);
      modelData{mm}.srcMask = reshape(repmat(modelData{mm}.srcMask', beamSize, 1), srcMaxLen, beamSize*batchSize)'; 
    end
  end
  
  decodeCompleteCount = 0; % count how many sentences we have completed collecting the translations
  beamWords = zeroMatrix([1, numElements], params.isGPU, params.dataType);
  beamIndices = zeroMatrix([1, numElements], params.isGPU, params.dataType);
  % align
  if params.align
    beamAlignIds = zeroMatrix([1, numElements], params.isGPU, params.dataType);
  end

  if beamSize == 1 % useful for force decoding
    doneFlags = zeros(1, batchSize); % mark if we have finished decoding a sentence
  end
  
  attnInfos = cell(numModels, 1);
  for sentPos = 1 : (maxLen-1)
    %% Description:
    % At this point, hypotheses of length sentPos are completed.
    % If sentPos<maxLen, this loop will prepare hypotheses of length(sentPos+1) by:
    %   (a) first, finding out what the top beamSize^2 nextWords are.
    %   (b) among these nextWords, those which are equal to <eos>, will signal
    %      a complete translations and we will collect. These complete
    %      translations have length (sentPos+1), inclusive of <eos>.
    %   (c) we keep beamSize non-eos nextWords to build hypotheses of length (sentPos+1).
    % For dependency parsing, we expect R(root) instead of <eos>. 
    %   When collect translations, only at sentPos==(maxLen-1), we will
    %   automatically append <eos>. So the final translations have length =
    %   (maxLen+1), ending in R(root) <eos>.
    tgtPos = sentPos+1;
    
    %% compute next lstm hidden states
    for mm=1:numModels
      % local monotonic alignment
      if models{mm}.params.attnLocalMono
        modelData{mm}.tgtPos = tgtPos;
      end
      
      decRnnFlags = struct('decode', 1, 'test', 1, 'attn', models{mm}.params.attnFunc, 'feedInput', models{mm}.params.feedInput);
      [beamStates{mm}, attnInfos{mm}] = rnnStepLayerForward(models{mm}.W_tgt, models{mm}.W_emb_tgt(:, beamHistory(sentPos, :)), beamStates{mm}, ...
        oneMask, models{mm}.params, decRnnFlags, modelData{mm}, models{mm});
    end

    %% output alignment
    if params.align
      [~, alignIdx] = getAlignWeights(attnInfos, data.srcLens, models, params);

      % we want to mimic the structure of allBestWords
      % size (beamSize * beamSize) x batchSize
      alignIdx = reshape(repmat(alignIdx, beamSize, 1), [], batchSize);
    end
    
    %% predict the next word
    % allBestScores, allBestWords should have size beamSize * (beamSize*batchSize)
    if (params.prefixDecoder || params.forceDecoder) && data.prefixLen>=tgtPos
      if params.forceDecoder
        [allBestScores, allBestWords] = nextBeamStep(models, beamStates, beamSize, data.prefixSent(tgtPos));
        assert(length(allBestWords) == 1);
        assert(length(allBestScores) == 1);
      else
        % TODO: remove all repmat here, since all hypotheses are the same at this point.
        [allBestScores, allBestWords] = nextBeamStep(models, beamStates, beamSize, repmat(data.prefixSent(tgtPos), 1, beamSize));
        assert(size(allBestWords, 1) == 1);
        assert(size(allBestScores, 1) == 1);
        allBestScores = repmat(allBestScores, beamSize, 1);
        allBestWords = repmat(allBestWords, beamSize, 1);
      end
    else
      [allBestScores, allBestWords] = nextBeamStep(models, beamStates, beamSize);
    end

    % allBestWords, allBestScores should have size: (beamSize*beamSize) * batchSize
    [allBestScores, allBestWords, indices] = addSortScores(allBestScores, allBestWords, beamScores, beamSize, batchSize);

    %% build new beam
    for sentId=1:batchSize
      rowIndices = indices(:, sentId)';
      bestWords = allBestWords(rowIndices, sentId);
      
      % get candidates
      selectedIndices = find(bestWords~=params.tgtEos, beamSize);
      if ~isempty(selectedIndices)
        % update words
        startId = (sentId-1)*beamSize+1;
        endId = sentId*beamSize;
        sentWords = bestWords(selectedIndices);
        beamWords(startId:endId) = sentWords;

        % align
        if params.align
          bestAlignIds = alignIdx(rowIndices, sentId);
          beamAlignIds(startId:endId) = bestAlignIds(selectedIndices);
        end

        % update beam
        sentBeamIndices = floor((rowIndices(selectedIndices)-1)/beamSize) + 1;
        beamIndices(startId:endId) = sentBeamIndices;

        % update scores
        beamScores(startId:endId) = allBestScores(selectedIndices, sentId);

        % get words that are eos and ranked before the last hypothesis in the next beam
        endIndices = find(bestWords(1:selectedIndices(end))==params.tgtEos); 
      else % special case, happen for beamSize = 1, useful for force decoding
        assert(beamSize == 1);
        if doneFlags(sentId) == 1
          continue;
        else
          doneFlags(sentId) = 1;
          endIndices = 1;
        end
      end
      
      %% store translations
      if ~isempty(endIndices) && tgtPos>minLen % we don't want to start recording very short translations
        numTranslations = length(endIndices);
        eosBeamIndices = floor((rowIndices(endIndices)-1)/beamSize) + 1;
        histIndices = (sentId-1)*beamSize + eosBeamIndices;

        transScores = allBestScores(endIndices, sentId);
        for ii=1:numTranslations
          if numDecoded(sentId)<stackSize % haven't collected enough translations
            numDecoded(sentId) = numDecoded(sentId) + 1;

            candidates{sentId}{numDecoded(sentId)} = [beamHistory(1:sentPos, histIndices(ii)); params.tgtEos];
            candScores(numDecoded(sentId), sentId) = transScores(ii);

            % align
            if params.align
              alignInfo{sentId}{numDecoded(sentId)} = [alignHistory(1:sentPos, histIndices(ii)); beamAlignIds(histIndices(ii))];
            end

            if numDecoded(sentId)==stackSize % done for sentId
              decodeCompleteCount = decodeCompleteCount + 1;
              break;
            end
          end
        end
      end
    end
    
    %% update history
    % overwrite previous history
    colIndices = (sentIndices-1)*beamSize + beamIndices;
    tmp = beamHistory(1:sentPos, colIndices);
    beamHistory(1:sentPos, :) = tmp; 
    beamHistory(sentPos+1, :) = beamWords;

    
    if params.debug
      assert(batchSize == 1);
      fprintf(2, '# Tgt pos %d\n', tgtPos);
      for ii=1:beamSize
        printSent(2, beamHistory(1:sentPos+1, ii), params.tgtVocab, [' ' num2str(beamScores(ii)) '  beam ' num2str(ii) ': ']);
      end
    end
    
    % align
    if params.align
      alignHistory(1:sentPos, :) = alignHistory(1:sentPos, colIndices);
      alignHistory(sentPos+1, :) = beamAlignIds;
    end
    
    %% update lstm states
    for mm=1:numModels % model
      for ll=1:models{mm}.params.numLayers
        % lstmSize * (numElements): h_t and c_t vectors of each sent are arranged near each other
        beamStates{mm}{ll}.c_t = beamStates{mm}{ll}.c_t(:, colIndices); 
        beamStates{mm}{ll}.h_t = beamStates{mm}{ll}.h_t(:, colIndices);
      end
      beamStates{mm}{end}.softmax_h = beamStates{mm}{end}.softmax_h(:, colIndices);
    end
    
    if decodeCompleteCount==batchSize % done decoding the entire batch
      break;
    end
  end % for sentPos
  
  for sentId=1:batchSize
    if numDecoded(sentId) == 0 % no translations found, output all we have in the beam
      fprintf(2, '  ! Sent %d: no translations end in eos\n', originalSentIndices(sentId));
      for bb = 1:beamSize
        eosIndex = (sentId-1)*beamSize + bb;
        numDecoded(sentId) = numDecoded(sentId) + 1;
        candidates{sentId}{numDecoded(sentId)} = [beamHistory(1:maxLen, eosIndex); params.tgtEos]; % append eos at the end
        
        % align
        if params.align
          if params.isReverse
            alignInfo{sentId}{numDecoded(sentId)} = [alignHistory(1:maxLen, eosIndex); 1];
          else
            alignInfo{sentId}{numDecoded(sentId)} = [alignHistory(1:maxLen, eosIndex); modelData{1}.srcLens(sentId)-1];
          end
        end
        
        candScores(numDecoded(sentId), sentId) = beamScores(eosIndex);
      end
    end
    candidates{sentId}(numDecoded(sentId)+1:end) = [];
  end
end

function [allBestScores, allBestWords, indices] = addSortScores(allBestScores, allBestWords, beamScores, beamSize, batchSize)
  % use previous beamScores, 1 * (beamSize*batchSize), update along the first dimentions
  allBestScores = bsxfun(@plus, allBestScores, beamScores);
  allBestScores = reshape(allBestScores, [beamSize*beamSize, batchSize]);
  allBestWords = reshape(allBestWords, [beamSize*beamSize, batchSize]);

  % for each sent, select the best beamSize candidates, out of beamSize*beamSize ones
  if beamSize > 1
    [allBestScores, indices] = sort(allBestScores, 'descend'); % (beamSize*beamSize) * batchSize
  else
    indices = ones(1, batchSize);
  end
end

%%
% return bestLogProbs, bestWords of sizes beamSize * curBatchSize
%%
function [bestLogProbs, bestWords] = nextBeamStep(models, lastDecStates, beamSize, varargin)
  softmax_input = models{1}.W_soft*lastDecStates{1}{end}.softmax_h;
  if length(models)>1 % aggregate predictions from multiple models
    for ii=2:length(models)
      softmax_input = softmax_input + models{ii}.W_soft*lastDecStates{ii}{end}.softmax_h;
    end
    softmax_input = softmax_input./length(models);
  end
  [logProbs] = softmaxDecode(softmax_input);
  
  % sort
  if length(varargin)==1 % force decoder
    bestWords = reshape(varargin{1}, 1, []);
    bestLogProbs = logProbs(sub2ind(size(logProbs), bestWords, 1:size(logProbs, 2)));
  else
    [sortedLogProbs, sortedWords] = sort(logProbs, 'descend');
    bestWords = sortedWords(1:beamSize, :);
    bestLogProbs = sortedLogProbs(1:beamSize, :);
  end
end

function [logProbs] = softmaxDecode(scores)
%% only compute logProbs
  mx = max(scores);
  scores = bsxfun(@minus, scores, mx); % subtract max elements 
  logProbs = bsxfun(@minus, scores, log(sum(exp(scores))));
end

function [srcHidVecs] = duplicateSrcHidVecs(srcHidVecs, batchSize, beamSize)
  numElements = batchSize*beamSize;
  lstmSize = size(srcHidVecs, 1);
  numPositions = size(srcHidVecs, 3);
  srcHidVecs = permute(srcHidVecs, [1, 3, 2]); % lstmSize * numPositions * batchSize
  srcHidVecs = reshape(srcHidVecs, lstmSize*numPositions, batchSize);
  srcHidVecs = repmat(srcHidVecs, beamSize, 1);
  srcHidVecs = reshape(srcHidVecs, lstmSize, numPositions, numElements);
  srcHidVecs = permute(srcHidVecs, [1, 3, 2]); % lstmSize * numElements * numPositions
end
