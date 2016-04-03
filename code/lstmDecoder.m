function [candidates, candScores, alignInfo, otherInfo] = lstmDecoder(models, data, params)
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
  batchSize = size(data.srcInput, 1);
  
  srcMaxLen = data.srcMaxLen;
  params.srcMaxLen = srcMaxLen;

  %% init
  minLen = floor(data.srcMinLen*params.minLenRatio);
  if params.forceDecoder
    maxLen = data.tgtMaxLen;
  else
    maxLen = floor(srcMaxLen*params.maxLenRatio);
  end
  fprintf(2, '# Decoding batch of %d sents, minLen=%d, maxLen=%d, tgtEos=%d, %s\n', batchSize, minLen, maxLen, params.tgtEos, datestr(now));
  fprintf(params.logId, '# Decoding batch of %d sents, minLen=%d, maxLen=%d, tgtEos=%d, %s\n', batchSize, minLen, maxLen, params.tgtEos, datestr(now));
  
  startTime = clock;

  %%%%%%%%%%%%
  %% encode %%
  %%%%%%%%%%%%
  %% multiple models
  numModels = length(models);
  modelData = cell(numModels, 1);
  zeroStates = cell(numModels, 1);
  for mm=1:numModels
    models{mm}.params.curBatchSize = batchSize;
    models{mm}.params.srcMaxLen = srcMaxLen;
    [models{mm}.params] = setAttnParams(models{mm}.params);

    [zeroStates{mm}] = createZeroState(models{mm}.params);
    modelData{mm} = data;    
    
    % compatible
    [models{mm}.params] = backwardCompatible(models{mm}.params, {'unkDiscount'}, 0);
  end
  
  % encoder
  prevStates = cell(numModels, 1);
  for mm=1:numModels
    [~, prevStates{mm}, ~, modelData{mm}, ~] = encoderLayerForward(models{mm}, zeroStates{mm}, modelData{mm}, models{mm}.params, 1);
  end

  %%%%%%%%%%%%
  %% decode %%
  %%%%%%%%%%%%
  [candidates, candScores, alignInfo, otherInfo] = rnnDecoder(models, params, prevStates, minLen, maxLen, beamSize, stackSize, batchSize, ...
  modelData, data, params.tgtEos, 0);
  
  % char: now generate words for <unk>
  if params.charTgtGen % assume all models are char-level models
    assert(stackSize == 1); % TODO: remove this, then char_initEmb needs to be initialized differently.
    ss = 1;
    
    % prepare starting states
    char_initEmb = cell(numModels, 1);
    for mm=1:numModels
      char_initEmb{mm} = zeroMatrix([params.lstmSize, batchSize * maxLen], params.isGPU, params.dataType);
    end
    
    numRareWords = 0;
    otherInfo.rarePositions = cell(batchSize, 1);
    for sentId=1:batchSize
      flags = candidates{sentId}{ss} == params.tgtUnk;
      positions = find(flags);
      
      for mm=1:numModels
        char_initEmb{mm}(:, numRareWords+1:numRareWords+length(positions)) = otherInfo.transStates{ss, mm}(:, positions, sentId);
        % assert(isempty(find(otherInfo.transStates{ss, mm}(:, length(flags), sentId) ~= 0, 1)));
      end
      
      numRareWords = numRareWords + length(positions);
      otherInfo.rarePositions{sentId} = positions;
    end
    
    for mm=1:numModels
      char_initEmb{mm}(:, numRareWords+1:end) = [];
    end
    
    otherInfo.numRareWords = numRareWords;
    otherInfo.rareWords = cell(batchSize, 1);
    if numRareWords
      % char params
      charParams = params;
      charParams.curBatchSize = numRareWords;

      % model
      charModels = cell(numModels, 1);
      for mm=1:numModels
        charModels{mm}.W_tgt = models{mm}.W_tgt_char;
        charModels{mm}.W_emb_tgt = models{mm}.W_emb_tgt_char;
        charModels{mm}.W_soft = models{mm}.W_soft_char;
        
        charParams.numLayers = models{mm}.params.charNumLayers;
        charParams.tgtVocab = models{mm}.params.tgtCharVocab;
        
        charModels{mm}.params = charParams;
      end

      % prev states
      char_minLen = 2;
      char_maxLen = 20;
      fprintf(2, '  # Decoding char , minLen=%d, maxLen=%d, charEos=%d, %s\n', char_minLen, char_maxLen, params.tgtCharEos, datestr(now));
      fprintf(params.logId, '  # Decoding char , minLen=%d, maxLen=%d, charEos=%d, %s\n', char_minLen, char_maxLen, params.tgtCharEos, datestr(now));
      
      char_modelData = cell(numModels, 1);
      char_prevStates = cell(numModels, 1);
      zeroBatch = zeroMatrix([params.lstmSize, charParams.curBatchSize], params.isGPU, params.dataType);
      
      for mm=1:numModels
        char_prevStates{mm} = cell(charModels{mm}.params.numLayers, 1);
        for ll=1:charModels{mm}.params.numLayers % layer
          if ll == 1
            char_prevStates{mm}{ll}.h_t = char_initEmb{mm}(:, :);
          else
            char_prevStates{mm}{ll}.h_t = zeroBatch;
          end
          char_prevStates{mm}{ll}.c_t = zeroBatch;
        end
      end
      
      [char_candidates, ~, ~, ~] = rnnDecoder(charModels, charParams, char_prevStates, char_minLen, ...
        char_maxLen, beamSize, stackSize, numRareWords, char_modelData, [], params.tgtCharEos, 1);
      
      % get words
      count = 0;
      for sentId=1:batchSize
        numCharWords = length(otherInfo.rarePositions{sentId});
        otherInfo.rareWords{sentId} = cell(numCharWords, 1);
        for jj=1:numCharWords
          otherInfo.rareWords{sentId}{jj} = [params.tgtCharVocab{char_candidates{count+jj}{1}(1:end-1)}];
        end
        count = count + numCharWords;
      end
    end
  end
  
  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  fprintf(2, '  Done, minLen=%d, maxLen=%d, speed %f sents/s, time %.0fs, %s\n', minLen, maxLen, batchSize/timeElapsed, timeElapsed, datestr(now));
  fprintf(params.logId, '  Done, minLen=%d, maxLen=%d, speed %f sents/s, time %.0fs, %s\n', minLen, maxLen, batchSize/timeElapsed, timeElapsed, datestr(now));
end

%%
% Take the initial states, starting symbol, and then decode!
% This method should be reusable.
%%
function [candidates, candScores, alignInfo, otherInfo] = rnnDecoder(models, params, prevStates, minLen, maxLen, beamSize, stackSize, ...
  batchSize, modelData, data, tgtEos, isChar)
  numModels = length(models);
  
  % first decoder timestep
  attnInfos = cell(numModels, 1);
  firstAlignIdx = [];
  for mm=1:numModels
    % char
    if isChar
      isFeedInput = 0;
      attnFunc = 0;
    else
      isFeedInput = models{mm}.params.feedInput;
      attnFunc = models{mm}.params.attnFunc;
    end
    initEmb = models{mm}.W_emb_tgt(:, repmat(models{mm}.params.tgtSos, batchSize, 1));
    decRnnFlags = struct('decode', 1, 'test', 1, 'attn', attnFunc, 'feedInput', isFeedInput);
    [prevStates{mm}, attnInfos{mm}] = rnnStepLayerForward(models{mm}.W_tgt, initEmb, ...
      prevStates{mm}, ones(batchSize, 1), models{mm}.params, decRnnFlags, modelData{mm}, models{mm});
  end
 
  % output alignment
  if params.align && isChar == 0
    [~, firstAlignIdx] = getAlignWeights(attnInfos, data.srcLens, models, params);
  end
  
  if isChar
    sentIndices = 1:batchSize;
  else
    sentIndices = data.startId:(data.startId+batchSize-1);
  end
  [candidates, candScores, alignInfo, otherInfo] = decodeBatch(models, params, prevStates, attnInfos, minLen, maxLen, beamSize, stackSize, batchSize, ...
    sentIndices, modelData, firstAlignIdx, data, tgtEos, isChar);
end


%%
% Beam decoder from an LSTM model, works for multiple sentences
% Input:
%   - encoded vector of the source sentences
%   - maximum length willing to go
%   - beamSize
%   - stackSize: maximum number of translations collected for one example
%%
function [candidates, candScores, alignInfo, otherInfo] = decodeBatch(models, params, prevStates, attnInfos, minLen, maxLen, beamSize, stackSize, batchSize, ...
originalSentIndices, modelData, firstAlignIdx, data, tgtEos, isChar)
  numElements = batchSize*beamSize;
  
  candidates = cell(batchSize, 1);
  candScores = -1e10*oneMatrix([stackSize, batchSize], params.isGPU, params.dataType); % set to a very small value
  numDecoded = zeros(batchSize, 1);
  for ii=1:batchSize
    candidates{ii} = cell(stackSize, 1);
  end
  
  numModels = length(models);
  otherInfo = [];
  
  % align
  if params.align && isChar == 0
    alignInfo = cell(batchSize, 1);
    for ii=1:batchSize
      alignInfo{ii} = cell(stackSize, 1);
    end
  else
    alignInfo = [];
  end
  
  %% first prediction
  % scores, words: beamSize * batchSize
  if params.forceDecoder
    assert(isChar == 0);
    [scores, words, otherData] = nextBeamStep(models, prevStates, beamSize, 0, isChar, params.tgtUnk, params.unkDiscount, data.tgtOutput(:, 1)); 
    otherInfo.forceDecodeOutputs = zeroMatrix([maxLen, numElements], params.isGPU, params.dataType); % maxLen * (numElements) 
    otherInfo.forceDecodeOutputs(1, :) = otherData.maxWords;
  else
    [scores, words] = nextBeamStep(models, prevStates, beamSize, tgtEos, isChar, params.tgtUnk, params.unkDiscount);
  end

  % TODO: by right, we should filter out words == tgtEos, but I
  % think for good models, we don't have to worry :)
  
  %% matrix dimension note
  % note that we order matrices in the following dimension: n * numElements
  % columns correspond to the 1st sent go first, then the 2nd one, until the batchSize-th sent.
  sentIndices = repmat(1:batchSize, [beamSize, 1]);
  sentIndices = sentIndices(:)'; % 1 ... 1, 2 ... 2, ...., batchSize ... batchSize . 1 * (beamSize*batchSize)
  
  %% init beam
  beamScores = scores(:)'; % 1 * numElements
  beamHistory = zeroMatrix([maxLen, numElements], params.isGPU, params.dataType); % maxLen * (numElements) 
  beamHistory(1, :) = words(:); % words for sent 1 go together, then sent 2, ...
  if params.align && isChar == 0
    alignHistory = zeroMatrix([maxLen, numElements], params.isGPU, params.dataType); % maxLen * (numElements) 
    for ii=1:batchSize
      alignHistory(1,(ii-1)*beamSize+1:ii*beamSize) = firstAlignIdx(ii);
    end
  end
  
  
  %% replicate
  beamStates = cell(numModels, 1);  
  for mm=1:numModels % model
    beamStates{mm} = cell(models{mm}.params.numLayers, 1);
    for ll=1:models{mm}.params.numLayers % lstmSize * numElements
      beamStates{mm}{ll}.c_t = reshape(repmat(prevStates{mm}{ll}.c_t, beamSize, 1),  models{mm}.params.lstmSize, numElements); 
      beamStates{mm}{ll}.h_t = reshape(repmat(prevStates{mm}{ll}.h_t, beamSize, 1),  models{mm}.params.lstmSize, numElements); 
    end
    beamStates{mm}{end}.softmax_h = reshape(repmat(prevStates{mm}{end}.softmax_h, beamSize, 1),  models{mm}.params.lstmSize, numElements); 
  end
  
  %% char
  if params.charTgtGen && isChar == 0
    % assert(stackSize == 1);
    beamHistTopStates = cell(numModels, 1);
    otherInfo.transStates = cell(stackSize, numModels);
    for ss=1:stackSize
      for mm=1:numModels
        % sent 1, ..., sent2, ..., sent_batchSize
        otherInfo.transStates{ss, mm} = zeroMatrix([models{mm}.params.lstmSize, maxLen-1, batchSize], params.isGPU, params.dataType);
      end
    end
    for mm=1:numModels
      assert(models{mm}.params.charTgtGen == 1);
      beamHistTopStates{mm} = zeroMatrix([models{mm}.params.lstmSize, numElements, maxLen], params.isGPU, params.dataType);
      if models{mm}.params.charFeedOpt
        tgtWordStates = hiddenLayerForward(models{mm}.W_h_char, attnInfos{mm}.input, params.nonlinear_f); 
        beamHistTopStates{mm}(:, :, 1) = reshape(repmat(tgtWordStates, beamSize, 1),  models{mm}.params.lstmSize, numElements); 
      else
        beamHistTopStates{mm}(:, :, 1) = beamStates{mm}{end}.softmax_h;
      end
    end
  end
  
  oneMask = ones(1, numElements);
  
  if isChar == 0
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
  end
  
  % decodeCompleteCount = 0; % count how many sentences we have completed collecting the translations
  beamWords = zeroMatrix([1, numElements], params.isGPU, params.dataType);
  beamIndices = zeroMatrix([1, numElements], params.isGPU, params.dataType);
  % align
  if params.align && isChar == 0
    beamAlignIds = zeroMatrix([1, numElements], params.isGPU, params.dataType);
  end

  if beamSize == 1 % useful for force decoding
    doneFlags = zeros(1, batchSize); % mark if we have finished decoding a sentence
  end
  
  attnInfos = cell(numModels, 1);
  for sentPos=1:(maxLen-1)
    %% Description:
    % At this point, hypotheses of length sentPos are completed.
    % This loop will prepare hypotheses of length(sentPos+1) by:
    %   (a) first, finding out what the top beamSize^2 nextWords are.
    %   (b) among these nextWords, those which are equal to <eos>, will signal
    %      a complete translations and we will collect. These complete
    %      translations have length (sentPos+1), inclusive of <eos>.
    %   (c) we keep beamSize non-eos nextWords to build hypotheses of length (sentPos+1).
    tgtPos = sentPos+1;
    
    %% compute next lstm hidden states
    for mm=1:numModels
      % char
      if isChar
        isFeedInput = 0;
        attnFunc = 0;
      else
        isFeedInput = models{mm}.params.feedInput;
        attnFunc = models{mm}.params.attnFunc;
      end
      
      decRnnFlags = struct('decode', 1, 'test', 1, 'attn', attnFunc, 'feedInput', isFeedInput);
      [beamStates{mm}, attnInfos{mm}] = rnnStepLayerForward(models{mm}.W_tgt, models{mm}.W_emb_tgt(:, beamHistory(sentPos, :)), beamStates{mm}, ...
        oneMask, models{mm}.params, decRnnFlags, modelData{mm}, models{mm});
    end

    %% output alignment
    if params.align && isChar == 0
      [~, alignIdx] = getAlignWeights(attnInfos, data.srcLens, models, params);

      % we want to mimic the structure of allBestWords
      % size (beamSize * beamSize) x batchSize
      alignIdx = reshape(repmat(alignIdx, beamSize, 1), [], batchSize);
    end
    
    %% predict the next word
    % allBestScores, allBestWords should have size beamSize * (beamSize*batchSize)
    if params.forceDecoder
      assert(isChar == 0);
      [allBestScores, allBestWords, otherData] = nextBeamStep(models, beamStates, beamSize, 0, isChar, params.tgtUnk, params.unkDiscount, data.tgtOutput(:, tgtPos));
      otherInfo.forceDecodeOutputs(tgtPos, :) = otherData.maxWords;
    else
      [allBestScores, allBestWords] = nextBeamStep(models, beamStates, beamSize, 0, isChar, params.tgtUnk, params.unkDiscount);
    end

    % allBestWords, allBestScores should have size: (beamSize*beamSize) * batchSize
    [allBestScores, allBestWords, indices] = addSortScores(allBestScores, allBestWords, beamScores, beamSize, batchSize);

    %% build new beam
    for sentId=1:batchSize
      rowIndices = indices(:, sentId)';
      bestWords = allBestWords(rowIndices, sentId);
      
      % get candidates
      selectedIndices = find(bestWords~=tgtEos, beamSize);
      if ~isempty(selectedIndices)
        % update words
        startId = (sentId-1)*beamSize+1;
        endId = sentId*beamSize;
        sentWords = bestWords(selectedIndices);
        beamWords(startId:endId) = sentWords;

        % align
        if params.align && isChar == 0
          bestAlignIds = alignIdx(rowIndices, sentId);
          beamAlignIds(startId:endId) = bestAlignIds(selectedIndices);
        end

        % update beam
        sentBeamIndices = floor((rowIndices(selectedIndices)-1)/beamSize) + 1;
        beamIndices(startId:endId) = sentBeamIndices;

        % update scores
        beamScores(startId:endId) = allBestScores(selectedIndices, sentId);

        % get words that are eos and ranked before the last hypothesis in the next beam
        endIndices = find(bestWords(1:selectedIndices(end))==tgtEos); 
      else % special case, happen for beamSize = 1, useful for force decoding
        assert(beamSize == 1);
        assert(bestWords == tgtEos);
        if doneFlags(sentId) == 1
          continue;
        else
          doneFlags(sentId) = 1;
          endIndices = 1;
        end
      end
      
      %% store translations
      if ~isempty(endIndices) && (sentPos+1)>=minLen % we don't want to start recording very short translations
        numTranslations = length(endIndices);
        eosBeamIndices = floor((rowIndices(endIndices)-1)/beamSize) + 1;
        histIndices = (sentId-1)*beamSize + eosBeamIndices;
        
        transScores = allBestScores(endIndices, sentId);
        for ii=1:numTranslations
          if numDecoded(sentId)<stackSize || transScores(ii) > candScores(stackSize, sentId) % haven't collected enough translations or better than the last
            if numDecoded(sentId)<stackSize
              numDecoded(sentId) = numDecoded(sentId) + 1;
            end
            
            candScores(numDecoded(sentId), sentId) = transScores(ii);
            candidates{sentId}{numDecoded(sentId)} = [beamHistory(1:sentPos, histIndices(ii)); tgtEos];
            
            % align
            if params.align && isChar == 0
              alignInfo{sentId}{numDecoded(sentId)} = [alignHistory(1:sentPos, histIndices(ii)); beamAlignIds(histIndices(ii))];
            end

            % char
            if params.charTgtGen && isChar == 0
              for mm=1:numModels % copy from existing states
                otherInfo.transStates{numDecoded(sentId), mm}(:, 1:sentPos, sentId) = beamHistTopStates{mm}(:, histIndices(ii), 1:sentPos);
              end
            end
          end
        end
      end
    end
    
    %% update history
    % overwrite previous history
    colIndices = (sentIndices-1)*beamSize + beamIndices;
    beamHistory(1:sentPos, :) = beamHistory(1:sentPos, colIndices); 
    beamHistory(sentPos+1, :) = beamWords;
    % char
    if params.charTgtGen && isChar == 0
      for mm=1:numModels
        beamHistTopStates{mm}(:, :, 1:sentPos) = beamHistTopStates{mm}(:, colIndices, 1:sentPos);
        if models{mm}.params.charFeedOpt
          beamHistTopStates{mm}(:, :, sentPos+1) = hiddenLayerForward(models{mm}.W_h_char, attnInfos{mm}.input(:, colIndices), params.nonlinear_f); 
        else
          beamHistTopStates{mm}(:, :, sentPos+1) = beamStates{mm}{end}.softmax_h(:, colIndices);
        end
      end
    end
    
    % align
    if params.align && isChar == 0
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
  end % for sentPos
  
  for sentId=1:batchSize
    if numDecoded(sentId) == 0 % no translations found, use what we have in the beam
      fprintf(2, '  ! Sent %d: no translations end in %s, appending.\n', originalSentIndices(sentId), params.tgtVocab{tgtEos});
      for bb = 1:beamSize
        eosIndex = (sentId-1)*beamSize + bb;
        
        if numDecoded(sentId)<stackSize || beamScores(eosIndex) > candScores(stackSize, sentId) % haven't collected enough translations or better than the last
          if numDecoded(sentId)<stackSize
            numDecoded(sentId) = numDecoded(sentId) + 1;
          end
          
          candidates{sentId}{numDecoded(sentId)} = [beamHistory(1:maxLen, eosIndex); tgtEos]; % append eos at the end
          candScores(numDecoded(sentId), sentId) = beamScores(eosIndex);
          fprintf('  %.2f, num decoded %d:', candScores(numDecoded(sentId), sentId), numDecoded(sentId));
          fprintf(' %s', candidates{sentId}{numDecoded(sentId)});
          fprintf('\n');
          
          % align
          if params.align && isChar == 0
            if params.isReverse
              alignInfo{sentId}{numDecoded(sentId)} = [alignHistory(1:maxLen, eosIndex); 1];
            else
              alignInfo{sentId}{numDecoded(sentId)} = [alignHistory(1:maxLen, eosIndex); modelData{1}.srcLens(sentId)-1];
            end
          end

          % char
          if params.charTgtGen && isChar == 0
            for mm=1:numModels
              otherInfo.transStates{numDecoded(sentId), mm}(:, 1:maxLen-1, sentId) = beamHistTopStates{mm}(:, eosIndex, 1:maxLen-1);
            end
          end 
        end
      end % for bb
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
function [bestLogProbs, bestWords, otherData] = nextBeamStep(models, lastDecStates, beamSize, ignoreSymbol, isChar, tgtUnk, unkDiscount, varargin)
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
    correctWords = reshape(varargin{1}, 1, []);
    bestWords = correctWords;
    bestLogProbs = logProbs(sub2ind(size(logProbs), correctWords, 1:size(logProbs, 2)));
    [~, otherData.maxWords] = max(logProbs, [], 1);
  else
    [sortedLogProbs, sortedWords] = sort(logProbs, 'descend');
    bestWords = sortedWords(1:beamSize, :);
    bestLogProbs = sortedLogProbs(1:beamSize, :);
    otherData = [];
    
    if ignoreSymbol>0
      flags = bestWords == ignoreSymbol; 
      cols = find(any(flags)); % mark which column has the ignoreSymbol
      if ~isempty(cols)
        indices = find(flags);
        assert(length(indices) == length(cols)); % each col has a single ignoreSymbol
        bestWords(indices) = sortedWords(beamSize+1, cols);
        bestLogProbs(indices) = sortedLogProbs(beamSize+1, cols);
      end
    end
    
    if isChar==0 && unkDiscount>0
      flags = bestWords == tgtUnk; 
      bestLogProbs(flags) = bestLogProbs(flags) - unkDiscount;
    end
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
