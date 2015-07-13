%%
%
% Decode from an LSTM model.
%   stackSize: the maximum number of translations we want to get.
% Output:
%   - candidates: list of candidates
%   - candScores: score of the corresponding candidates (stackSize * batchSize)
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%   With help from Hieu Pham.
%
%%
function [candidates, candScores, alignInfo] = lstmDecoder(models, data, params)
  % backward compatibility: not a cell, single model, put into a cell format
  if ~iscell(models)
    tmpModels = cell(1, 1);
    tmpModels{1} = models;
    tmpModels{1}.params = params;
    models = tmpModels;
  end
  
  beamSize = params.beamSize;
  stackSize = params.stackSize;
  
  input = data.input;
  inputMask = data.inputMask; 
  srcMaxLen = data.srcMaxLen;
  batchSize = size(input, 1);
  data.curBatchSize = batchSize;
  if params.attnFunc>0
    params.numSrcHidVecs = srcMaxLen-1;
    
    if params.attnGlobal && params.attnOpt>0 % global, content-based alignments
      params.numAttnPositions = params.numSrcHidVecs;
    end
  end
  
  printSent(2, input(1, 1:srcMaxLen-1), params.srcVocab, '  src: ');
  if params.isReverse
    printSent(2, input(1, srcMaxLen-1:-1:1), params.srcVocab, ' rsrc: ');
  end
  printSent(2, input(1, srcMaxLen+1:end), params.tgtVocab, '  tgt: ');
      
  %% init
  fprintf(2, '# Decoding batch of %d sents, srcMaxLen=%d, %s\n', batchSize, srcMaxLen, datestr(now));
  fprintf(params.logId, '# Decoding batch of %d sents, srcMaxLen=%d, %s\n', batchSize, srcMaxLen, datestr(now));
  
  curMask.mask = ones(1, batchSize);
  curMask.unmaskedIds = 1:batchSize;
  curMask.maskedIds = [];
  
  %%%%%%%%%%%%
  %% encode %%
  %%%%%%%%%%%%
  %% multiple models
  numModels = length(models);
  lstm = cell(numModels, 1); % lstm can be over written, as we do not need to backprop
  softmax_h = cell(numModels, 1);
  W = cell(numModels, 1);
  W_emb = cell(numModels, 1);
  modelData = cell(numModels, 1);
  zeroStates = cell(numModels, 1);
  for mm=1:numModels
    lstm{mm} = cell(models{mm}.params.numLayers, 1);
    zeroStates{mm} = zeroMatrix([models{mm}.params.lstmSize, batchSize], params.isGPU, params.dataType);
    softmax_h{mm} = zeroStates{mm};
    modelData{mm} = data;
    models{mm}.params.curBatchSize = batchSize;
    
    % attention
    if models{mm}.params.attnFunc || models{mm}.params.sameLength
      models{mm}.params.numSrcHidVecs = params.numSrcHidVecs;
      models{mm}.params.numSrcHidVecs = params.numSrcHidVecs;
      if params.attnGlobal && params.attnOpt>0 % global, content-based alignments
        models{mm}.params.numAttnPositions = params.numAttnPositions;
      end
      modelData{mm}.curMask = curMask;

      modelData{mm}.srcHidVecsOrig = zeroMatrix([models{mm}.params.lstmSize, batchSize, models{mm}.params.numSrcHidVecs], params.isGPU, params.dataType);  
      if models{mm}.params.attnGlobal %&& (models{mm}.params.attnOpt==1 || models{mm}.params.attnOpt==2)
        % modelData{mm}.alignMask = modelData{mm}.srcMask(:, 1:models{mm}.params.numSrcHidVecs)'; % numSrcHidVecs * curBatchSize
        modelData{mm}.alignMask = oneMatrix([models{mm}.params.numSrcHidVecs, models{mm}.params.curBatchSize], params.isGPU, params.dataType);
        modelData{mm}.srcMaskedIds = [];
      end
    else
      models{mm}.params.numSrcHidVecs = 0;
    end
    
    W{mm} = models{mm}.W_src;
    if models{mm}.params.tieEmb % tie embeddings
      W_emb{mm} = models{mm}.W_emb_tie;
    else
      W_emb{mm} = models{mm}.W_emb_src;
    end
  end
  
  if params.align
    alignWeights = cell(batchSize, 1);
    for sentId=1:batchSize % init
      alignWeights{sentId} = zeroMatrix([data.srcLens(sentId)-1, 1], params.isGPU, params.dataType); % ignore eos
    end
  end
  
  for tt=1:srcMaxLen % time
    maskedIds = find(~inputMask(:, tt)); % curBatchSize * 1
    tgtPos = tt-srcMaxLen+1; % = 1
    if tt==srcMaxLen % due to implementation in lstmCostGrad, we have to switch to W_tgt here. THIS IS VERY IMPORTANT!
      for mm=1:numModels
        W{mm} = models{mm}.W_tgt;
        if models{mm}.params.tieEmb==0
          W_emb{mm} = models{mm}.W_emb_tgt;
        end
      end
    end
    
    for mm=1:numModels % model
      for ll=1:models{mm}.params.numLayers % layer
        % previous-time input
        if tt==1 % first time step
          h_t_1 = zeroStates{mm};
          c_t_1 = zeroStates{mm};
        else
          h_t_1 = lstm{mm}{ll}.h_t; 
          c_t_1 = lstm{mm}{ll}.c_t;
        end

        % current-time input
        if ll==1 % first layer
          if tt==srcMaxLen % decoder input
            x_t = getLstmDecoderInput(input(:, tt), tgtPos, W_emb{mm}, softmax_h{mm}, modelData{mm}, zeroStates{mm}, models{mm}.params); %, curMask);
          else
            x_t = W_emb{mm}(:, input(:, tt));
          end
        else % subsequent layer, use the previous-layer hidden state
          x_t = lstm{mm}{ll-1}.h_t;
        end

        % masking
        x_t(:, maskedIds) = 0; 
        h_t_1(:, maskedIds) = 0;
        c_t_1(:, maskedIds) = 0;

        % lstm cell
        [lstm{mm}{ll}, h_t, c_t] = lstmUnit(W{mm}{ll}, x_t, h_t_1, c_t_1, ll, tt, srcMaxLen, models{mm}.params, 1);
        lstm{mm}{ll}.h_t = h_t;
        lstm{mm}{ll}.c_t = c_t;

        % attention
        if tt<=models{mm}.params.numSrcHidVecs && ll==models{mm}.params.numLayers %&& (models{mm}.params.attnFunc>0 || models{mm}.params.sameLength==1)
          modelData{mm}.srcHidVecsOrig(:, :, tt) = h_t;

          % done generating all srcHidVecs, collect
          if tt==models{mm}.params.numSrcHidVecs
            [modelData{mm}] = updateDataSrcVecs(modelData{mm}, models{mm}.params);
          end
        end
        
        % h_t -> softmax_h
        if tt==srcMaxLen && ll==models{mm}.params.numLayers
          modelData{mm}.posMask = curMask;
          if models{mm}.params.attnFunc
            [softmax_h{mm}, h2sInfo] = attnLayerForward(h_t, models{mm}.params, models{mm}, modelData{mm}, tgtPos); 
          else
            softmax_h{mm} = h_t;
          end
          
          
          % output alignment
          if params.align 
            if models{mm}.params.attnGlobal==0 % local
              [startIds, endIds, startAttnIds, endAttnIds] = computeAttnBound(h2sInfo.srcPositions, models{mm}.params);
            end
            
            for sentId=1:batchSize % go through each sent
              srcLen = modelData{mm}.srcLens(sentId);
              
              if models{mm}.params.attnGlobal
                alignWeights{sentId} = alignWeights{sentId} + h2sInfo.alignWeights(end-srcLen+2:end, sentId);
              else
                if startIds(sentId)<=endIds(sentId)
                  offset = srcMaxLen-srcLen;
                  indices = startAttnIds(sentId)-offset:endAttnIds(sentId)-offset;
                  alignWeights{sentId}(indices) = alignWeights{sentId}(indices) + h2sInfo.alignWeights(startIds(sentId):endIds(sentId), sentId);
                end
              end
            end
            
            if mm==numModels
              firstAlignIdx = zeroMatrix([1, batchSize], params.isGPU, params.dataType);
              for sentId=1:batchSize % go through each sent
                [~, firstAlignIdx(sentId)] = max(alignWeights{sentId}, [], 1); % srcLen includes eos, alignWeights excludes eos.
              end
            end % end if last model
          end
        end

        % assert
        if params.assert
          assert(sum(sum(abs(lstm{mm}{ll}.c_t(:, maskedIds))))<1e-5);
          assert(sum(sum(abs(lstm{mm}{ll}.h_t(:, maskedIds))))<1e-5);
        end
      end % end for layer
    end % end for model
  end % end for time
  
  %%%%%%%%%%%%
  %% decode %%
  %%%%%%%%%%%%
  startTime = clock;
 
  if params.sameLength % same-length decoding
    minLen = srcMaxLen;
    maxLen = minLen; % inclusive of <eos>
  else
    if batchSize==1
      minLen = floor(srcMaxLen*params.minLenRatio);
    else
      minLen = 2;
    end
    maxLen = floor(srcMaxLen*params.maxLenRatio);
  end
  
  sentIndices = data.startId:(data.startId+batchSize-1);
  [candidates, candScores, alignInfo] = decodeBatch(models, params, lstm, softmax_h, minLen, maxLen, beamSize, stackSize, batchSize, sentIndices, srcMaxLen, modelData, zeroStates, firstAlignIdx);
  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  fprintf(2, '  Done, minLen=%d, maxLen=%d, speed %f sents/s, time %.0fs, %s\n', minLen, maxLen, batchSize/timeElapsed, timeElapsed, datestr(now));
  fprintf(params.logId, '  Done, minLen=%d, maxLen=%d, speed %f sents/s, time %.0fs, %s\n', minLen, maxLen, batchSize/timeElapsed, timeElapsed, datestr(now));
end

%%
% Beam decoder from an LSTM model, works for multiple sentences
% Input:
%   - encoded vector of the source sentences
%   - maximum length willing to go
%   - beamSize
%   - stackSize: maximum number of translations collected for one example
%%
function [candidates, candScores, alignInfo] = decodeBatch(models, params, lstmStart, softmax_h, minLen, maxLen, beamSize, stackSize, batchSize, originalSentIndices, srcMaxLen, modelData, zeroStates, firstAlignIdx)
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
  end
  
  
  %% first prediction
  numModels = length(models);
  [scores, words] = nextBeamStep(models, softmax_h, beamSize); %lstmStart{numLayers}.h_t, beamSize, params, data, curMask, tgtPos); % scores, words: beamSize * batchSize

  % TODO: by right, we should filter out words == params.tgtEos, but I
  % think for good models, we don't have to worry :)
  
  %% matrix dimension note
  % note that we order matrices in the following dimension: n * numElements
  % columns correspond to the 1st sent go first, then the 2nd one, until the batchSize-th sent.
  sentIndices = repmat(1:batchSize, beamSize, 1);
  sentIndices = sentIndices(:)'; % 1 ... 1, 2 ... 2, ...., batchSize ... batchSize . 1 * (beamSize*batchSize)
  
  %% init beam
  beamScores = scores(:)'; % 1 * numElements
  beamHistory = zeroMatrix([maxLen, numElements], params.isGPU, params.dataType); % maxLen * (numElements) 
  beamHistory(1, :) = words(:); % words for sent 1 go together, then sent 2, ...
  % align
  if params.align
    alignHistory = zeroMatrix([maxLen, numElements], params.isGPU, params.dataType); % maxLen * (numElements) 
    alignHistory(1, :) = firstAlignIdx;
  end
  
  % replicate
  beamStates = cell(numModels, 1);
  for mm=1:numModels % model    
    beamStates{mm} = cell(models{mm}.params.numLayers, 1);
    for ll=1:models{mm}.params.numLayers % lstmSize * numElements
      beamStates{mm}{ll}.c_t = reshape(repmat(lstmStart{mm}{ll}.c_t, beamSize, 1),  models{mm}.params.lstmSize, numElements); 
      beamStates{mm}{ll}.h_t = reshape(repmat(lstmStart{mm}{ll}.h_t, beamSize, 1),  models{mm}.params.lstmSize, numElements); 
    end
    
    softmax_h{mm} = reshape(repmat(softmax_h{mm}, beamSize, 1),  models{mm}.params.lstmSize, numElements); 
  end
  
  
  curMask.mask = ones(1, batchSize);
  curMask.unmaskedIds = 1:batchSize;
  curMask.maskedIds = [];
  % attentional / positional models
  for mm=1:numModels % model
    if models{mm}.params.attnFunc || models{mm}.params.sameLength % || models{mm}.params.posModel>=2 
      curMask.mask = ones(1, numElements);
      curMask.unmaskedIds = 1:numElements;
      models{mm}.params.curBatchSize = numElements;
    
    
      modelData{mm}.curBatchSize = numElements;
      modelData{mm}.srcLens = reshape(repmat(modelData{mm}.srcLens, beamSize, 1), 1, []);
      
      % duplicate srcHidVecs
      if models{mm}.params.attnGlobal % soft, global
        modelData{mm}.absSrcHidVecs = duplicateSrcHidVecs(modelData{mm}.absSrcHidVecs, batchSize, beamSize);

        if models{mm}.params.attnOpt==1 || models{mm}.params.attnOpt==2
           % alignMask: batchSize * numSrcHidVecs
           % alignMask: numSrcHidVecs * (batchSize*beamSize), mask columns of the same sentence are nearby
           modelData{mm}.alignMask = reshape(repmat(modelData{mm}.alignMask, 1, beamSize)', models{mm}.params.numSrcHidVecs, numElements);
           modelData{mm}.srcMaskedIds = find(modelData{mm}.alignMask==0);
        end
      else % hard, local
        modelData{mm}.srcHidVecs = duplicateSrcHidVecs(modelData{mm}.srcHidVecsOrig, batchSize, beamSize);
      end
    end
  end
  
  decodeCompleteCount = 0; % count how many sentences we have completed collecting the translations
  beamWords = zeroMatrix([1, numElements], params.isGPU, params.dataType);
  beamIndices = zeroMatrix([1, numElements], params.isGPU, params.dataType);
  % align
  if params.align
    beamAlignIds = zeroMatrix([1, numElements], params.isGPU, params.dataType);
  end
  
  W = cell(numModels, 1);
  W_emb = cell(numModels, 1);
  for mm=1:numModels % model
    if params.sameLength
      zeroStates{mm} = zeroMatrix([models{mm}.params.lstmSize, numElements], params.isGPU, params.dataType);
    end
    
    if models{mm}.params.tieEmb
      W_emb{mm} = models{mm}.W_emb_tie;
    else
      W_emb{mm} = models{mm}.W_emb_tgt;
    end
  end

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
    words = beamHistory(sentPos, :);
    
    
  
    if params.align
      alignWeights = cell(numElements, 1);
      for sentId=1:numElements % init
        alignWeights{sentId} = zeroMatrix([modelData{1}.srcLens(sentId)-1, 1], params.isGPU, params.dataType); % ignore eos
      end
    end
    
    for mm=1:numModels % model
      for ll = 1 : models{mm}.params.numLayers
        W{mm} = models{mm}.W_tgt{ll};
        % current input
        if ll == 1
          x_t = getLstmDecoderInput(words, tgtPos, W_emb{mm}, softmax_h{mm}, modelData{mm}, zeroStates{mm}, models{mm}.params); %, curMask);
        else
          x_t = beamStates{mm}{ll-1}.h_t;
        end
        % previous input
        h_t_1 = beamStates{mm}{ll}.h_t;
        c_t_1 = beamStates{mm}{ll}.c_t;

        [beamStates{mm}{ll}, h_t, c_t] = lstmUnit(W{mm}, x_t, h_t_1, c_t_1, ll, srcMaxLen+sentPos, srcMaxLen, models{mm}.params, 1);
        beamStates{mm}{ll}.h_t = h_t;
        beamStates{mm}{ll}.c_t = c_t;

        % h_t -> softmax_h
        if ll==models{mm}.params.numLayers
          modelData{mm}.posMask = curMask;

          if params.attnFunc
            [softmax_h{mm}, h2sInfo] = attnLayerForward(h_t, models{mm}.params, models{mm}, modelData{mm}, tgtPos); 
          else
            softmax_h{mm} = h_t;
          end
          
          % align
          if params.align
            if models{mm}.params.attnGlobal==0 % local
              [startIds, endIds, startAttnIds, endAttnIds] = computeAttnBound(h2sInfo.srcPositions, models{mm}.params);
            end
            
            for sentId=1:numElements % go through each sent
              srcLen = modelData{mm}.srcLens(sentId);
              
              if models{mm}.params.attnGlobal
                alignWeights{sentId} = alignWeights{sentId} + h2sInfo.alignWeights(end-srcLen+2:end, sentId);
              else
                if startIds(sentId)<=endIds(sentId)
                  offset = srcMaxLen-srcLen;
                  indices = startAttnIds(sentId)-offset:endAttnIds(sentId)-offset;
                  alignWeights{sentId}(indices) = alignWeights{sentId}(indices) + h2sInfo.alignWeights(startIds(sentId):endIds(sentId), sentId);
                end
              end
            end
          
            if mm==numModels
              alignIdx = zeroMatrix([1, numElements], params.isGPU, params.dataType);
              for sentId=1:numElements % go through each sent
                [~, alignIdx(sentId)] = max(alignWeights{sentId}, [], 1); % srcLen includes eos, alignWeights excludes eos.
              end

              % we want to mimic the structure of allBestWords later on of size (beamSize * beamSize) x 1
              alignIdx = reshape(repmat(alignIdx, beamSize, 1), [], 1);
            end
          end
        end
      end
    end
    
    %% predict the next word

    if params.sameLength && sentPos == (maxLen-1) % same length decoding
      [~, ~, logProbs] = nextBeamStep(models, softmax_h, beamSize);
      allBestScores = logProbs(params.tgtEos, :);
      allBestWords = params.tgtEos*ones(1, beamSize*batchSize);
      card = 1;
    else
      [allBestScores, allBestWords] = nextBeamStep(models, softmax_h, beamSize);
      card = beamSize;
    end
    [allBestScores, allBestWords, indices] = addSortScores(allBestScores, allBestWords, beamScores, card, beamSize, batchSize);
    
    
%     beamScores
%     params.vocab(beamHistory(1:sentPos, :))
%     params.vocab(allBestWords)

    %% build new beam
    for sentId=1:batchSize
      rowIndices = indices(:, sentId)';
      bestWords = allBestWords(rowIndices, sentId);
      
      % get candidates
      if params.sameLength==1 && sentPos==(maxLen-1)% same-length decoding, already choose tgtEos
        selectedIndices = 1:beamSize; 
      else
        selectedIndices = find(bestWords~=params.tgtEos, beamSize);
      end

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
            
      %% store translations
      if params.sameLength==0 || sentPos==(maxLen-1) % for sameLength decoding, we only collect output at (maxLen-1)
        if params.sameLength % for same length decoding, we already choose tgtEos
          endIndices = selectedIndices;
        else
          endIndices = find(bestWords(1:selectedIndices(end))==params.tgtEos); % get words that are eos and ranked before the last hypothesis in the next beam
        end
        
        if ~isempty(endIndices) && (sentPos+1)>=minLen % we don't want to start recording very short translations
          numTranslations = length(endIndices);
          eosBeamIndices = floor((rowIndices(endIndices)-1)/beamSize) + 1;
          translations = beamHistory(1:sentPos, (sentId-1)*beamSize + eosBeamIndices);
          % align
          if params.align
            alignments = alignHistory(1:sentPos, (sentId-1)*beamSize + eosBeamIndices);
            lastAlignIds = beamAlignIds((sentId-1)*beamSize + eosBeamIndices);
          end
          
          transScores = allBestScores(endIndices, sentId);
          for ii=1:numTranslations
            if numDecoded(sentId)<stackSize % haven't collected enough translations
              numDecoded(sentId) = numDecoded(sentId) + 1;

              candidates{sentId}{numDecoded(sentId)} = [translations(:, ii); params.tgtEos];
              candScores(numDecoded(sentId), sentId) = transScores(ii);

              % align
              if params.align
                alignInfo{sentId}{numDecoded(sentId)} = [alignments(:, ii); lastAlignIds(ii)];
              end
              
              %printSent(2, translations(:, ii), params.vocab, ['  trans sent ' num2str(originalSentIndices(sentId)) ', ' num2str(transScores(ii)), ': ']);
              if numDecoded(sentId)==stackSize % done for sentId
                decodeCompleteCount = decodeCompleteCount + 1;
                break;
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
      softmax_h{mm} = softmax_h{mm}(:, colIndices);
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

function [allBestScores, allBestWords, indices] = addSortScores(allBestScores, allBestWords, beamScores, card, beamSize, batchSize)
  % use previous beamScores, 1 * (beamSize*batchSize), update along the first dimentions
  allBestScores = bsxfun(@plus, allBestScores, beamScores);
  allBestScores = reshape(allBestScores, [card*beamSize, batchSize]);
  allBestWords = reshape(allBestWords, [card*beamSize, batchSize]);

  % for each sent, select the best beamSize candidates, out of card*beamSize ones
  [allBestScores, indices] = sort(allBestScores, 'descend'); % (card*beamSize) * batchSize
end

%%
% return bestLogProbs, bestWords of sizes beamSize * curBatchSize
%%
function [bestLogProbs, bestWords, logProbs, sortedLogProbs, sortedWords] = nextBeamStep(models, softmax_h, beamSize) %h_t, beamSize, params, data, curMask, tgtPos)
  softmax_input = models{1}.W_soft*softmax_h{1};
  if length(models)>1 % aggregate predictions from multiple models
    for ii=2:length(models)
      softmax_input = softmax_input + models{ii}.W_soft*softmax_h{ii};
    end
    softmax_input = softmax_input./length(models);
  end
  [logProbs] = softmaxDecode(softmax_input);
  
  % sort
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

%%%%%%%%%%%%%
%             % h_t -> scales=sigmoid(v_pos*h_pos) in [0, 1]
%             scales = scaleLayerForward(models{mm}.W_pos, models{mm}.v_pos, h_t, models{mm}.params);
%             modelData{mm}.positions = floor(modelData{mm}.srcLens.*scales);

% function [srcHidVecs] = computeRelativeSrcHidVecs(srcHidVecsOrig, srcMaxLen, tgtPos, batchSize, params, beamSize)
%   [srcHidVecs] = buildSrcHidVecs(srcHidVecsOrig, srcMaxLen, tgtPos, params);
% 
%   % duplicate srcHidVecs along the curBatchSize dimension beamSize times
%   if (beamSize>1)
%     [srcHidVecs] = duplicateSrcHidVecs(srcHidVecs, batchSize, beamSize);
%   end
% end

%         % masking
%         %[data.posMask] = createPosMask(tgtPos, params, data, curMask);
% 
%         if params.predictPos==1 % regression
%           % h_t -> scales=sigmoid(v_pos*h_pos) in [0, 1]
%           scales = scaleLayerForward(model.W_pos, model.v_pos, h_t, params);
%           data.positions = floor(data.srcLens.*scales);
%           data.posMask = curMask;
%         elseif params.predictPos==2 % classification
%           % h_t -> h_pos=f(W_pos*h_t)
%           h_pos = hiddenLayerForward(model.W_pos, h_t, params.nonlinear_f);
% 
%           % h_pos -> predictions
%           [probs] = softmax(model.W_softPos*h_pos, curMask.mask);
%   
%           [~, maxIndices] = max(probs, [], 1);
%           curPosOutput = maxIndices + params.startPosId-1;
%           data.posMask.mask = curMask.mask & curPosOutput~=params.nullPosId & curPosOutput~=params.tgtEos;
%           data.posMask.unmaskedIds = find(data.posMask.mask);
%           data.posMask.maskedIds = find(~data.posMask.mask);
%   
%           data.positions = (tgtPos+params.zeroPosId) - curPosOutput; %  = tgtPos - (curPosOutput-zeroPosId)  = srcPos = tgtPos - relative distance
%         end          

%         % pos masking. TODO: FIX HERE for attn 3, 4, ...
%         %[data.posMask] = createPosMask(tgtPos, params, data, curMask);

%         if params.predictPos==1 % regression
%           
%         elseif params.predictPos==2 % classification
%           % h_t -> h_pos=f(W_pos*h_t)
%           h_pos = hiddenLayerForward(model.W_pos, h_t, params.nonlinear_f);
% 
%           % h_pos -> predictions
%           [probs] = softmax(model.W_softPos*h_pos, curMask.mask);
%   
%           [~, maxIndices] = max(probs, [], 1);
%           curPosOutput = maxIndices + params.startPosId-1;
%           data.posMask.mask = curMask.mask & curPosOutput~=params.nullPosId & curPosOutput~=params.tgtEos;
%           data.posMask.unmaskedIds = find(data.posMask.mask);
%           data.posMask.maskedIds = find(~data.posMask.mask);
%           
%           data.positions = (tgtPos+params.zeroPosId) - curPosOutput; %  = tgtPos - (curPosOutput-zeroPosId)  = srcPos = tgtPos - relative distance
%         end          


%     % duplicate srcLens
%     if params.posModel>=2
%       params.curBatchSize = batchSize;
%       data.srcLens = reshape(repmat(data.srcLens, beamSize, 1), 1, []);
%       data.srcHidVecs = duplicateSrcHidVecs(data.srcHidVecsOrig, batchSize, beamSize);
%     end

%   if params.depParse % dependency parsing
%     minLen = (srcMaxLen-1)*2; % srcMaxLen include eos, we want our dependency parse will have length = 2 * input (no eos) length 
%     maxLen = minLen;
%   else

%   if params.depParse % dependency parsing, the first symbol needs to be S
%     [~, ~, logProbs] = nextBeamStep(model, softmax_h, beamSize); %lstmStart{numLayers}.h_t, beamSize, params, data, curMask, tgtPos);
%     scores = repmat(logProbs(params.depShiftId, :), beamSize, 1);
%     words = params.depShiftId*ones(1, beamSize*batchSize);
%     
%     assert(batchSize==1);
%     srcLen = data.srcLens-1;
%     stackCounts = 2*ones(numElements, 1); % at first, all hypotheses have R(root) and the first word in the stack
%     bufferCounts = (srcLen - 1)*ones(numElements, 1); % don't count eos, and minus the first word
%     shiftCounts = ones(numElements, 1);
%   else
%     
%   end

%       if params.depParse % dependency parsing
%         if sentPos<(maxLen-1) % exclude eos and R(root)
%           % because of shiftIndices, we know that only the following top choices are valid: 
%           %   beamSize*(beamSize-mustShiftCount) for non-shift operators + 
%           %   mustShiftCount operators.
%           validCount = beamSize*(beamSize-mustShiftCount) + mustShiftCount - noshiftCount;
%           selectedIndices = find(bestWords(1:validCount)~=params.tgtEos & bestWords(1:validCount)~=params.depRootId, beamSize);
%         else % we already chose R(root) above
%           assert(length(bestWords)==beamSize);
%           assert(isempty(find(bestWords~=params.depRootId, 1)));
%           selectedIndices = 1:beamSize;
%           %selectedIndices = find(bestWords==params.depRootId, beamSize);
%         end
%       else

%       if params.depParse % dependency parsing
%         % shift: increase stack count, decrease buffer count
%         shiftIds = find(sentWords == params.depShiftId);
%         shiftCounts(shiftIds) = shiftCounts(sentBeamIndices(shiftIds)) + 1;
%         
%         % assert: we should not shift more than the number of words present
%         assert(isempty(find(shiftCounts>=data.srcLens,1)));
%         
%         oldStackCounts = stackCounts;
%         stackCounts(shiftIds) = stackCounts(sentBeamIndices(shiftIds)) + 1;
%         bufferCounts(shiftIds) = bufferCounts(sentBeamIndices(shiftIds)) - 1;
%         
%         % not shift: decrease stack count
%         noshiftIds = find(sentWords ~= params.depShiftId);
%         stackCounts(noshiftIds) = oldStackCounts(sentBeamIndices(noshiftIds)) -1; % Important: to use oldStackCounts here
%       end

%       if (params.depParse==0 && params.sameLength==0) || sentPos==(maxLen-1) % for dependency parsing, we only collect output at maxLen, i.e. end at R(root), then append eos.
%         if params.depParse || params.sameLength % dependency parsing, we already chose R(root) / for same length decoding, we already choose tgtEos
%           endIndices = selectedIndices; %find(bestWords==params.depRootId);
%         else
%           endIndices = find(bestWords(1:selectedIndices(end))==params.tgtEos); % get words that are eos and ranked before the last hypothesis in the next beam
%         end

%               if params.depParse % dependency parsing, append R(root) and eos
%                 candidates{sentId}{numDecoded(sentId)} = [translations(:, ii); params.depRootId; params.tgtEos];
%               else
%                 
%               end

%%%%%%%%%%%%%

%         % attention feed input model
%         if params.attnFeedInput && tt==srcMaxLen
%           if params.attnRelativePos % relative pos
%             [srcHidVecs] = computeRelativeSrcHidVecs(data.srcHidVecsOrig, srcMaxLen, tgtPos, batchSize, params, 1);
%           else
%             srcHidVecs = data.absSrcHidVecs;
%           end
%           
%           % attnForward: h_t -> attnVecs (used the previous hidden state
%           % here we use the top hidden state
%           [attnVecs] = attnLayerForward(model.W_a, lstm{params.numLayers}.h_t, srcHidVecs, data.curMask.mask);
%           x_t = [W_emb(:, input(:, tt)); attnVecs];
%         else

%         % attention model 3, 4
%         if params.attnFeedInput
%           if params.attnRelativePos % relative position
%             srcHidVecs = computeRelativeSrcHidVecs(data.srcHidVecsOrig, srcMaxLen, tgtPos, batchSize, params, 1);
%           else
%             srcHidVecs = data.absSrcHidVecs;
%           end
%           
%           % attnForward: h_t -> attnVecs
%           % here we use the top hidden state of the previous time step
%           [attnVecs] = attnLayerForward(model, beamStates{params.numLayers}.h_t, srcHidVecs, ones(1, batchSize*beamSize)); % here we use the previous time step mask
%           x_t = [W_emb(:, words); attnVecs];
%         else

%% Code for class-based softmax %%
%   if params.numClasses == 0 % normal softmax
%     [logProbs] = softmaxDecode(model.W_soft*softmax_h);
%   else
%     batch_size = size(softmax_h, 2);
%     [class_log_probs] = softmaxDecode(model.W_soft_class*softmax_h);
% 
%     if params.assert
%       assert(isempty( find( abs(sum(exp(class_log_probs))-1)>1e-8, 1 ) ), 'sum of class_probs is not one\n');
%     end
%       
%     % W_soft_inclass: classSize * lstmSize * numClasses
%     % softmax_h: lstmSize * batchSize
%     % build classSize * lstmSize * numClasses * batchSize, 
%     % then sum across lstmSize (dim 2)
%     % in_class_raws: classSize * numClasses * batchSize, 
%     in_class_raws = squeeze(sum(bsxfun(@times, permute(model.W_soft_inclass,[1 2 3 4]), permute(softmax_h,[3 1 4 2])), 2));
%     mx = max(in_class_raws, [], 1); % max along classSize (dim 1)
%     in_class_raws = bsxfun(@minus, in_class_raws, mx);
%     in_class_log_probs = bsxfun(@minus, in_class_raws, log(sum(exp(in_class_raws),1))); % sum along classSize (dim 1)
% 
%     % class_log_probs: numClasses * batchSize
%     % in_class_log_probs, total_log_probs: classSize * numClasses * batchSize
%     total_log_probs = bsxfun(@plus, permute(class_log_probs,[3 1 2]), in_class_log_probs);
%     logProbs = reshape(total_log_probs, params.classSize*params.numClasses, batch_size);
%     correct_order = repmat(params.classSize*(0:(params.numClasses-1))', [1 params.classSize]);
%     correct_order = bsxfun(@plus, correct_order, 1:params.classSize);
% 
%     logProbs = logProbs(correct_order(:),:);
%   end

%     if params.depParse % dependency parsing
%       if sentPos == (maxLen-1) % we want R(root) to be the next word
%         [~, ~, logProbs] = nextBeamStep(model, softmax_h, beamSize); %beamStates{numLayers}.h_t, beamSize, params, data, curMask, tgtPos);
%         allBestScores = logProbs(params.depRootId, :);
%         allBestWords = params.depRootId*ones(1, beamSize*batchSize);
%         card = 1;
%       else
%         [allBestScores, allBestWords, logProbs] = nextBeamStep(model, softmax_h, beamSize); %beamStates{numLayers}.h_t, beamSize, params, data, curMask, tgtPos); % beamSize * (beamSize*batchSize)
%         
%         %% NOTE: this code require batchSize to be 1.
%         % shift: when stack has one word and buffer is not empty
%         shiftIndices = find(stackCounts==1 & bufferCounts>0);
%         mustShiftCount = length(shiftIndices);
%         if mustShiftCount>0
%           allBestScores(1, shiftIndices) = logProbs(params.depShiftId, shiftIndices); 
%           allBestWords(1, shiftIndices) = params.depShiftId;
%           allBestScores(2:end, shiftIndices) = -1e10; % make the scores very small, so it won't make into the beam
%         end
%         
%         % noshift: when buffer is empty
%         noshiftIndices = find(bufferCounts==0);
%         noshiftCount = length(noshiftIndices);
%         if noshiftCount>0
%           assert(isempty(intersect(shiftIndices, noshiftIndices))==1);
%           linearIndices = find(allBestWords(:, noshiftIndices)==params.depShiftId);
%           if ~isempty(linearIndices)
%             [xIndices, yIndices] = ind2sub([beamSize, noshiftCount], linearIndices);
%             linearIndices = sub2ind(size(allBestScores), xIndices, noshiftIndices(yIndices));
%             allBestScores(linearIndices) = -1e10; % make the scores very small, so it won't make into the beam
%             
%             noshiftCount = length(linearIndices); % the actual number of shift opeators in the candidates.
%           end
%         end
%         
%         card = beamSize;
%       end
%     else

%% Unused %%  
%   % separate emb
%   if params.separateEmb==1 
%   else
%     W_emb = model.W_emb;
%   end
   

%       data.srcHidVecs = zeroMatrix([params.lstmSize, batchSize, params.numAttnPosi  tions], params.isGPU, params.dataType);
%       [startAttnId, endAttnId, startHidId, endHidId] = buildSrcHidVecs(srcMaxLen, tgtPos, params);
%       data.srcHidVecs(:, :, startHidId:endHidId) = data.srcHidVecsOrig(:, :, startAttnId:endAttnId);
%       
%       % duplicate srcHidVecs along the curBatchSize dimension beamSize times
%       data.srcHidVecs = permute(data.srcHidVecs, [1, 3, 2]); % lstmSize * numAttnPositions * batchSize
%       data.srcHidVecs = reshape(data.srcHidVecs, params.lstmSize*params.numAttnPositions, batchSize);
%       data.srcHidVecs = repmat(data.srcHidVecs, beamSize, 1);
%       data.srcHidVecs = reshape(data.srcHidVecs, params.lstmSize, params.numAttnPositions, numElements);
%       data.srcHidVecs = permute(data.srcHidVecs, [1, 3, 2]); % lstmSize * batchSize * numAttnPositions
