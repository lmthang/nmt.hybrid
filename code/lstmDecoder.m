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

%   printSent(2, input(1, 1:srcMaxLen-1), params.srcVocab, '  src: ');
%   if params.isReverse
%     printSent(2, input(1, srcMaxLen-1:-1:1), params.srcVocab, ' rsrc: ');
%   end
%   printSent(2, input(1, srcMaxLen+1:end), params.tgtVocab, '  tgt: ');
      
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
  firstAlignIdx = [];
  for mm=1:numModels
    lstm{mm} = cell(models{mm}.params.numLayers, 1);
    zeroStates{mm} = zeroMatrix([models{mm}.params.lstmSize, batchSize], params.isGPU, params.dataType);
    softmax_h{mm} = zeroStates{mm};
    modelData{mm} = data;
    models{mm}.params.curBatchSize = batchSize;
    
    % attention
    if models{mm}.params.attnFunc
      models{mm}.params.numSrcHidVecs = srcMaxLen - 1;
      if models{mm}.params.attnGlobal
        if models{mm}.params.attnOpt==0 % for attnOpt==1, we use variable-length alignment vectors
          models{mm}.params.numAttnPositions = models{mm}.params.maxSentLen-1;
        else % global, content-based alignments
          models{mm}.params.numAttnPositions = models{mm}.params.numSrcHidVecs;
        end
      else % local
        models{mm}.params.numAttnPositions = 2*models{mm}.params.posWin + 1;
      end
      
      modelData{mm}.curMask = curMask;

      modelData{mm}.srcHidVecsOrig = zeroMatrix([models{mm}.params.lstmSize, batchSize, models{mm}.params.numSrcHidVecs], params.isGPU, params.dataType);  
      if models{mm}.params.attnGlobal
        modelData{mm}.alignMask = oneMatrix([models{mm}.params.numSrcHidVecs, models{mm}.params.curBatchSize], params.isGPU, params.dataType);
        modelData{mm}.srcMaskedIds = [];
      end
    else
      models{mm}.params.numSrcHidVecs = 0;
    end
    
    W{mm} = models{mm}.W_src;
    W_emb{mm} = models{mm}.W_emb_src;
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
        W_emb{mm} = models{mm}.W_emb_tgt;
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
            x_t = getLstmDecoderInput(input(:, tt), W_emb{mm}, softmax_h{mm}, models{mm}.params);
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
        [lstm{mm}{ll}, h_t, c_t] = lstmLayerForward(W{mm}{ll}, x_t, h_t_1, c_t_1, ll, tt, srcMaxLen, models{mm}.params, 1);
        lstm{mm}{ll}.h_t = h_t;
        lstm{mm}{ll}.c_t = c_t;

        % attention
        if tt<=models{mm}.params.numSrcHidVecs && ll==models{mm}.params.numLayers
          modelData{mm}.srcHidVecsOrig(:, :, tt) = h_t;

          % done generating all srcHidVecs, collect
          if tt==models{mm}.params.numSrcHidVecs && models{mm}.params.attnGlobal == 0 % local
            modelData{mm}.srcHidVecs = modelData{mm}.srcHidVecsOrig;
          end
        end
        
        % h_t -> softmax_h
        if tt==srcMaxLen && ll==models{mm}.params.numLayers
          modelData{mm}.curMask = curMask;
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

                  % out of boundary
                  if startAttnIds(sentId) <= offset
                    startIds(sentId) = startIds(sentId) + offset + 1 - startAttnIds(sentId);
                    startAttnIds(sentId) = offset + 1;
                  end

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
 
  if batchSize==1
    minLen = floor(srcMaxLen*params.minLenRatio);
  else
    minLen = 2;
  end
  maxLen = floor(srcMaxLen*params.maxLenRatio);
  
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
  else
    alignInfo = [];
  end
  
  %% first prediction
  numModels = length(models);
  [scores, words] = nextBeamStep(models, softmax_h, beamSize); %lstmStart{numLayers}.h_t, beamSize, params, data, curMask, tgtPos); % scores, words: beamSize * batchSize

  % TODO: by right, we should filter out words == params.tgtEos, but I
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
  if params.align
    alignHistory = zeroMatrix([maxLen, numElements], params.isGPU, params.dataType); % maxLen * (numElements) 
    % alignHistory(1, :) = firstAlignIdx;
    for ii=1:batchSize
      alignHistory(1,(ii-1)*beamSize+1:ii*beamSize) = firstAlignIdx(ii);
    end
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
    if models{mm}.params.attnFunc
      curMask.mask = ones(1, numElements);
      curMask.unmaskedIds = 1:numElements;
      models{mm}.params.curBatchSize = numElements;
    
      modelData{mm}.curBatchSize = numElements;
      modelData{mm}.srcLens = reshape(repmat(modelData{mm}.srcLens, beamSize, 1), 1, []);
      
      % duplicate srcHidVecs
      if models{mm}.params.attnGlobal % soft, global
        modelData{mm}.srcHidVecsOrig = duplicateSrcHidVecs(modelData{mm}.srcHidVecsOrig, batchSize, beamSize);

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
    W_emb{mm} = models{mm}.W_emb_tgt;
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
          x_t = getLstmDecoderInput(words, W_emb{mm}, softmax_h{mm}, models{mm}.params);
        else
          x_t = beamStates{mm}{ll-1}.h_t;
        end
        % previous input
        h_t_1 = beamStates{mm}{ll}.h_t;
        c_t_1 = beamStates{mm}{ll}.c_t;

        [beamStates{mm}{ll}, h_t, c_t] = lstmLayerForward(W{mm}, x_t, h_t_1, c_t_1, ll, srcMaxLen+sentPos, srcMaxLen, models{mm}.params, 1);
        beamStates{mm}{ll}.h_t = h_t;
        beamStates{mm}{ll}.c_t = c_t;

        % h_t -> softmax_h
        if ll==models{mm}.params.numLayers
          modelData{mm}.curMask = curMask;

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
                  
                  % out of boundary
                  if startAttnIds(sentId) <= offset
                    startIds(sentId) = startIds(sentId) + offset + 1 - startAttnIds(sentId);
                    startAttnIds(sentId) = offset+1;
                  end

                  indices = startAttnIds(sentId)-offset:endAttnIds(sentId)-offset;
                  alignWeights{sentId}(indices) = alignWeights{sentId}(indices) + h2sInfo.alignWeights(startIds(sentId):endIds(sentId), sentId);
                end
              end
            end
          
            if mm==numModels
              alignIdx = zeroMatrix([1, numElements], params.isGPU, params.dataType);
              for sentId=1:numElements % go through each sent
                [~, alignIdx(sentId)] = max(alignWeights{sentId}, [], 1); % srcLen includes eos, alignWeights excludes eos.
                % alignWeights{sentId}
                % alignIdx(sentId)
              end
              % we want to mimic the structure of allBestWords
              % size (beamSize * beamSize) x batchSize
              % alignIdx = reshape(repmat(alignIdx, beamSize, 1), [], 1);
              alignIdx = reshape(repmat(alignIdx, beamSize, 1), [], batchSize);
            end
          end
        end
      end
    end
    
    %% predict the next word
    % allBestScores, allBestWords should have size beamSize * (beamSize*batchSize)
    [allBestScores, allBestWords] = nextBeamStep(models, softmax_h, beamSize);

    % allBestWords, allBestScores should have size: (beamSize*beamSize) * batchSize
    [allBestScores, allBestWords, indices] = addSortScores(allBestScores, allBestWords, beamScores, beamSize, batchSize);

    %% build new beam
    for sentId=1:batchSize
      rowIndices = indices(:, sentId)';
      bestWords = allBestWords(rowIndices, sentId);
      
      % get candidates
      selectedIndices = find(bestWords~=params.tgtEos, beamSize);

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
      endIndices = find(bestWords(1:selectedIndices(end))==params.tgtEos); % get words that are eos and ranked before the last hypothesis in the next beam
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
    
    %% update history
    % overwrite previous history
    colIndices = (sentIndices-1)*beamSize + beamIndices;
    tmp = beamHistory(1:sentPos, colIndices);
    beamHistory(1:sentPos, :) = tmp; 
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

function [allBestScores, allBestWords, indices] = addSortScores(allBestScores, allBestWords, beamScores, beamSize, batchSize)
  % use previous beamScores, 1 * (beamSize*batchSize), update along the first dimentions
  allBestScores = bsxfun(@plus, allBestScores, beamScores);
  allBestScores = reshape(allBestScores, [beamSize*beamSize, batchSize]);
  allBestWords = reshape(allBestWords, [beamSize*beamSize, batchSize]);

  % for each sent, select the best beamSize candidates, out of beamSize*beamSize ones
  [allBestScores, indices] = sort(allBestScores, 'descend'); % (beamSize*beamSize) * batchSize
end

%%
% return bestLogProbs, bestWords of sizes beamSize * curBatchSize
%%
function [bestLogProbs, bestWords, logProbs, sortedLogProbs, sortedWords] = nextBeamStep(models, softmax_h, beamSize) %h_t, beamSize, params, data, curMask, tgtPos)
  softmax_input = models{1}.W_soft*softmax_h{1};
  if length(models)>1 % aggregate predictions from multiple models
    for ii=2:length(models)
      softmax_input = softmax_input + models{ii}.W_soft*softmax_h{ii};
      x = models{ii}.W_soft*softmax_h{ii};
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

function [startIds, endIds, startAttnIds, endAttnIds] = computeAttnBound(srcPositions, params)
  batchSize = length(srcPositions);
  
  % where to pay attention to on the source side (params.numSrcHidVecs)
  startAttnIds = srcPositions-params.posWin;
  endAttnIds = srcPositions + params.posWin;
  
  % where to get the align weights (numAttnPositions = 2*params.posWin+1)
  startIds = oneMatrix([1, batchSize], params.isGPU, params.dataType);
  endIds = params.numAttnPositions*startIds;
  
  %% boundary condition for startAttnIds
  indices = find(startAttnIds<1);
  startIds(indices) = startIds(indices) - (startAttnIds(indices)-1);
  startAttnIds(indices) = 1; % Note: don't swap these two lines
  % here, we are sure that startId>=1, startAttnId>=1
  
  %% boundary condition for endAttnIds
  indices = find(endAttnIds>params.numSrcHidVecs);
  endIds(indices) = endIds(indices) - (endAttnIds(indices)-params.numSrcHidVecs);
  endAttnIds(indices) = params.numSrcHidVecs; % Note: don't swap these two lines
  % here, we are sure that endId<=numAttnPositions, endAttnId<=params.numSrcHidVecs
  
  %% last boundary condition checks
  flags = startIds<=endIds & startAttnIds<=endAttnIds; % & flags;
  % out of boundary
  indices = find(~flags);
  startIds(indices) = 1; endIds(indices) = 0; startAttnIds(indices) = 1; endAttnIds(indices) = 0;
end
