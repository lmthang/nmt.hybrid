function [candidates, candScores] = lstmDecoder(model, data, params)
%%%
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
%%%
  beamSize = params.beamSize;
  stackSize = params.stackSize;
  
  input = data.input;
  inputMask = data.inputMask; 
  srcMaxLen = data.srcMaxLen;
  curBatchSize = size(input, 1);
  
  %printSent(2, input(1, 1:srcMaxLen), params.vocab, 'src 1: ');
  %printSent(2, input(1, srcMaxLen:end), params.vocab, 'tgt 1: ');
  %printSent(2, input(end, 1:srcMaxLen), params.vocab, 'src end: ');
  
  %% init
  batchSize = size(input, 1);
  zeroState = zeroMatrix([params.lstmSize, batchSize], params.isGPU, params.dataType);
  fprintf(2, '# Decoding batch of %d sents, srcMaxLen=%d, %s\n', batchSize, srcMaxLen, datestr(now));
  fprintf(params.logId, '# Decoding batch of %d sents, srcMaxLen=%d, %s\n', batchSize, srcMaxLen, datestr(now));
  
  %%%%%%%%%%%%
  %% encode %%
  %%%%%%%%%%%%
  lstm = cell(params.numLayers, 1); % lstm can be over written, as we do not need to backprop
  
  % attentional / positional models
  if params.attnFunc>0 || params.posModel>=2 
    params.numSrcHidVecs = srcMaxLen-1;
    data.curMask.mask = ones(1, curBatchSize);
    data.curMask.unmaskedIds = 1:curBatchSize;
    data.curMask.maskedIds = [];

    
    if params.attnFunc==2
      data.srcHidVecsAll = zeroMatrix([params.lstmSize, curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);  
    end
    
    if params.attnFunc==1
      data.srcHidVecs = zeroMatrix([params.lstmSize, curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);  
    end
  else
    params.numSrcHidVecs = 0;
  end
  
  W = model.W_src;
  
  % separate emb
  if params.separateEmb==1 
    W_emb = model.W_emb_src;
  else
    W_emb = model.W_emb;
  end
  
  for tt=1:srcMaxLen % time
    maskedIds = find(~inputMask(:, tt)); % curBatchSize * 1
    if tt==srcMaxLen % due to implementation in lstmCostGrad, we have to switch to W_tgt here. THIS IS VERY IMPORTANT!
      W = model.W_tgt;  
      
      if params.separateEmb==1 
        W_emb = model.W_emb_tgt;
      end
    end
    
    for ll=1:params.numLayers % layer
      % previous-time input
      if tt==1 % first time step
        h_t_1 = zeroState;
        c_t_1 = zeroState;
      else
        h_t_1 = lstm{ll}.h_t; 
        c_t_1 = lstm{ll}.c_t;
      end

      % current-time input
      if ll==1 % first layer
        x_t = W_emb(:, input(:, tt));
        %params.vocab(input(:, t))

      else % subsequent layer, use the previous-layer hidden state
        x_t = lstm{ll-1}.h_t;
      end
     
      % masking
      x_t(:, maskedIds) = 0; 
      h_t_1(:, maskedIds) = 0;
      c_t_1(:, maskedIds) = 0;

      % lstm cell
      [lstm{ll}, h_t, c_t] = lstmUnit(W{ll}, x_t, h_t_1, c_t_1, ll, tt, srcMaxLen, params, 1);
      lstm{ll}.h_t = h_t;
      lstm{ll}.c_t = c_t;
      
      % attentional  models
      if tt<=params.numSrcHidVecs && ll==params.numLayers
        if params.attnFunc==1
          data.srcHidVecs(:, :, params.numAttnPositions-params.numSrcHidVecs+tt) = h_t;
        elseif params.attnFunc==2
          data.srcHidVecsAll(:, :, tt) = h_t;
        end
      end
      
      % assert
      if params.assert
        assert(sum(sum(abs(lstm{ll}.c_t(:, maskedIds))))<1e-5);
        assert(sum(sum(abs(lstm{ll}.h_t(:, maskedIds))))<1e-5);
      end
    end
  end
  
  %%%%%%%%%%%%
  %% decode %%
  %%%%%%%%%%%%
  startTime = clock;
  if params.depParse % dependency parsing
    minLen = (srcMaxLen-1)*2; % srcMaxLen include eos, we want our dependency parse will have length = 2 * input (no eos) length 
    maxLen = minLen;
  else
    if batchSize==1
      minLen = floor(srcMaxLen*params.minLenRatio);
    else
      minLen = 2;
    end
    maxLen = floor(srcMaxLen*params.maxLenRatio);
  end
  
  sentIndices = data.startId:(data.startId+batchSize-1);
  [candidates, candScores] = decodeBatch(model, params, lstm, minLen, maxLen, beamSize, stackSize, batchSize, sentIndices, srcMaxLen, data);
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
function [candidates, candScores] = decodeBatch(model, params, lstmStart, minLen, maxLen, beamSize, stackSize, batchSize, originalSentIndices, srcMaxLen, data)
  numLayers = params.numLayers;
  numElements = batchSize*beamSize;
  
  candidates = cell(batchSize, 1);
  candScores = -1e10*oneMatrix([stackSize, batchSize], params.isGPU, params.dataType); % set to a very small value
  numDecoded = zeros(batchSize, 1);
  for ii=1:batchSize
    candidates{ii} = cell(stackSize, 1);
  end
  
  %% first prediction
  if params.attnFunc==2
    tgtPos = 1;
    data.srcHidVecs = zeroMatrix([params.lstmSize, batchSize, params.numAttnPositions], params.isGPU, params.dataType);
    [startAttnId, endAttnId, startHidId, endHidId] = buildSrcHidVecs(srcMaxLen, tgtPos, params);
    data.srcHidVecs(:, :, startHidId:endHidId) = data.srcHidVecsAll(:, :, startAttnId:endAttnId);
  end
  [scores, words] = nextBeamStep(model, lstmStart{numLayers}.h_t, beamSize, params, data); % scores, words: beamSize * batchSize
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
  beamStates = cell(numLayers, 1);
  for ll=1:numLayers % lstmSize * numElements
    beamStates{ll}.c_t = reshape(repmat(lstmStart{ll}.c_t, beamSize, 1),  params.lstmSize, numElements); 
    beamStates{ll}.h_t = reshape(repmat(lstmStart{ll}.h_t, beamSize, 1),  params.lstmSize, numElements); 
  end
  
  % dependency parse
  if params.depParse
    assert(batchSize==1);
    stackCounts = ones(1, numElements); % at first, all hypotheses have R(root) in the stack
    bufferCounts = data.srcLens - 1 - stackCounts; % don't count eos, here we assume batchSize = 1
  end
  
  % attentional / positional models
  if params.attnFunc>0 || params.posModel>=2 
    curBatchSize = numElements;
    data.curMask.mask = ones(1, curBatchSize);
    data.curMask.unmaskedIds = 1:curBatchSize;
    data.curMask.maskedIds = [];
    
    if params.attnFunc==1
      [data.srcHidVecs] = duplicateSrcHidVecs(data.srcHidVecs, batchSize, params, beamSize);
    end
  end
  
  decodeCompleteCount = 0; % count how many sentences we have completed collecting the translations
  beamWords = zeroMatrix([1, numElements], params.isGPU, params.dataType);
  beamIndices = zeroMatrix([1, numElements], params.isGPU, params.dataType);

  % separate emb
  if params.separateEmb==1 
    W_emb = model.W_emb_tgt;
  else
    W_emb = model.W_emb;
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
    for ll = 1 : numLayers
      % current input
      if ll == 1
        x_t = W_emb(:, words);

      else
        x_t = beamStates{ll-1}.h_t;
      end
      % previous input
      h_t_1 = beamStates{ll}.h_t;
      c_t_1 = beamStates{ll}.c_t;

      [beamStates{ll}, h_t, c_t] = lstmUnit(model.W_tgt{ll}, x_t, h_t_1, c_t_1, ll, srcMaxLen+sentPos, srcMaxLen, params, 1);
      beamStates{ll}.h_t = h_t;
      beamStates{ll}.c_t = c_t;
    end
    
    %% predict the next word
    if params.attnFunc==2
      [data.srcHidVecs] = computeRelativeSrcHidVecs(data.srcHidVecsAll, srcMaxLen, tgtPos, batchSize, params, beamSize);
    end
    if params.depParse % dependency parsing
      if sentPos == (maxLen-1) % we want R(root) to be the next word
        [~, ~, logProbs] = nextBeamStep(model, beamStates{numLayers}.h_t, beamSize, params, data);
        allBestScores = logProbs(params.depRootId, :);
        allBestWords = params.depRootId*ones(1, beamSize*batchSize);
        card = 1;
      else
        [allBestScores, allBestWords, logProbs] = nextBeamStep(model, beamStates{numLayers}.h_t, beamSize, params, data); % beamSize * (beamSize*batchSize)
        
        %% NOTE: this code require batchSize to be 1.
        % shift
        shiftIndices = find(stackCounts==1 & bufferCounts>0); % when stack has one word and buffer is not empty, needs shift
        mustShiftCount = length(shiftIndices);
        if mustShiftCount>0
          allBestScores(1, shiftIndices) = logProbs(params.depShiftId, shiftIndices); 
          allBestWords(1, shiftIndices) = params.depShiftId;
          allBestScores(2:end, shiftIndices) = -1e10; % make the scores very small, so it won't make into the beam
        end
        card = beamSize;
      end
    else
      [allBestScores, allBestWords] = nextBeamStep(model, beamStates{numLayers}.h_t, beamSize, params, data); % beamSize * (beamSize*batchSize)
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
      if params.depParse % dependency parsing
        if sentPos<(maxLen-1) % exclude eos and R(root)
          % because of shiftIndices, we know that only the following top choices are valid: 
          %   beamSize*(beamSize-mustShiftCount) for non-shift operators + 
          %   mustShiftCount operators.
          validCount = beamSize*(beamSize-mustShiftCount) + mustShiftCount;
          selectedIndices = find(bestWords(1:validCount)~=params.tgtEos & bestWords(1:validCount)~=params.depRootId, beamSize);
        else % we already chose R(root) above
          assert(length(bestWords)==beamSize);
          assert(isempty(find(bestWords~=params.depRootId, 1)));
          selectedIndices = 1:beamSize;
          %selectedIndices = find(bestWords==params.depRootId, beamSize);
        end
      else
        selectedIndices = find(bestWords~=params.tgtEos, beamSize);
      end

      % update words
      startId = (sentId-1)*beamSize+1;
      endId = sentId*beamSize;
      sentWords = bestWords(selectedIndices);
      beamWords(startId:endId) = sentWords;

      % update beam
      sentBeamIndices = floor((rowIndices(selectedIndices)-1)/beamSize) + 1;
      beamIndices(startId:endId) = sentBeamIndices;

      % update scores
      beamScores(startId:endId) = allBestScores(selectedIndices, sentId);
      
      if params.depParse % dependency parsing
        % shift: increase stack count, decrease buffer count
        shiftIndices = find(sentWords == params.depShiftId);
        oldStackCounts = stackCounts;
        stackCounts(shiftIndices) = stackCounts(sentBeamIndices(shiftIndices)) + 1;
        bufferCounts(shiftIndices) = bufferCounts(sentBeamIndices(shiftIndices)) - 1;

        % not shift: decrease stack count
        nonshiftIndices = find(sentWords ~= params.depShiftId);
        stackCounts(nonshiftIndices) = oldStackCounts(sentBeamIndices(nonshiftIndices)) -1; % Important: to use oldStackCounts here
      end
      
      % store translations
      if params.depParse==0 || sentPos==(maxLen-1) % for dependency parsing, we only collect output at maxLen, i.e. end at R(root), then append eos.
        if params.depParse % dependency parsing, we already chose R(root)
          endIndices = selectedIndices; %find(bestWords==params.depRootId);
        else
          endIndices = find(bestWords(1:selectedIndices(end))==params.tgtEos); % get words that are eos and ranked before the last hypothesis in the next beam
        end
        
        if ~isempty(endIndices) && (sentPos+1)>=minLen % we don't want to start recording very short translations
          numTranslations = length(endIndices);
          eosBeamIndices = floor((rowIndices(endIndices)-1)/beamSize) + 1;
          translations = beamHistory(1:sentPos, (sentId-1)*beamSize + eosBeamIndices);
          transScores = allBestScores(endIndices, sentId);
          for ii=1:numTranslations
            if numDecoded(sentId)<stackSize % haven't collected enough translations
              numDecoded(sentId) = numDecoded(sentId) + 1;

              if params.depParse % dependency parsing, append R(root) and eos
                candidates{sentId}{numDecoded(sentId)} = [translations(:, ii); params.depRootId; params.tgtEos];
              else
                candidates{sentId}{numDecoded(sentId)} = [translations(:, ii); params.tgtEos];
              end
              candScores(numDecoded(sentId), sentId) = transScores(ii);

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

    %% update lstm states
    for ll=1:numLayers
      % lstmSize * (numElements): h_t and c_t vectors of each sent are arranged near each other
      beamStates{ll}.c_t = beamStates{ll}.c_t(:, colIndices); 
      beamStates{ll}.h_t = beamStates{ll}.h_t(:, colIndices);
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
function [bestLogProbs, bestWords, logProbs, sortedLogProbs, sortedWords] = nextBeamStep(model, h_t, beamSize, params, data)
  % softmax
  if params.attnFunc>0 % attention mechanism
    [softmax_h] = lstm2softHid(h_t, params, model, data.srcHidVecs, data.curMask);
  elseif params.posModel>0
    error('! Have not implemented decoder for positional models\n');
  else
    [softmax_h] = lstm2softHid(h_t, params, model);
  end
  
  [logProbs] = softmaxDecode(model.W_soft*softmax_h);
  
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

function [srcHidVecs] = computeRelativeSrcHidVecs(srcHidVecsAll, srcMaxLen, tgtPos, batchSize, params, beamSize)
  srcHidVecs = zeroMatrix([params.lstmSize, batchSize, params.numAttnPositions], params.isGPU, params.dataType);
  [startAttnId, endAttnId, startHidId, endHidId] = buildSrcHidVecs(srcMaxLen, tgtPos, params);
  srcHidVecs(:, :, startHidId:endHidId) = srcHidVecsAll(:, :, startAttnId:endAttnId);

  % duplicate srcHidVecs along the curBatchSize dimension beamSize times
  if (beamSize>1)
    [srcHidVecs] = duplicateSrcHidVecs(srcHidVecs, batchSize, params, beamSize);
  end
end

function [srcHidVecs] = duplicateSrcHidVecs(srcHidVecs, batchSize, params, beamSize)
  numElements = batchSize*beamSize;
  srcHidVecs = permute(srcHidVecs, [1, 3, 2]); % lstmSize * numAttnPositions * batchSize
  srcHidVecs = reshape(srcHidVecs, params.lstmSize*params.numAttnPositions, batchSize);
  srcHidVecs = repmat(srcHidVecs, beamSize, 1);
  srcHidVecs = reshape(srcHidVecs, params.lstmSize, params.numAttnPositions, numElements);
  srcHidVecs = permute(srcHidVecs, [1, 3, 2]); % lstmSize * batchSize * numAttnPositions
end

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

%% Unused %%
