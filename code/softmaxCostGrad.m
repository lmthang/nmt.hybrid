%%%
%
% Perform softmax prediction and backprop.
%
% From the point of view of the model graph structure, this method handles
% all subgraphs from the LSTM hiddent state to the softmax predictions,
% i.e., there can be other intermediate layers for the case of positional
% models, attention functions, and softmax compression.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
function [costs, softmaxGrad, otherGrads] = softmaxCostGrad(model, params, trainData)
  topHidVecs = trainData.topHidVecs;
  curBatchSize = params.curBatchSize;
  inputMask = trainData.inputMask;
  T = trainData.T;
  srcMaxLen = trainData.srcMaxLen;
  srcLens = trainData.srcLens;
  
  % init grads
  [softmaxGrad, otherGrads] = initSoftmaxGrad(model, params);
  if params.attnFunc>0 || params.posModel==3
    softmaxGrad.srcHidVecs = zeroMatrix([params.lstmSize, curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
  end
  otherGrads.ht = zeroMatrix(size(topHidVecs), params.isGPU, params.dataType);
  
  % init costs
  costs.total = zeroMatrix([1, 1], params.isGPU, params.dataType);
  costs.word = zeroMatrix([1, 1], params.isGPU, params.dataType);
  if params.posModel>0
    costs.pos = zeroMatrix([1, 1], params.isGPU, params.dataType);
  end
  
  if params.posModel==3
    wordCount = 0;
    otherGrads.allEmbGrads = zeroMatrix([params.lstmSize, trainData.numInputWords], params.isGPU, params.dataType);
    otherGrads.allEmbIndices = zeros(trainData.numInputWords, 1);
  end
  
  %% predict words from srcMaxLen to T
  isPredictPos = 0;
  if params.attnFunc==1
    batchData.srcHidVecs = trainData.srcHidVecs;
  else
    batchData = [];
  end
  for startT=srcMaxLen:params.softmaxStep:T
    endT = startT + params.softmaxStep -1;
    if endT>T
      endT = T;
    end
    numTimeSteps = (endT-startT+1);
    numExamples = numTimeSteps*curBatchSize;
  
    range = startT:endT;
    tgtRange = (startT-srcMaxLen+1):(endT-srcMaxLen+1);
        
    % prepare mask
    curMask.mask = reshape(inputMask(:, range), 1, numExamples);
    curMask.unmaskedIds = find(curMask.mask);
    curMask.maskedIds = find(~curMask.mask);
    curMask.count = length(curMask.unmaskedIds);
    
    % h_t
    h_t = reshape(topHidVecs(:, :, range), params.lstmSize, numExamples);
    
    %% attention model 2: relative positions, we assume softmaxStep=1
    if params.attnFunc==2
      tgtPos = (startT-srcMaxLen+1);
      
      % srcMaxLen-srcLens(i) -> srcMaxLen-1
      srcPositions = params.numSrcHidVecs - srcLens(curMask.unmaskedIds) + tgtPos; %srcMaxLen - srcLens + tgtPos - 1;
      startAttnIds = srcPositions-params.posWin;
      endAttnIds = srcPositions + params.posWin;
      startHidIds = ones(1, curMask.count);
      endHidIds = params.numAttnPositions * ones(1, curMask.count);
      
      % < 1
      flagIndices = find(startAttnIds<1);
      startHidIds(flagIndices) = startHidIds(flagIndices) - (startAttnIds(flagIndices)-1);
      startAttnIds(flagIndices) = 1; % Note: don't swap these two lines
       
      % > numSrcHidVecs
      flagIndices = find(endAttnIds>params.numSrcHidVecs);
      endHidIds(flagIndices) = endHidIds(flagIndices) - (endAttnIds(flagIndices)-params.numSrcHidVecs);
      endAttnIds(flagIndices) = params.numSrcHidVecs; % Note: don't swap these two lines
        
      % populate srcHidVecs: have to use for loop.
      batchData.srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
      yIds = zeros(1, params.curBatchSize*params.numAttnPositions);
      batch_zIds = zeros(1, params.curBatchSize*params.numAttnPositions);
      train_zIds = zeros(1, params.curBatchSize*params.numAttnPositions);
      numHidVecs = 0;
      if params.assert
        assert(isempty(startHidIds>endHidIds));
      end
      for ii=1:curMask.count
        count = endHidIds(ii) - startHidIds(ii) + 1;
        if count<=0
          continue;
        end
        yIds(numHidVecs+1:numHidVecs+count) = curMask.unmaskedIds(ii)*ones(1, count);
        batch_zIds(numHidVecs+1:numHidVecs+count) = startHidIds(ii):endHidIds(ii);
        train_zIds(numHidVecs+1:numHidVecs+count) = startAttnIds(ii):endAttnIds(ii);
        numHidVecs = numHidVecs + count;
        % batchData.srcHidVecs(:, ii, startHidIds(ii):endHidIds(ii)) = trainData.srcHidVecs(:, ii, startAttnIds(ii):endAttnIds(ii));
      end
      yIds(numHidVecs+1:end) = [];
      batch_zIds(numHidVecs+1:end) = [];
      train_zIds(numHidVecs+1:end) = [];
      batchData.batchLinearIndices = getTensorLinearIndices(batchData.srcHidVecs, yIds, batch_zIds);
      batchData.trainLinearIndices = getTensorLinearIndices(trainData.srcHidVecs, yIds, train_zIds);
      batchData.srcHidVecs(batchData.batchLinearIndices) = trainData.srcHidVecs(batchData.trainLinearIndices);
    end
    
    %% positional model 3: use hidden state info
    if params.posModel==3
      % srcPosVecs: src hidden states at specific positions
      srcPosData = repmat(struct('eosIds', [], 'nullIds', [], 'posIds', [], 'colIndices', [], 'embIndices', []), numTimeSteps, 1); 
      batchData = struct('srcPosVecs', zeroMatrix([params.lstmSize, curBatchSize, numTimeSteps], params.isGPU, params.dataType), ...
        'posIds', [], 'nullIds', [], 'eosIds', [], 'colIndices', []);
      for t=1:numTimeSteps
        [batchData.srcPosVecs(:, :, t), srcPosData(t)] = buildSrcPosVecs(t+startT-1, model, params, trainData, trainData.maskInfo{t+startT-1});
      end
      batchData.posIds = [srcPosData(:).posIds];
      batchData.nullIds = [srcPosData(:).nullIds];
      batchData.eosIds = [srcPosData(:).eosIds];
      batchData.colIndices = [srcPosData(:).colIndices];
    end
    
    % predicted words
    predWords = reshape(trainData.tgtOutput(:, tgtRange), 1, numExamples);
    
    % predict
    if params.numClasses>0
      predClasses = mod(predWords, params.numClasses) + 1;
      predInClasses = floor((predWords-1)/params.numClasses + 1e-9) + 1;
      [word_cost, word_softmaxGrad, word_ht_grad , otherBatchGrads] = batchSoftmax('W_soft_class', h_t, predClasses, model, params, trainData, batchData, curMask, isPredictPos, predInClasses);
    else
      [word_cost, word_softmaxGrad, word_ht_grad, otherBatchGrads] = batchSoftmax('W_soft', h_t, predWords, model, params, trainData, batchData, curMask, isPredictPos);
    end
    costs.total = costs.total + word_cost;
    costs.word = costs.word + word_cost;
    
    %% grads
    if trainData.isTest==0
      fields = fieldnames(word_softmaxGrad);
      for ii=1:length(fields)
        field = fields{ii};
        softmaxGrad.(field) = softmaxGrad.(field) + word_softmaxGrad.(field);
      end
      
      % attention models: srcHidVecs
      if params.attnFunc==1
        softmaxGrad.srcHidVecs = softmaxGrad.srcHidVecs + otherBatchGrads.srcHidVecs;
      elseif params.attnFunc==2
        % mask: 1 * curBatchSize
        % srcHidVecs: lstmSize * curBatchSize * numAttnPoints
        % mask out grad before accumulating
        softmaxGrad.srcHidVecs(batchData.trainLinearIndices) = softmaxGrad.srcHidVecs(batchData.trainLinearIndices) + otherBatchGrads.srcHidVecs(batchData.batchLinearIndices);
      end
      
      % class-based softmax
      if params.numClasses>0 
        otherGrads.W_soft_inclass(:, :, otherBatchGrads.indices) = otherGrads.W_soft_inclass(:, :, otherBatchGrads.indices) + otherBatchGrads.W_soft_inclass;
        otherGrads.classIndices = unique([otherGrads.classIndices otherBatchGrads.indices]);
      end
      
      % positional model 3
      if params.posModel==3
        % update embs of <p_n> and <p_eos>
        tmpIndices = [batchData.nullIds batchData.eosIds];
        numWords = length(tmpIndices);
        otherGrads.allEmbIndices(wordCount+1:wordCount+numWords) = [params.nullPosId*ones(1, length(batchData.nullIds)) params.eosPosId*ones(1, length(batchData.eosIds))];
        otherGrads.allEmbGrads(:, wordCount+1:wordCount+numWords) = otherBatchGrads.srcPosVecs(:, tmpIndices);
        
        
        wordCount = wordCount + numWords;
        if ~isempty(batchData.posIds)
          [linearIndices] = getTensorLinearIndices(trainData.srcHidVecs, batchData.posIds, batchData.colIndices);
          softmaxGrad.srcHidVecs(linearIndices) = softmaxGrad.srcHidVecs(linearIndices) + reshape(otherBatchGrads.srcPosVecs(:, batchData.posIds), 1, []);
        end 
      end
            
      % grad_ht
      word_ht_grad = reshape(word_ht_grad, [params.lstmSize, curBatchSize, numTimeSteps]);
      if params.assert && numTimeSteps == 1
        assert(sum(sum(abs(word_ht_grad(:, curMask.maskedIds))))==0);
      end
      otherGrads.ht(:, :, range) = otherGrads.ht(:, :, range) + word_ht_grad;
    end
  end
  
  %% positional model 1, 2: predict positions from (srcMaxLen-1) to (T-1)
  %% positional model 3: predict positions from srcMaxLen to T
  if params.posModel==1 || params.posModel==2
    origStartT = srcMaxLen-1;
    origEndT = T-1;
  elseif params.posModel==3
    origStartT = srcMaxLen;
    origEndT = T;
    
    otherGrads.allEmbGrads(:, wordCount+1:end) = [];
    otherGrads.allEmbIndices(wordCount+1:end) = [];
  end    
  if params.posModel>0
    isPredictPos = 1;
    for startT=origStartT:params.softmaxStep:origEndT
      endT = startT + params.softmaxStep -1;
      if endT>origEndT
        endT = origEndT;
      end
      numTimeSteps = (endT-startT+1);
      numExamples = numTimeSteps*curBatchSize;
      range = startT:endT;

      % predicted positions
      predPositions = reshape(trainData.srcPos(:, (startT-origStartT+1):(endT-origStartT+1)), 1, numExamples) - (params.startPosId-1);
    
      % h_t
      h_t = reshape(topHidVecs(:, :, range), params.lstmSize, numExamples);

      % prepare mask
      curMask.mask = reshape(inputMask(:, range), 1, numExamples);
      curMask.unmaskedIds = find(curMask.mask);
      curMask.maskedIds = find(~curMask.mask);

      [pos_cost, pos_softmaxGrad, pos_ht_grad] = batchSoftmax('W_softPos', h_t, predPositions, model, params, trainData, batchData, curMask, isPredictPos);
      costs.total = costs.total + pos_cost;
      costs.pos = costs.pos + pos_cost;
      
      % grads
      if trainData.isTest==0
        fields = fieldnames(pos_softmaxGrad);
        for ii=1:length(fields)
          field = fields{ii};
          softmaxGrad.(field) = softmaxGrad.(field) + pos_softmaxGrad.(field);
        end
        
        % grad_ht
        otherGrads.ht(:, :, range) = otherGrads.ht(:, :, range) + reshape(pos_ht_grad, [params.lstmSize, curBatchSize, numTimeSteps]);
      end
    end
  end
end

function [cost, softmaxGrad, grad_ht, otherBatchGrads] = batchSoftmax(matrixName, h_t, origPredLabels, model, params, trainData, batchData, curMask, isPredictPos, varargin)
%%%
%
% Perform softmax prediction and backprop.
%   matrixName: 'W_soft' for predicting words (mostly used), and
%               'W_softPos' for predicting positions.
% When param.numClasses>0: perform class-based softmax. origPredLabels are class
%   labels. Predicting class labels is like predicting words in the regular
%   softmax. Besides, we need to predict the inclass word indices.
%
%%%
  mask = curMask.mask;
  unmaskedIds = curMask.unmaskedIds;
  maskedIds = curMask.maskedIds;
  
  %% h_t -> softmax_h
  if params.attnFunc>0 % attention mechanism
    [softmax_h, ~, attn_h_concat, alignWeights] = lstm2softHid(h_t, params, model, batchData.srcHidVecs, curMask);
  elseif params.posModel==3 % positional model
    [softmax_h, interSoftInput] = lstm2softHid(h_t, params, model, isPredictPos, batchData.srcPosVecs); %, curMask);
  else
    [softmax_h, interSoftInput] = lstm2softHid(h_t, params, model);
  end

  %% softmax_h -> predictions
  [probs, scores, norms] = softmax(model.(matrixName)*softmax_h, mask);
  % cost
  predLabels = origPredLabels(unmaskedIds);
  scoreIndices = sub2ind(size(scores), predLabels, unmaskedIds); % 1 * length(tgtPredictedWords)
  cost = - sum(scores(scoreIndices)) + sum(log(norms).*mask);
  
  % class-based softmax
  if params.numClasses > 0
    predInClass = varargin{1};
    predInClass = predInClass(unmaskedIds);
    
    % in class loss
    % model.W_soft_inclass(:,:,curClasses): classSize * softmaxSize * batchSize
    % softmax_h: softmaxSize * batchSize
    % inClassRaw: classSize * batchSize (sum across softmaxSize, dim 2)
    curClasses = origPredLabels;
    inClassRaw = reshape(sum(bsxfun(@times, model.W_soft_inclass(:,:,curClasses), permute(softmax_h, [3 1 2])), 2), params.classSize, params.curBatchSize);
    [inClassProbs, inClassScores, inClassNorms] = softmax(inClassRaw, mask);
    
    inClassScoreIndices = sub2ind(size(inClassScores), predInClass, unmaskedIds);
    inClassCost = - sum(inClassScores(inClassScoreIndices)) + sum(log(inClassNorms).*mask);
    
    cost = cost + inClassCost;
  end
  
  %% grad
  softmaxGrad = [];
  grad_ht = [];
  otherBatchGrads = [];
  if trainData.isTest==0 % compute grad
    %% loss -> grad_softmax_h
    probs(scoreIndices) = probs(scoreIndices) - 1; % minus one at predicted words

    % softmax_h
    grad_softmax_h = model.(matrixName)'* probs;
    
    % class-based softmax
    if params.numClasses > 0 
      inClassProbs(inClassScoreIndices) = inClassProbs(inClassScoreIndices) - 1;

      % W_soft_inclass(:,:,curClasses): classSize * softmaxSize * batchSize
      % inClassProbs: classSize * batchSize
      % grad_softmax_h: softmaxSize * batchSize (sum across classSize, dim 1)
      grad_softmax_h = grad_softmax_h + squeeze(sum(bsxfun(@times, model.W_soft_inclass(:,:,curClasses), permute(inClassProbs, [1 3 2])), 1));
    end

    if params.assert && ~isempty(maskedIds)
      assert(sum(sum(abs(grad_softmax_h(:, maskedIds))))==0);
    end
    
    %% grad_softmax_h -> h_t
    if params.softmaxDim>0 || params.attnFunc>0 || (params.posModel==3 && isPredictPos==0) % softmax compression or attention
      if params.softmaxDim>0 || params.posModel==3 % f(W_h * interSoftInput)
        % f'(softmax_h).*grad_softmax_h
        tmpResult = params.nonlinear_f_prime(softmax_h).*grad_softmax_h;

        % grad.W_h
        softmaxGrad.W_h = tmpResult*interSoftInput';

        if params.softmaxDim>0
          % grad_ht
          grad_ht = model.W_h'*tmpResult;
        else
          % grad_srcPosH: [srcPosVecs; h_t]
          grad_srcPosH = model.W_h'*tmpResult;
          
          % grad_ht
          grad_ht = grad_srcPosH(params.lstmSize+1:end, :);
          
          % grad srcPosVecs
          otherBatchGrads.srcPosVecs = grad_srcPosH(1:params.lstmSize, :);
        end
      elseif params.attnFunc>0 % f(W_ah*[attn_t; tgt_h_t])
        if params.assert && ~isempty(maskedIds)
          assert(sum(sum(abs(attn_h_concat(:, maskedIds))))==0);
        end
        [softmaxGrad, grad_ht, otherBatchGrads.srcHidVecs] = attnBackprop(model, batchData, softmax_h, grad_softmax_h, attn_h_concat, alignWeights, h_t, params, curMask);
      end
    else % normal softmax
      grad_ht = grad_softmax_h;
    end
    
    %% loss -> W_soft. 
    % Note: do not move this part up; otherwise, the case attnFunc>0 will
    % fail because softmaxGrad is only created after calling attnBackprop,
    % which means anything touches softmaxGrad before will be lost.
    softmaxGrad.(matrixName) = probs*softmax_h';
    
    if params.numClasses>0 % class-based softmax
      numWords = length(unmaskedIds);
      % inClassProbs(:, unmaskedIds): classSize * numWords
      % softmax_h(:, unmaskedIds): softmaxSize * numWords
      % W_soft_inclass: classSize * softmaxSize * numWords
      W_soft_inclass_grads = bsxfun(@times, permute(inClassProbs(:, unmaskedIds),[1 3 2]), permute(softmax_h(:, unmaskedIds),[3 1 2]));
      
      % grad.W_soft_inclass: (classSize * softmaxSize) * numWords
      W_soft_inclass_grads = reshape(W_soft_inclass_grads, [params.classSize*params.softmaxSize numWords]);
      
      % accumulate
      [otherBatchGrads.W_soft_inclass, otherBatchGrads.indices] = aggregateMatrix(W_soft_inclass_grads, curClasses(unmaskedIds), params.isGPU, params.dataType);
      otherBatchGrads.W_soft_inclass = reshape(otherBatchGrads.W_soft_inclass, [params.classSize params.softmaxSize length(otherBatchGrads.indices)]);
    end
  end % end isTest
  
  
  % assert
  if params.assert
    if params.isGPU
      assert(gather(sum(sum(abs(scores(:, maskedIds)))))==0);
    else
      assert(sum(sum(abs(scores(:, maskedIds))))==0);
    end
    
    if params.numClasses > 0
      if params.isGPU
        assert(gather(sum(sum(abs(inClassScores(:, maskedIds)))))==0);
      else
        assert(sum(sum(abs(inClassScores(:, maskedIds))))==0);
      end
    end
  end
end

function [softmaxGrad, otherGrads] = initSoftmaxGrad(model, params)
  for ii=1:length(params.softmaxVars)
    field = params.softmaxVars{ii};
    softmaxGrad.(field) = zeroMatrix(size(model.(field)), params.isGPU, params.dataType);
  end
  
  if params.numClasses == 0 % normal
    otherGrads = [];
  else % class-based
    otherGrads.W_soft_inclass = zeroMatrix(size(model.W_soft_inclass), params.isGPU, params.dataType);
    otherGrads.classIndices = [];
  end
end

%       startAttnId = tgtPos-params.posWin;
%       endAttnId = tgtPos + params.posWin;
%       startHidId = 1;
%       endHidId = params.numAttnPositions;
%       if startAttnId<1
%         startHidId = startHidId - (startAttnId-1);
%         startAttnId = 1; % Note: don't swap these two lines
%       end
%       if endAttnId>params.numSrcHidVecs
%         endHidId = endHidId - (endAttnId-params.numSrcHidVecs);
%         endAttnId = params.numSrcHidVecs; % Note: don't swap these two lines
%       end
%       
%       batchData.srcHidVecs(:, :, startHidId:endHidId) = trainData.srcHidVecs(:, :, startAttnId:endAttnId);
%       
%       % zero out the rest
%       batchData.srcHidVecs(:, :, 1:startHidId-1) = 0;
%       batchData.srcHidVecs(:, :, endHidId+1:end) = 0;
%       
%       batchData.startAttnId = startAttnId;
%       batchData.endAttnId = endAttnId;
%       batchData.startHidId = startHidId;
%       batchData.endHidId = endHidId;