function [params] = evalValidTest(model, validData, testData, params)
  startTime = clock;
  [validCosts] = evalCost(model, validData, params);
  [testCosts] = evalCost(model, testData, params);
  
  validCounts = initCosts(params);
  validCounts = updateCounts(validCounts, validData, params);
  validCosts = scaleCosts(validCosts, validCounts, params);
  
  testCounts = initCosts(params);
  testCounts = updateCounts(testCounts, testData, params);
  testCosts = scaleCosts(testCosts, testCounts, params);
  
  modelStr = wInfo(model);
  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  
  params.curTestPerpWord = exp(testCosts.word);
  
  if params.posSignal % positions
    params.curTestCostPos = testCosts.pos;
    
    if params.predictPos==1 % regression
      logStr = sprintf('%.4f, ', testCosts.pos);
    elseif params.predictPos==2 % classification
      logStr = sprintf('%.2f, ', exp(testCosts.pos));
    end
    
    if params.predictNull
      params.curTestPerpNull = exp(testCosts.null);
      logStr = sprintf('%s%.2f, ', logStr, params.curTestPerpNull);
    end
  else
    logStr = '';
  end
  logStr = sprintf('# eval %s%.2f, %d, %d, %.2fK, %.2f, train=%s, valid=%s, test=%s,%s, time=%.2fs', logStr, params.curTestPerpWord, ...
    params.epoch, params.iter, params.speed, params.lr, ...
    getCostStr(params.scaleTrainCosts), getCostStr(validCosts), getCostStr(testCosts), modelStr, timeElapsed);
  fprintf(2, '%s\n', logStr);
  fprintf(params.logId, '%s\n', logStr);
      
  if validCosts.total < params.bestCostValid
    params.bestCostValid = validCosts.total;
    params.costTest = testCosts.total;
    params.testPerplexity = params.curTestPerpWord;
    if params.posSignal
      params.bestCostValidPos = validCosts.pos;
      params.bestCostValidWord = validCosts.word;
      params.testCostPos = params.curTestCostPos;
      params.testPerplexity = params.curTestPerpWord;
    end
    fprintf(2, '  save model test perplexity %.2f to %s\n', params.testPerplexity, params.modelFile);
    fprintf(params.logId, '  save model test perplexity %.2f to %s\n', params.testPerplexity, params.modelFile);
    save(params.modelFile, 'model', 'params');
  end
end

%% Eval
function [evalCosts] = evalCost(model, data, params) %input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen, params)
  numSents = size(data.input, 1);
  numBatches = floor((numSents-1)/params.batchSize) + 1;

  [evalCosts] = initCosts(params);
  trainData.srcMaxLen = data.srcMaxLen;
  trainData.tgtMaxLen = data.tgtMaxLen;
  for batchId = 1 : numBatches
    startId = (batchId-1)*params.batchSize+1;
    endId = batchId*params.batchSize;
    if endId > numSents
      endId = numSents;
    end
    
    trainData.input = data.input(startId:endId, :);
    trainData.inputMask = data.inputMask(startId:endId, :);
    if params.isBi
      trainData.srcMask = data.srcMask(startId:endId, :);
    end
    trainData.tgtMask = data.tgtMask(startId:endId, :);
    trainData.tgtOutput = data.tgtOutput(startId:endId, :);
    trainData.srcLens = data.srcLens(startId:endId); 
    if params.posSignal
      trainData.posOutput = data.posOutput(startId:endId, :);
    end
    
    % eval
    costs = lstmCostGrad(model, trainData, params, 1);
    [evalCosts] = updateCosts(evalCosts, costs, params);
  end
end
