function [params] = evalValidTest(model, validData, testData, params)
  startTime = clock;
  [validCosts, numChars] = evalCost(model, validData, params);
  validData.numChars = numChars;
  [testCosts, numChars] = evalCost(model, testData, params);
  testData.numChars = numChars;
  
  validCounts = initCosts();
  validCounts = updateCounts(validCounts, validData);
  validCosts = scaleCosts(validCosts, validCounts);
  
  testCounts = initCosts();
  testCounts = updateCounts(testCounts, testData);
  testCosts = scaleCosts(testCosts, testCounts);
  
  modelStr = wInfo(model);
  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  
  params.curTestPerpTotal = exp(testCosts.total);
  params.curTestPerpWord = exp(testCosts.word);
  [params.scaleTrainCosts] = scaleCosts(params.trainCosts, params.trainCounts);
  if params.charTgtGen
    params.curTestPerpChar = exp(testCosts.char);
    logStr = sprintf('# eval %.2f (%.2f, %.2f), %d, %d, %.2fK, %.2f (%.2f, %.2f), train=%.2f (%.2f, %.2f), valid=%.2f (%.2f, %.2f), test=%.2f (%.2f, %.2f),%s, time=%.2fs', ...
      params.curTestPerpTotal, params.curTestPerpWord, params.curTestPerpChar, ...
      params.epoch, params.iter, params.speed, params.lr, ...
      params.scaleTrainCosts.total, params.scaleTrainCosts.word, params.scaleTrainCosts.char, ...
      validCosts.total, validCosts.word, validCosts.char, testCosts.total, testCosts.word, testCosts.char, modelStr, timeElapsed);
  else
    logStr = sprintf('# eval %.2f, %d, %d, %.2fK, %.2f, train=%.2f, valid=%.2f, test=%.2f,%s, time=%.2fs', params.curTestPerpWord, ...
      params.epoch, params.iter, params.speed, params.lr, ...
      params.scaleTrainCosts.total, validCosts.total, testCosts.total, modelStr, timeElapsed);
  end
  fprintf(2, '%s\n', logStr);
  fprintf(params.logId, '%s\n', logStr);
      
  if validCosts.total < params.bestCostValid
    params.bestCostValid = validCosts.total;
    params.costTest = testCosts.total;
    params.testPerplexity = params.curTestPerpWord;

    fprintf(2, '  save model test perplexity %.2f to %s\n', params.testPerplexity, params.modelFile);
    fprintf(params.logId, '  save model test perplexity %.2f to %s\n', params.testPerplexity, params.modelFile);
    save(params.modelFile, 'model', 'params');
    
    if params.saveHDF
      saveHDF5([params.modelFile '.h5'], model, params);
    end
  end
end

%% Eval
function [evalCosts, totalNumChars] = evalCost(model, data, params) %input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen, params)
  numSents = size(data.tgtInput, 1);
  numBatches = floor((numSents-1)/params.batchSize) + 1;

  [evalCosts] = initCosts();
  trainData.srcMaxLen = data.srcMaxLen;
  trainData.tgtMaxLen = data.tgtMaxLen;
  totalNumChars = 0;
  for batchId = 1 : numBatches
    startId = (batchId-1)*params.batchSize+1;
    endId = batchId*params.batchSize;
    if endId > numSents
      endId = numSents;
    end
    
    if params.isBi
      trainData.srcInput = data.srcInput(startId:endId, :);
      trainData.srcMask = data.srcMask(startId:endId, :);
    end
    trainData.tgtInput = data.tgtInput(startId:endId, :);
    trainData.tgtMask = data.tgtMask(startId:endId, :);
    trainData.tgtOutput = data.tgtOutput(startId:endId, :);
    trainData.srcLens = data.srcLens(startId:endId); 
    trainData.tgtLens = data.tgtLens(startId:endId); 
    
    % eval
    [costs, ~, numChars] = lstmCostGrad(model, trainData, params, 1);
    totalNumChars = totalNumChars + numChars;
    [evalCosts] = updateCosts(evalCosts, costs);
  end
end
