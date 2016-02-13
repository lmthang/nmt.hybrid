function [params] = evalValidTest(model, validData, testData, params)
  startTime = clock;
  [validCosts, numChars] = evalCost(model, validData, params);
  validData.numChars = numChars;
  [testCosts, numChars] = evalCost(model, testData, params);
  testData.numChars = numChars;
  
  validCounts = initCosts(params);
  validCounts = updateCounts(validCounts, validData);
  validCosts = scaleCosts(validCosts, validCounts);
  
  testCounts = initCosts(params);
  testCounts = updateCounts(testCounts, testData);
  testCosts = scaleCosts(testCosts, testCounts);
  
  modelStr = wInfo(model);
  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  
  avgTrainCosts = scaleCosts(params.trainCosts, params.trainCounts);
  params.testPplWord = exp(testCosts.word);
  if params.charTgtGen
    params.testPplChar = exp(testCosts.char/params.charWeight);
    logStr = sprintf('# eval (%.2f, %.2f), %d, %d, %.2fK, %.2f (%.2f, %.2f), train=(%.2f, %.2f), valid=(%.2f, %.2f), test=(%.2f, %.2f),%s, time=%.2fs', ...
      params.testPplWord, params.testPplChar, ...
      params.epoch, params.iter, params.speed, params.lr, ...
      avgTrainCosts.word, avgTrainCosts.char, ...
      validCosts.word, validCosts.char, testCosts.word, testCosts.char, modelStr, timeElapsed);
  else
    logStr = sprintf('# eval (%.2f), %d, %d, %.2fK, %.2f, train=(%.2f), valid=(%.2f), test=(%.2f),%s, time=%.2fs', params.testPplWord, ...
      params.epoch, params.iter, params.speed, params.lr, ...
      avgTrainCosts.word, validCosts.word, testCosts.word, modelStr, timeElapsed);
  end
  fprintf(2, '%s\n', logStr);
  fprintf(params.logId, '%s\n', logStr);
  
  params.curValidCost = validCosts.word;
  if params.charOpt > 1
    params.curValidCost = params.curValidCost + validCosts.char;
  end
  
  if params.curValidCost < params.bestCostValid
    params.bestCostValid = params.curValidCost;
    
    fprintf(2, '  save model best valid cost %.2f to %s\n', params.curValidCost, params.modelFile);
    fprintf(params.logId, '  save model best valid cost %.2f to %s\n', params.curValidCost, params.modelFile);
    save(params.modelFile, 'model', 'params');
    
    if params.saveHDF
      saveHDF5([params.modelFile '.h5'], model, params);
    end
  end
end

%% Eval
function [evalCosts, totalNumChars] = evalCost(model, data, params)
  numSents = size(data.tgtInput, 1);
  numBatches = floor((numSents-1)/params.batchSize) + 1;

  evalCosts = initCosts(params);
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
    [costs, ~, charInfo] = lstmCostGrad(model, trainData, params, 1);
    totalNumChars = totalNumChars + charInfo.numChars;
    [evalCosts] = updateCosts(evalCosts, costs);
  end
end
