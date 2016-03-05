function [params] = evalValidTest(model, validData, testData, params)
  startTime = clock;
  [validCosts] = evalCost(model, validData, params);
  [testCosts] = evalCost(model, testData, params);
  
  validCounts = initCosts();
  validCounts = updateCounts(validCounts, validData);
  validCosts = scaleCosts(validCosts, validCounts);
  
  testCounts = initCosts();
  testCounts = updateCounts(testCounts, testData);
  testCosts = scaleCosts(testCosts, testCounts);
  
  modelStr = wInfo(model);
  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  
  params.curTestPerpWord = exp(testCosts.word);
  [params.scaleTrainCosts] = scaleCosts(params.trainCosts, params.trainCounts);
  logStr = sprintf('# eval %.2f, %d, %d, %.2fK, %.2f, train=%.2f, valid=%.2f, test=%.2f,%s, time=%.2fs', params.curTestPerpWord, ...
    params.epoch, params.iter, params.speed, params.lr, ...
    params.scaleTrainCosts.total, validCosts.total, testCosts.total, modelStr, timeElapsed);
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
