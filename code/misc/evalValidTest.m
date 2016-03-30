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
  
  unkStats = '';
  if params.charOpt > 0
    unkStats = sprintf(' src unk types %.2f, src unk tokens %.2f, tgt unk tokens %.2f,', params.trainNumSrcUnkTypes/params.trainNumSents, ...
       params.trainNumSrcUnkTokens/params.trainNumSents, params.trainNumTgtUnkTokens/params.trainNumSents);
  end
    
  if params.charTgtGen
    params.testPplChar = exp(testCosts.char/params.charWeight);
    logStr = sprintf('# eval (%.2f, %.2f), %d, %d, %.2fK, %.2f (%.2f, %.2f),%s train=(%.2f, %.2f), valid=(%.2f, %.2f), test=(%.2f, %.2f),%s, time=%.2fs', ...
      params.testPplWord, params.testPplChar, ...
      params.epoch, params.iter, params.speed, params.lr, ...
      avgTrainCosts.word, avgTrainCosts.char, unkStats, ...
      validCosts.word, validCosts.char, testCosts.word, testCosts.char, modelStr, timeElapsed);
  else
    logStr = sprintf('# eval (%.2f), %d, %d, %.2fK, %.2f,%s train=(%.2f), valid=(%.2f), test=(%.2f),%s, time=%.2fs', params.testPplWord, ...
      params.epoch, params.iter, params.speed, params.lr, unkStats, ...
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
