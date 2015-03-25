function [params] = evalValidTest(model, validData, testData, params)
  startTime = clock;
  [costValid] = evalCost(model, validData, params); % inputValid, inputValidMask, tgtValidOutput, tgtValidMask, srcValidMaxLen, tgtValidMaxLen, params);
  [costTest] = evalCost(model, testData, params); %inputTest, inputTestMask, tgtTestOutput, tgtTestMask, srcTestMaxLen, tgtTestMaxLen, params);
  
  costValid.total = costValid.total/validData.numWords;
  costTest.total = costTest.total/testData.numWords;
  modelStr = wInfo(model);
  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  if params.posModel>=0 % positional model
    costValid.pos = costValid.pos*2/validData.numWords;
    costValid.word = costValid.word*2/validData.numWords;
    costTest.pos = costTest.pos*2/testData.numWords;
    costTest.word = costTest.word*2/testData.numWords;
    fprintf(2, '# eval %.2f (%.2f, %.2f), %d, %d, %.2fK, %.2f, train=%.4f (%.2f, %.2f), valid=%.4f (%.2f, %.2f), test=%.4f (%.2f, %.2f),%s, time=%.2fs\n', exp(costTest.total), exp(costTest.pos), exp(costTest.word), params.epoch, params.iter, params.speed, params.lr, params.costTrain, params.costTrainPos, params.costTrainWord, costValid.total, costValid.pos, costValid.word, costTest.total, costTest.pos, costTest.word, modelStr, timeElapsed);
    fprintf(params.logId, '# eval %.2f (%.2f, %.2f), %d, %d, %.2fK, %.2f, train=%.4f (%.2f, %.2f), valid=%.4f (%.2f, %.2f), test=%.4f (%.2f, %.2f),%s, time=%.2fs\n', exp(costTest.total), exp(costTest.pos), exp(costTest.word), params.epoch, params.iter, params.speed, params.lr, params.costTrain, params.costTrainPos, params.costTrainWord, costValid.total, costValid.pos, costValid.word, costTest.total, costTest.pos, costTest.word, modelStr, timeElapsed);
  else
    fprintf(2, '# eval %.2f, %d, %d, %.2fK, %.2f, train=%.4f, valid=%.4f, test=%.4f, %s, time=%.2fs\n', exp(costTest.total), params.epoch, params.iter, params.speed, params.lr, params.costTrain, costValid.total, costTest.total, modelStr, timeElapsed);
    fprintf(params.logId, '# eval %.2f, %d, %d, %.2fK, %.2f, train=%.4f, valid=%.4f, test=%.4f, %s, time=%.2fs\n', exp(costTest.total), params.epoch, params.iter, params.speed, params.lr, params.costTrain, costValid.total, costTest.total, modelStr, timeElapsed);
  end
    
  params.curTestPerplexity = exp(costTest.total);
  if params.posModel>0 % positional model
    params.curTestPerplexityPos = exp(costTest.pos);
    params.curTestPerplexityWord = exp(costTest.pos);
  end
  if costValid.total < params.bestCostValid
    params.bestCostValid = costValid.total;
    params.costTest = costTest.total;
    params.testPerplexity = params.curTestPerplexity;
    if params.posModel>0 % positional model
      params.bestCostValidPos = costValid.pos;
      params.bestCostValidWord = costValid.word;
      params.testPerplexityPos = params.curTestPerplexityPos;
      params.testPerplexityWord = params.curTestPerplexityWord;
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

  evalCosts.total = 0;
  if params.posModel>=0 % positional model
    evalCosts.pos = 0;
    evalCosts.word = 0;
  end
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
    trainData.tgtMask = data.tgtMask(startId:endId, :);
    trainData.tgtOutput = data.tgtOutput(startId:endId, :);
    trainData.srcLens = data.srcLens(startId:endId);
    if params.posModel>0 % positional model
      trainData.srcPos = data.srcPos(startId:endId, :);
    end
    
    % eval
    costs = lstmCostGrad(model, trainData, params, 1);
    evalCosts.total = evalCosts.total + costs.total;
    if params.posModel>=0 % positional model
      evalCosts.pos = evalCosts.pos + costs.pos;
      evalCosts.word = evalCosts.word + costs.word;
    end
    
  end
end
