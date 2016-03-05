%% Eval
function [evalCosts] = evalCost(model, data, params) %input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen, params)
  numSents = size(data.tgtInput, 1);
  numBatches = floor((numSents-1)/params.batchSize) + 1;

  [evalCosts] = initCosts();
  trainData.srcMaxLen = data.srcMaxLen;
  trainData.tgtMaxLen = data.tgtMaxLen;
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
    costs = lstmCostGrad(model, trainData, params, 1);
    [evalCosts] = updateCosts(evalCosts, costs);
  end
end