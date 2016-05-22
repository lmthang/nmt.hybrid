%% Eval
function [evalCosts, totalNumChars] = evalCost(model, data, params)
  numBatches = data.numBatches;
  evalCosts = initCosts(params);
  
  totalNumChars = 0;
  for batchId = 1 : numBatches 
    % eval
    [costs, ~, charInfo] = lstmCostGrad(model, data.batches{batchId}, params, 1);
    totalNumChars = totalNumChars + charInfo.numChars;
    [evalCosts] = updateCosts(evalCosts, costs);
  end
end
