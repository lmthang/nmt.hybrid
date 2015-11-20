function [totalCost, grad_W_soft_total, grad_softmax_all] = softmaxCostGrad(T, lstmStates, attnInfos, W_soft, tgtOutput, maskInfo, params, isTest)
%%%
%
% Compute softmax cost/grad for LSTM. 
%
% If isTest==1, this method only computes cost (for testing purposes).
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%

grad_softmax_all = cell(T, 1);
totalCost = 0;
for tt=1:T % time
  % h_t -> softmax_h
  if params.attnFunc
    softmax_h = attnInfos{tt}.softmax_h;
  else
    softmax_h = lstmStates{tt}{params.numLayers}.h_t;
  end

  % softmax_h -> loss
  predWords = tgtOutput(:, tt)';
  [cost, probs, scores, scoreIndices] = softmaxLayerForward(W_soft, softmax_h, predWords, maskInfo{tt});
  totalCost = totalCost + cost;

  % backprop: loss -> softmax_h
  if isTest==0
    % loss -> softmax_h
    [grad_W_soft, grad_softmax_all{tt}] = softmaxLayerBackprop(W_soft, softmax_h, probs, scoreIndices);
    
    if tt==1
      grad_W_soft_total = grad_W_soft;
    else
      grad_W_soft_total = grad_W_soft_total + grad_W_soft;
    end
  end

  % assert
  if params.assert
    assert(computeSum(scores(:, maskInfo{tt}.maskedIds), params.isGPU)==0);
    if isTest==0
      assert(computeSum(grad_softmax_all{tt}(:, maskInfo{tt}.maskedIds), params.isGPU)==0);
    end
  end
end
