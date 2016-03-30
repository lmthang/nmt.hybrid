function [totalCost, grad_W_soft_total, grad_softmax_all, indLosses] = softmaxCostGrad(lstmStates, W_soft, output, masks, params, isTest)
%%%
%
% Compute softmax cost/grad for LSTM. 
%
% If isTest==1, this method only computes cost (for testing purposes).
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%

T = size(output, 2);

% mask
[maskInfos] = prepareMask(masks);

grad_softmax_all = cell(T, 1);
totalCost = 0;
grad_W_soft_total = 0;
indLosses = zeroMatrix([T, 1], params.isGPU, params.dataType);
for tt=1:T % time
  softmax_h = lstmStates{tt}{end}.softmax_h;

  % softmax_h -> loss
  predWords = output(:, tt)';
  [indLosses(tt), probs, scores, scoreIndices] = softmaxLayerForward(W_soft, softmax_h, predWords, maskInfos{tt});
  totalCost = totalCost + indLosses(tt);

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
    assert(computeSum(scores(:, maskInfos{tt}.maskedIds), params.isGPU)==0);
    if isTest==0
      assert(computeSum(grad_softmax_all{tt}(:, maskInfos{tt}.maskedIds), params.isGPU)==0);
    end
  end
end
