function [cost_pos, posGrad, trainData] = posSignalCostGrad(model, h_t, tgtPos, curMask, trainData, params, isTest)    
  % h_t -> scales=sigmoid(v_pos*h_pos) in [0, 1]
  [scales, posData] = scaleLayerForward(model.W_pos, model.v_pos, h_t, params);
  
  % positions
  trainData.positions = floor((trainData.srcLens-1).*scales)+1; % scales in (0, 1) srcLen includes <eos>
  
  % backprop: position loss -> h_t
  if isTest==0
    curPosOutput = trainData.posOutput(:, tgtPos)';
    posMask.mask = curMask.mask & curPosOutput~=params.nullPosId & curPosOutput~=params.tgtEos;
    posMask.unmaskedIds = find(posMask.mask);
    posMask.maskedIds = find(~posMask.mask);
    trainData.posMask = posMask;  
    
    % scales -> L2 loss -> scales
    refScales = curPosOutput./(trainData.srcLens-1); % from unsupervised alignments, srcLen includes <eos>
    [cost_pos, grad_scales] = l2CostGrad(scales, refScales, params.posWeight, trainData.posMask.maskedIds, isTest);
  
    [posGrad.ht, posGrad.W_pos, posGrad.v_pos] = scaleLayerBackprop(model.W_pos, model.v_pos, grad_scales, h_t, scales, posData, params);  
  else
    posGrad = [];
    cost_pos = 0.0;
    trainData.posMask = curMask;
  end
  
  % assert
  if params.assert
    assert(isempty(find(trainData.srcLens<=1,1)));
    assert(isempty(find(trainData.positions>(trainData.srcLens-1),1)));
    assert(isempty(find(refScales(trainData.posMask.unmaskedIds)<0 | refScales(trainData.posMask.unmaskedIds)>1, 1))); % we assume that the reference positions are in [0, 1].
    if isTest==0
      assert(sum(sum(abs(posGrad.ht(:, trainData.posMask.maskedIds))))==0);
    end
  end
end