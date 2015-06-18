% Use gaussian probabilities to weight src hidden states
function [h2sInfo] = gaussLayerForward(mu, h2sInfo, trainData, params)
  if params.isReverse % get back correct source positions
    srcPositions = trainData.srcMaxLen - h2sInfo.indicesAll;
  end

  % since linearIdSub is for matrix of size [curBatchSize, numAttnPositions], we need to create alignWeights with this size first
  h2sInfo.alignWeights = zeroMatrix([trainData.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);

  % for computing the guassian probs faster
  h2sInfo.variances = h2sInfo.origVariances(h2sInfo.unmaskedIds);
  h2sInfo.sigAbs = sqrt(h2sInfo.variances);
  h2sInfo.scaledPositions = (srcPositions-mu(h2sInfo.unmaskedIds))./h2sInfo.sigAbs;
  if params.isGPU
    h2sInfo.alignWeights(h2sInfo.linearIdSub) = arrayfun(@gaussProb, h2sInfo.scaledPositions, sqrt(2*pi)*h2sInfo.sigAbs);
  else
    h2sInfo.alignWeights(h2sInfo.linearIdSub) = exp(-0.5*h2sInfo.scaledPositions.^2)./(params.sqrt2pi*h2sInfo.sigAbs);
  end

  h2sInfo.alignWeights = h2sInfo.alignWeights'; % numAttnPositions * curBatchSize
end

% x is already scaled x = (x_orig - mu)/sigAbs
function [prob] = gaussProb(scaledX, norm)
  %prob = exp(-0.5*(x-mu)^2/variance)/sqrt(2*pi*variance);
  prob = exp(-0.5*scaledX^2)/norm;
end
