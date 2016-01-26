% Use gaussian probabilities to weight src hidden states
function [distWeights, scaleX] = distLayerForward(mu, h2sInfo, params)
  if params.isReverse % get back correct source positions
    srcPositions = params.srcMaxLen - h2sInfo.indicesAll;
  else
    error('attn4 has not supported inverse yet\n');
  end

  scaleX = (srcPositions-mu(h2sInfo.unmaskedIds))/params.distSigma;

  % since linearIdSub is for matrix of size [curBatchSize, numAttnPositions], we need to create alignWeights with this size first
  distWeights = zeroMatrix([params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);

  if params.isGPU
    distWeights(h2sInfo.linearIdSub) = arrayfun(@distFunc, scaleX);
  else
    distWeights(h2sInfo.linearIdSub) = exp(-0.5*scaleX.^2);
  end

  distWeights = distWeights'; % numAttnPositions * curBatchSize
end

% unnormalized standard gaussian distribution
function [prob] = distFunc(scaleX)
  prob = exp(-0.5*scaleX^2);
end
