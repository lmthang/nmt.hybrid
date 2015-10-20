function [grad_mu] = distLayerBackprop(grad_distWeights, h2sInfo, params)
  % since linearIdSub is for matrix of size [curBatchSize, numAttnPositions], 
  % we need to transpose grad_alignWeights to be of that size.
  grad_distWeights = grad_distWeights';
  distWeights = h2sInfo.distWeights';
    
  if params.assert
    assert(isequal(size(grad_distWeights), [params.curBatchSize, params.numAttnPositions]));
  end

  % grad_alignWeights -> grad_mu
  % grad_d*d*scaleX/distSigma
  if params.isGPU
    grad_mu = arrayfun(@gradMu, grad_distWeights(h2sInfo.linearIdSub), distWeights(h2sInfo.linearIdSub), h2sInfo.scaleX)/params.distSigma;
  else
   grad_mu = grad_distWeights(h2sInfo.linearIdSub).*distWeights(h2sInfo.linearIdSub).*h2sInfo.scaleX/params.distSigma;
  end
  
  % accumulate grad_mu
  [grad_mu_accum, indices_mu] = aggregateMatrix(grad_mu, h2sInfo.unmaskedIds, params.isGPU, params.dataType);
  grad_mu = zeroMatrix([1, params.curBatchSize], params.isGPU, params.dataType);
  grad_mu(indices_mu) = grad_mu_accum;
end

function [grad_mu] = gradMu(grad_dist, distWeight, scaledX)
  grad_mu = grad_dist*distWeight*scaledX;
end
