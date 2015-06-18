function [grad_mu, grad_variances] = gaussLayerBackprop(grad_alignWeights, h2sInfo, params)
  if params.assert
    assert(isequal(size(grad_alignWeights), [params.curBatchSize, params.numAttnPositions]));
  end

  % grad_alignWeights -> grad_variances
  % 0.5*p*(scaleX^2/variance - 1/sigAbs)
  if params.isGPU
    grad_variances = arrayfun(@gradSigSquare, grad_alignWeights(h2sInfo.linearIdSub), ...
      h2sInfo.alignWeights(h2sInfo.linearIdSub), h2sInfo.scaledPositions, h2sInfo.variances, h2sInfo.sigAbs);
  else
    grad_variances = 0.5*grad_alignWeights(h2sInfo.linearIdSub).*h2sInfo.alignWeights(h2sInfo.linearIdSub).*...
      (h2sInfo.scaledPositions.^2./h2sInfo.variances-1./h2sInfo.sigAbs);
  end

  % grad_alignWeights -> grad_mu
  % p*scaleX/sigAbs
  if params.isGPU
    grad_mu = arrayfun(@gradMu, grad_alignWeights(h2sInfo.linearIdSub), h2sInfo.alignWeights(h2sInfo.linearIdSub), ...
      h2sInfo.scaledPositions, h2sInfo.sigAbs);
  else
    grad_mu = grad_alignWeights(h2sInfo.linearIdSub).*h2sInfo.alignWeights(h2sInfo.linearIdSub).*h2sInfo.scaledPositions./h2sInfo.sigAbs;
  end

  % accumulate grad_variances
  [grad_variances_accum, indices_variances] = aggregateMatrix(grad_variances, h2sInfo.unmaskedIds, params.isGPU, params.dataType);
  grad_variances = zeroMatrix([1, params.curBatchSize], params.isGPU, params.dataType);
  grad_variances(indices_variances) = grad_variances_accum;
  
  % accumulate grad_mu
  [grad_mu_accum, indices_mu] = aggregateMatrix(grad_mu, h2sInfo.unmaskedIds, params.isGPU, params.dataType);
  grad_mu = zeroMatrix([1, params.curBatchSize], params.isGPU, params.dataType);
  grad_mu(indices_mu) = grad_mu_accum;
end

function [grad_variance] = gradSigSquare(grad_align, alignWeight, scaledX, variance, sigAbs)
  grad_variance = 0.5*grad_align*alignWeight*(scaledX^2/variance-1/sigAbs);
end

function [grad_mu] = gradMu(grad_align, alignWeight, scaledX, sigAbs)
  grad_mu = grad_align*alignWeight*scaledX/sigAbs;
end